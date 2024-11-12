# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools
import pickle
import os
import torch
import numpy as np
import abc

from score_sde_spatiotemporal.utils import (
    from_flattened_numpy,
    to_flattened_numpy,
    get_score_fn,
)
from scipy import integrate
from score_sde_spatiotemporal import sde_lib
from score_sde_spatiotemporal import utils as mutils
import torch.distributed as dist
from lib import sde_lib
from lib.metrics_agcrn import All_Metrics
import wandb

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, eps, device):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == "ode":
        sampling_fn = get_ode_sampler(
            sde=sde,
            shape=shape,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
        )
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == "pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=True,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=device,
        )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, guider, adj):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t, guider, adj)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, guider, adj, args, st_version=True):
        f, G = self.rsde.discretize(x, t, guider, adj, args, st_version)
        z = torch.randn_like(x)
        x_mean = x - f
        G = G[:, None, None, None] if len(G.shape) == 1 else G
        x = x_mean + G * z
        return x, x_mean


@register_predictor(name="ancestral_sampling")
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        assert (
            not probability_flow
        ), "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            sde.discrete_sigmas.to(t.device)[timestep - 1],
        )
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma**2 - adjacent_sigma**2)[:, None, None, None]
        std = torch.sqrt(
            (adjacent_sigma**2 * (sigma**2 - adjacent_sigma**2)) / (sigma**2)
        )
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1.0 - beta)[
            :, None, None, None
        ]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, guider, adj):
        return x, x


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
            and not isinstance(sde, sde_lib.stVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t, guider, adj):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t) + guider(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t, guider, adj):
        return x, x


def shared_predictor_update_fn(
    x,
    t,
    sde,
    model,
    guider,
    adj,
    cond,
    pos_w,
    pos_d,
    predictor,
    probability_flow,
    continuous,
    args,
    st_version=True,
):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = get_score_fn(
        sde, model, adj, cond, pos_w, pos_d, train=False, continuous=continuous
    )
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, guider, adj, args, st_version)


def shared_corrector_update_fn(
    x,
    t,
    sde,
    model,
    guider,
    adj,
    cond,
    pos_w,
    pos_d,
    corrector,
    continuous,
    snr,
    n_steps,
):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(
        sde, model, adj, cond, pos_w, pos_d, train=False, continuous=continuous
    )
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t, guider, adj)


def get_pc_sampler(
    sde,
    shape,
    predictor,
    corrector,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )

    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_sampler(
        model,
        guider,
        adj,
        cond,
        pos_w,
        pos_d,
        history,
        args,
        self,
        inverse_transform,
        label,
        batch_index,
        all_batches,
    ):
        """The PC sampler function.

        Args:
        model: A score model.
        Returns:
        Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape, args.n_samples).to(device)
            x = x.reshape(shape[0] * args.n_samples, shape[1], shape[2], shape[3])
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
            is_distributed = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if is_distributed else 0

            best_mae = float("inf")
            best_mis = float("inf")
            best_predictions = None
            best_metrics = None  # To store the metrics for the best MAE iteration
            no_improvement_counter = (
                0  # Counter for consecutive non-improving iterations
            )
            no_mis_improvement_counter = (
                0  # Counter for consecutive non-improving MIS iterations
            )
            st_false_counter = 0  # Counter for consecutive ST_version=False iterations
            mae_patience = 30  # Number of iterations to wait for MAE improvement
            mis_patience = 400  # Number of iterations to wait for MIS improvement
            stop_signal = torch.tensor(0, device=device)  # Stop signal tensor

            # List to store predictions and metrics for each iteration
            all_predictions = []

            for i in range(sde.N):
                if stop_signal.item() > 0:  # Check if stop signal is set
                    break

                t = timesteps[i]
                vec_t = torch.ones(shape[0] * args.n_samples, device=t.device) * t

                # Predictor update with st_version=False
                x_st_false, x_mean_st_false = predictor_update_fn(
                    x,
                    vec_t,
                    model=model,
                    guider=guider,
                    adj=adj,
                    cond=cond,
                    pos_w=pos_w,
                    pos_d=pos_d,
                    args=args,
                    st_version=False,
                )

                # Predictor update with st_version=True
                x_st_true, x_mean_st_true = predictor_update_fn(
                    x,
                    vec_t,
                    model=model,
                    guider=guider,
                    adj=adj,
                    cond=cond,
                    pos_w=pos_w,
                    pos_d=pos_d,
                    args=args,
                    st_version=True,
                )

                # Compute MAE for both versions
                predictions_st_false = x_mean_st_false if denoise else x_st_false
                predictions_st_true = x_mean_st_true if denoise else x_st_true

                predictions_samples_st_false = (
                    predictions_st_false.reshape(
                        -1, args.n_samples, *predictions_st_false.shape[1:]
                    )[:, :, :, :, -args.dataset.dif_model.AGCRN.T_p :]
                    .permute(0, 1, 4, 3, 2)
                    .contiguous()
                )
                predictions_samples_st_true = (
                    predictions_st_true.reshape(
                        -1, args.n_samples, *predictions_st_true.shape[1:]
                    )[:, :, :, :, -args.dataset.dif_model.AGCRN.T_p :]
                    .permute(0, 1, 4, 3, 2)
                    .contiguous()
                )

                # Inverse transform
                predictions_inv_st_false, labels_inv = inverse_transform(
                    predictions_samples_st_false, label
                )
                predictions_inv_st_true, _ = inverse_transform(
                    predictions_samples_st_true, label
                )

                predictions_det_st_false = torch.mean(
                    predictions_inv_st_false, dim=1
                )  # (B, T, N, D)
                predictions_det_st_true = torch.mean(
                    predictions_inv_st_true, dim=1
                )  # (B, T, N, D)

                (
                    mae_st_false,
                    rmse_st_false,
                    mape_st_false,
                    crps_st_false,
                    mis_st_false,
                ) = All_Metrics(
                    predictions_det_st_false,
                    predictions_inv_st_false,
                    labels_inv,
                    None,
                    0,
                )
                mae_st_true, rmse_st_true, mape_st_true, crps_st_true, mis_st_true = (
                    All_Metrics(
                        predictions_det_st_true,
                        predictions_inv_st_true,
                        labels_inv,
                        None,
                        0,
                    )
                )

                # Decide which MAE to use
                if mae_st_false < mae_st_true or args.dataset.neighbors_sum_c == 0:
                    ST_version = False
                    selected_mae = mae_st_false
                    selected_predictions = predictions_st_false
                    selected_x = x_st_false
                    selected_x_mean = x_mean_st_false
                    selected_rmse = rmse_st_false
                    selected_mape = mape_st_false
                    selected_crps = crps_st_false
                    selected_mis = mis_st_false
                    if i < 20:
                        st_false_counter += 1  # Increment counter for ST_version=False
                else:
                    ST_version = True
                    selected_mae = mae_st_true
                    selected_predictions = predictions_st_true
                    selected_x = x_st_true
                    selected_x_mean = x_mean_st_true
                    selected_rmse = rmse_st_true
                    selected_mape = mape_st_true
                    selected_crps = crps_st_true
                    selected_mis = mis_st_true
                    st_false_counter = 0  # Reset counter if ST_version=True

                x = selected_x

                current_metrics = {
                    "MAE": selected_mae,
                    "RMSE": selected_rmse,
                    "MAPE": selected_mape,
                    "CRPS": selected_crps,
                    "MIS": selected_mis,
                }

                if selected_mae < best_mae:
                    best_mae = selected_mae
                    best_predictions = selected_predictions
                    best_iteration = i + 1
                    best_timestep = t.item()
                    best_metrics = current_metrics
                    no_improvement_counter = (
                        0  # Reset counter if there is an improvement
                    )
                else:
                    no_improvement_counter += 1  # Increment counter if no improvement

                # Save the predictions and metrics for this iteration
                all_predictions.append(
                    {
                        "iteration": i + 1,
                        "timestep": t.item(),
                        "predictions": selected_predictions.cpu().numpy(),
                        "metrics": current_metrics,
                        "best_predictions": (
                            best_predictions.cpu().numpy()
                            if best_predictions is not None
                            else None
                        ),
                        "best_metrics": best_metrics,
                    }
                )

                if selected_mis < best_mis:
                    best_mis = selected_mis
                    no_mis_improvement_counter = (
                        0  # Reset counter if there is an improvement in MIS
                    )
                else:
                    no_mis_improvement_counter += (
                        1  # Increment counter if no improvement in MIS
                    )

                # dist.barrier() if args.use_distributed else None

                if i < 20 and st_false_counter >= 14:
                    st_false_ranks = torch.tensor(1, device=device)
                else:
                    st_false_ranks = torch.tensor(0, device=device)

                if ST_version:
                    self.logger.info(
                        f"Rank {rank} has ST_version=True for iteration {i+1}."
                    )

                if args.use_distributed:
                    dist.all_reduce(st_false_ranks, op=dist.ReduceOp.SUM)
                    if st_false_ranks.item() >= ((dist.get_world_size() / 2) + 1):
                        reason = f"ST_version=False for 14 consecutive iterations in the first 20 iterations across at least {((dist.get_world_size()/2)+1)} ranks"
                        self.logger.info(
                            f"Stopping early at iteration {i+1} for rank {rank} due to {reason}."
                        )
                        stop_signal.fill_(1)  # Set stop signal to 1

                # Reduce the stop signals to rank 0
                (
                    dist.reduce(stop_signal, dst=0, op=dist.ReduceOp.SUM)
                    if args.use_distributed
                    else None
                )

                # Rank 0 checks if any rank set the stop signal and broadcasts it
                if rank == 0 and stop_signal.item() > 0:
                    stop_signal.fill_(1)
                    self.logger.info(f"Rank {rank} broadcasting stop signal.")
                    (
                        dist.broadcast(stop_signal, src=0)
                        if args.use_distributed
                        else None
                    )  # Broadcast stop signal to all ranks

                # dist.broadcast(stop_signal, src=0)

                # dist.barrier() if args.use_distributed else None
                # self.logger.info(f"Rank {rank} has {stop_signal.item()} stop signal.")
                if stop_signal.item() > 0:
                    self.logger.info(f"Rank {rank} breaking the loop for ST VERSION.")
                    break

                # dist.barrier() if args.use_distributed else None
                if (no_improvement_counter > mae_patience and i <= 500) or (
                    no_mis_improvement_counter > mis_patience and i <= 500
                ):
                    reason = "MAE" if no_improvement_counter > mae_patience else "MIS"
                    self.logger.info(
                        f"Stopping early at iteration {i+1} for rank {rank} due to no improvement in {reason} for {mae_patience if reason == 'MAE' else mis_patience} consecutive iterations."
                    )
                    stop_signal.fill_(1)  # Set stop signal to 1

                # Reduce the stop signals to rank 0
                (
                    dist.reduce(stop_signal, dst=0, op=dist.ReduceOp.SUM)
                    if args.use_distributed
                    else None
                )

                # Rank 0 checks if any rank set the stop signal and broadcasts it
                if rank == 0 and stop_signal.item() > 0:
                    stop_signal.fill_(1)
                    self.logger.info(f"Rank {rank} broadcasting stop signal.")
                    (
                        dist.broadcast(stop_signal, src=0)
                        if args.use_distributed
                        else None
                    )  # Broadcast stop signal to all ranks

                dist.broadcast(stop_signal, src=0) if args.use_distributed else None

                if stop_signal.item() > 0:
                    self.logger.info(f"Rank {rank} breaking the loop for MAE.")
                    break

                if rank == 0:
                    self.logger.info(
                        f"Batch {batch_index + 1}/{all_batches}, "
                        f"Iteration {i+1}/{sde.N}, ST_version: {ST_version}, "
                        f"MAE: {selected_mae:.4f}, RMSE: {selected_rmse:.4f}, MAPE: {selected_mape:.4f}, CRPS: {selected_crps:.4f}, MIS: {selected_mis:.4f}"
                    )

            if rank == 0:
                self.logger.info(
                    f"Best MAE for Rank 0: {best_mae:.4f} at iteration {best_iteration}/{sde.N}, timestep: {best_timestep:.4f} in batch {batch_index + 1}/{args.num_batches_to_process}. "
                    f"Best Metrics for Rank 0: MAE: {best_metrics['MAE']:.4f}, RMSE: {best_metrics['RMSE']:.4f}, "
                    f"MAPE: {best_metrics['MAPE']:.4f}, CRPS: {best_metrics['CRPS']:.4f}, MIS: {best_metrics['MIS']:.4f}"
                )
                self.logger.info(
                    f"Last Metrics for Rank 0: MAE: {current_metrics['MAE']:.4f}, RMSE: {current_metrics['RMSE']:.4f}, "
                    f"MAPE: {current_metrics['MAPE']:.4f}, CRPS: {current_metrics['CRPS']:.4f}, MIS: {current_metrics['MIS']:.4f}"
                )

                # Save all predictions and best metrics to a file
                save_data = {
                    "all_predictions": all_predictions,
                }
                with open(
                    os.path.join(
                        self.args.log_dir,
                        f"predictions_and_metrics_batch_{batch_index + 1}.pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(save_data, f)

            # dist.barrier() if args.use_distributed else None
            # self.logger.info(f"Rank {rank} has passed the final barrier.")
            return selected_predictions, best_predictions

    return pc_sampler


def get_ode_sampler(
    sde,
    shape,
    inverse_scaler,
    denoise=False,
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    eps=1e-3,
    device="cuda",
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      inverse_scaler: The inverse data normalizer.
      denoise: If `True`, add one-step denoising to final samples.
      rtol: A `float` number. The relative tolerance level of the ODE solver.
      atol: A `float` number. The absolute tolerance level of the ODE solver.
      method: A `str`. The algorithm used for the black-box ODE solver.
        See the documentation of `scipy.integrate.solve_ivp`.
      eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
          model: A score model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func,
                (sde.T, eps),
                to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method=method,
            )
            nfe = solution.nfev
            x = (
                torch.tensor(solution.y[:, -1])
                .reshape(shape)
                .to(device)
                .type(torch.float32)
            )

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
