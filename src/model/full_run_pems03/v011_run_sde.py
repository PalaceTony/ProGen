import torch
import hydra
import logging
import os
import time
import torch.distributed as dist
from datetime import datetime
import numpy as np
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf
import traceback

from score_sde_spatiotemporal.ddpm import DDPM
from v111_trainer_sde import Trainer
from lib.utils import (
    seed_everything,
    set_local_rank,
    setup_logging,
    write_all_to_excel,
    get_adjacency_matrix2,
    write_all_to_sqlite,
    collect_code_files,
)
from hydra.core.hydra_config import HydraConfig

from lib.dataloader import get_dataloader
from lib import sde_lib
from lib.losses import optimization_manager, get_step_fn
from lib.ema import ExponentialMovingAverage
from lib.sampling import get_sampling_fn
from collections import OrderedDict
from datetime import timedelta

from AGCRN import AGCRN
from guider_ugnet.score_sde_spatiotemporal.ddpm_guider import DDPM as GUIDER
from optuna.trial import TrialState

import optuna
import math


def objective(trial, args):
    trial = (
        optuna.integration.TorchDistributedTrial(trial)
        if args.use_distributed and args.hyperopt
        else trial
    )
    seed_everything(args.seed)
    if args.use_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        torch.cuda.set_device(local_rank)

    # Optuna parameters
    def encode_tuple(t):
        return str(t)

    def decode_tuple(s):
        return eval(s)

    args.dataset.lr_init = (
        trial.suggest_float("lr_init", 1e-4, 1e-2, log=True)
        if args.hyperopt
        else args.dataset.lr_init
    )
    args.dataset.dif_model.model.nf = (
        trial.suggest_categorical("nf", [16, 32, 64, 128, 256])
        if args.hyperopt
        else args.dataset.dif_model.model.nf
    )
    args.dataset.dif_model.stlayer.hidden_size = (
        int(trial.suggest_int("hidden_size", 16, 256, step=1))
        if args.hyperopt
        else args.dataset.dif_model.stlayer.hidden_size
    )
    args.dataset.dif_model.model.pos_emb = (
        trial.suggest_categorical("pos_emb", [8, 16, 32, 64, 128])
        if args.hyperopt
        else args.dataset.dif_model.model.pos_emb
    )
    args.dataset.dif_model.model.num_res_blocks = (
        trial.suggest_categorical("num_res_blocks", [1, 2, 3, 4, 5])
        if args.hyperopt
        else args.dataset.dif_model.model.num_res_blocks
    )
    args.dataset.dif_model.model.ch_mult = (
        decode_tuple(
            trial.suggest_categorical(
                "ch_mult",
                [
                    encode_tuple((1, 2)),
                    encode_tuple((1, 2, 3)),
                    encode_tuple((1, 2, 3, 4)),
                    encode_tuple((1, 2, 2)),
                ],
            )
        )
        if args.hyperopt
        else args.dataset.dif_model.model.ch_mult
    )
    args.dataset.column_wise = (
        trial.suggest_categorical("column_wise", [True, False])
        if args.hyperopt
        else args.dataset.column_wise
    )
    args.notes = f"{args.excel_notes}, args.dataset.lr_init: {args.dataset.lr_init}, args.dataset.dif_model.model.nf: {args.dataset.dif_model.model.nf}, args.dataset.dif_model.stlayer.hidden_size: {args.dataset.dif_model.stlayer.hidden_size}, args.dataset.dif_model.model.pos_emb: {args.dataset.dif_model.model.pos_emb}, args.dataset.dif_model.model.num_res_blocks: {args.dataset.dif_model.model.num_res_blocks}, args.dataset.dif_model.model.ch_mult: {args.dataset.dif_model.model.ch_mult}, args.dataset.column_wise: {args.dataset.column_wise}"

    if args.use_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device if torch.cuda.is_available() else "cpu"
        local_rank = 0

    if local_rank == 0 and args.hyperopt:
        logger.info(f"Current parameters: {trial.params}")

    args.device = device_str
    logging.info(f"Using device: {args.device}")
    device = torch.device(args.device)

    # Adjust the current time by adding 8 hours
    current_time_adjusted = datetime.now() + timedelta(hours=8)

    args.unique_run_id = (
        (f"trial_{trial.number}_{current_time_adjusted.strftime('%m%d_%H_%M_%S')}")
        if args.hyperopt
        else f"{current_time_adjusted.strftime('%m%d_%H_%M_%S')}"
    )

    unique_run_path = os.path.join(args.log_dir, args.unique_run_id)
    os.makedirs(unique_run_path, exist_ok=True) if local_rank == 0 else None
    args.best_model_path = (
        unique_run_path if not args.test_only else args.best_model_path
    )
    args.notes = f"{args.unique_run_id}, {args.notes}"
    if local_rank == 0:
        if args.excel_notes == "DEBUG":
            args.db_path = "run_logs_debug.db"
        elif args.log_to_debug_file:
            args.db_path = "run_logs_debug.db"
        else:
            args.db_path = "run_logs.db"

        write_all_to_sqlite(
            notes=args.notes,
            log_dir=f"{args.log_dir}/v011_run_sde.log",
            db_path=args.db_path,
        )

    # guider
    if args.guider_model is not None:
        guider = GUIDER(args.guider.dif_model).to(device)
        dif_state_dict = torch.load(args.guider_model)
        new_state_dict = OrderedDict()
        for k, v in dif_state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        guider.load_state_dict(new_state_dict)
        for param in guider.parameters():
            param.requires_grad = False
    else:
        guider = 0

    adj = get_adjacency_matrix2(
        distance_df_filename=args.dataset.adj,
        num_of_vertices=args.dataset.dif_model.V,
        type_="connectivity",
        id_filename=args.dataset.id_filename if args.dataset.name == "PEMS03" else None,
    )

    # Model
    model = DDPM(args.dataset.dif_model, adj).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    model = DDP(model, device_ids=[local_rank]) if args.use_distributed else model

    if local_rank == 0:
        logging.info(
            f"Total number of parameters: {sum(p.numel() for p in model.parameters())}"
        )

    ema = ExponentialMovingAverage(
        model.parameters(), decay=args.dataset.dif_model.model.ema_rate
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.dataset.lr_init,
        betas=(args.dataset.dif_model.model.beta1, 0.999),
        eps=args.dataset.dif_model.model.eps,
        weight_decay=args.dataset.dif_model.model.weight_decay,
    )

    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    optimize_fn = optimization_manager(args)

    if args.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=args.dataset.dif_model.model.beta_min,
            beta_max=args.dataset.dif_model.model.beta_max,
            N=args.dataset.dif_model.model.num_scales,
        )
        sampling_eps = 1e-3
    elif args.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=args.dataset.dif_model.model.beta_min,
            beta_max=args.dataset.dif_model.model.beta_max,
            N=args.dataset.dif_model.model.num_scales,
        )
        sampling_eps = 1e-3
    elif args.sde.lower() == "stvpsde":
        sde = sde_lib.stVPSDE(
            beta_min=args.dataset.dif_model.model.beta_min,
            beta_max=args.dataset.dif_model.model.beta_max,
            N=args.dataset.dif_model.model.num_scales,
        )
        sampling_eps = 1e-3
    elif args.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=args.dataset.dif_model.model.sigma_min,
            sigma_max=args.dataset.dif_model.model.sigma_max,
            N=args.dataset.dif_model.model.num_scales,
        )
        sampling_eps = 1e-5
    continuous = True
    reduce_mean = True
    likelihood_weighting = False

    train_step_fn = get_step_fn(
        sde,
        train=True,
        args=args,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )
    eval_step_fn = get_step_fn(
        sde,
        train=False,
        args=args,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )
    sampling_fn = get_sampling_fn(
        args,
        sde,
        (
            args.sampling_first_batch,
            1,
            args.dataset.dif_model.V,
            args.dataset.dif_model.model.shape,
        ),
        sampling_eps,
        device,
    )

    seed_everything(args.seed)

    # Data loading
    (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        train_sampler,
        val_sampler,
        _,
        original_data,
    ) = get_dataloader(
        args,
        local_rank,
        normalizer=args.dataset.normalizer,
        single=False,
        use_distributed=args.use_distributed,
    )

    # Wandb setup
    if local_rank == 0:
        script_dir = os.path.abspath(os.path.dirname(__file__))
        code_dir = os.path.join(args.log_dir, "code")
        collect_code_files(script_dir, code_dir)

        wandb_dir = args.wandb_dir
        os.environ["WANDB_MODE"] = args.wandb_mode
        wandb.init(
            project=args.wandb_project_name if not args.test_only else "sampling",
            name=args.run_name,
            dir=wandb_dir,
            notes=args.notes,
            settings=wandb.Settings(code_dir=wandb_dir),
            save_code=True,
        )
    try:
        trainer = Trainer(
            model,
            ema,
            optimize_fn,
            guider,
            state,
            train_step_fn,
            eval_step_fn,
            sampling_fn,
            train_loader,
            val_loader,
            test_loader,
            original_data,
            scaler,
            train_sampler,
            val_sampler,
            device,
            local_rank,
            args,
            logger,
            adj,
            sde,
        )
        if args.test_only:
            mae = trainer.sampling()
            if math.isnan(mae):
                mae = 9999999999
            result = mae
        else:
            best_val_loss, best_model_path = trainer.train()
            result = best_val_loss
    except Exception as e:
        result = 9999999999
        logging.error(f"Error: {e}")
        logging.error(traceback.format_exc())

    # Finish Wandb
    if local_rank == 0:
        wandb.run.finish()

    return result


@hydra.main(
    version_base=None,
    config_path="../../configuration/modules/clean",
    config_name="DSTGCRN",
)
def main(args: DictConfig) -> None:
    if args.use_distributed:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=1800),
        )
        rank = dist.get_rank()
    else:
        rank = 0
    study = None
    if args.hyperopt and rank == 0:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, args), n_trials=args.hyperopt_trial
        )
    else:
        for _ in range(args.hyperopt_trial if args.hyperopt else 1):
            try:
                result = objective(None, args)
            except optuna.TrialPruned:
                pass

    if rank == 0:
        if study is not None:
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(
                deepcopy=False, states=[TrialState.COMPLETE]
            )

            logger.info("Study statistics: ")
            logger.info(f"Number of finished trials: {len(study.trials)}")
            logger.info(f"Number of pruned trials: {len(pruned_trials)}")
            logger.info(f"Number of complete trials: {len(complete_trials)}")

            logger.info("Best trial:")
            trial = study.best_trial

            logger.info(f"Value: {trial.value}")

            logger.info("  Params: ")
            for key, value in trial.params.items():
                logger.info("    {}: {}".format(key, value))


if __name__ == "__main__":

    setup_logging()
    logger = logging.getLogger("score_sde")
    main()
