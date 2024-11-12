import torch
import os
import time
import wandb
import os
import torch
import torch.utils.data
import torch.distributed as dist
from lib import sde_lib
from lib.sampling import get_sampling_fn
import numpy as np

import torch.nn.functional as F

from lib.utils import (
    save_checkpoint,
    restore_checkpoint,
    write_metrics_to_excel,
    write_train_metrics_to_excel,
    write_all_to_excel,
    write_all_to_sqlite,
)
from lib.metrics_agcrn import All_Metrics
from lib.utils import seed_everything


class Trainer(object):
    def __init__(
        self,
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
    ):
        super(Trainer, self).__init__()
        self.dif_model = model
        self.ema = ema
        self.optimize_fn = optimize_fn
        self.state = state
        self.train_step_fn = train_step_fn
        self.eval_step_fn = eval_step_fn
        self.sampling_fn = sampling_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.device = device
        self.local_rank = local_rank
        self.args = args
        self.best_path = os.path.join(
            self.args.best_model_path, self.args.best_model_name
        )
        self.logger = logger
        if self.args.num_batches_to_process == "inf":
            self.args.num_batches_to_process = float("inf")
        elif self.args.num_batches_to_process == float("inf"):
            self.args.num_batches_to_process = float("inf")
        else:
            self.args.num_batches_to_process = int(self.args.num_batches_to_process)
        self.guider = guider
        self.adj = adj
        self.sde = sde
        self.original_data = original_data

    def train_epoch(self, epoch):
        self.dif_model.train()
        total_loss = 0
        num_batches = (
            self.args.num_batches_to_process
            if self.args.num_batches_to_process != float("inf")
            else len(self.train_loader)
        )
        seed_everything(self.args.seed)
        for batch_index, (source, label, pos_w, pos_d, _) in enumerate(
            self.train_loader
        ):
            if batch_index >= self.args.num_batches_to_process:
                break

            source = source[..., : self.args.dataset.dif_model.AGCRN.input_dim].to(
                self.device
            )  # (B, T, N, D)
            label = label[..., : self.args.dataset.dif_model.AGCRN.output_dim].to(
                self.device
            )  # (B, T, N, D)
            pos_w = pos_w.to(self.device)
            pos_d = pos_d.to(self.device)

            # agcrn
            adj = self.adj

            optimizer = self.state["optimizer"]
            optimizer.zero_grad()

            inputs = label.permute(0, 3, 2, 1).contiguous()  # (B, D, N, T)
            cond = source.permute(0, 3, 2, 1).contiguous()

            eps_loss = self.train_step_fn(self.state, inputs, adj, cond, pos_w, pos_d)

            loss = eps_loss
            loss.backward()

            self.optimize_fn(
                optimizer, self.dif_model.parameters(), step=self.state["step"]
            )
            self.state["step"] += 1
            self.state["ema"].update(self.dif_model.parameters())

            total_loss += loss.item()

            if (
                self.local_rank == 0
            ):  # Optionally restrict logging to just one process in distributed settings
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_index + 1}/{num_batches}: Batch Loss = {loss.item():.4f}"
                )

        if self.args.use_distributed:
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            torch.distributed.all_reduce(
                total_loss_tensor, op=torch.distributed.ReduceOp.SUM
            )
            total_loss = total_loss_tensor.item()

            train_epoch_loss = total_loss / (
                (num_batches * torch.distributed.get_world_size())
            )
        else:
            train_epoch_loss = total_loss / (num_batches)

        if self.local_rank == 0:
            # Only the master process computes and logs the average loss
            self.logger.info(
                "**********Train Epoch {}: Train Loss: {:.4f}".format(
                    epoch,
                    train_epoch_loss,
                )
            )

        return train_epoch_loss, self.state

    def val_epoch(self, epoch):
        self.dif_model.eval()
        total_loss = 0
        num_batches = (
            self.args.num_batches_to_process
            if self.args.num_batches_to_process != float("inf")
            else len(self.val_loader)
        )
        with torch.no_grad():
            for batch_index, (source, label, pos_w, pos_d, _) in enumerate(
                self.val_loader
            ):
                if batch_index >= self.args.num_batches_to_process:
                    break

                source = source[..., : self.args.dataset.dif_model.AGCRN.input_dim].to(
                    self.device
                )
                label = label[..., : self.args.dataset.dif_model.AGCRN.output_dim].to(
                    self.device
                )
                pos_w = pos_w.to(self.device)
                pos_d = pos_d.to(self.device)

                adj = self.adj

                inputs = label.permute(0, 3, 2, 1).contiguous()  # (B, D, N, T)
                cond = source.permute(0, 3, 2, 1).contiguous()

                eps_loss = self.eval_step_fn(
                    self.state, inputs, adj, cond, pos_w, pos_d
                )
                loss = eps_loss
                total_loss += loss.item()
                if self.local_rank == 0:
                    self.logger.info(
                        f"Epoch {epoch}, Batch {batch_index + 1}/{num_batches}: Batch Loss = {loss.item():.4f}"
                    )

        if self.args.use_distributed:
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            torch.distributed.all_reduce(
                total_loss_tensor, op=torch.distributed.ReduceOp.SUM
            )
            total_loss = total_loss_tensor.item()
            val_epoch_loss = total_loss / (
                (num_batches * torch.distributed.get_world_size())
            )
        else:
            val_epoch_loss = total_loss / (num_batches)

        if self.local_rank == 0:
            # Only the master process computes and logs the average loss
            self.logger.info(
                "**********Val Epoch {}: Val Loss: {:.4f}".format(
                    epoch,
                    val_epoch_loss,
                )
            )

        return val_epoch_loss, self.state

    def train(self):
        best_loss = float("inf")
        not_improved_count = 0
        start_time = time.time()

        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(1, self.args.dataset.epochs + 1):
            self.train_sampler.set_epoch(epoch) if self.args.use_distributed else None
            train_epoch_loss, self.state = self.train_epoch(epoch)
            if self.local_rank == 0:
                wandb.log(
                    {
                        "Train_loss_Epoch": train_epoch_loss,
                        "epoch": epoch,
                    }
                )
            self.val_sampler.set_epoch(epoch) if self.args.use_distributed else None
            val_loss, self.state = self.val_epoch(epoch)
            if self.local_rank == 0:
                wandb.log(
                    {
                        "Validate_loss_Epoch": val_loss,
                        "epoch": epoch,
                    }
                )
            early_stop_flag = torch.tensor(0, device="cuda")
            if val_loss < best_loss:
                not_improved_count = 0
                best_loss = val_loss
                best_epoch_number = epoch
            else:
                not_improved_count += 1

            if self.local_rank == 0:
                self.logger.info("Saving current best model to " + self.best_path)
                save_checkpoint(
                    self.best_path,
                    self.state,
                )

            if (
                self.args.early_stop
                and not_improved_count == self.args.early_stop_patience
            ):
                (
                    self.logger.info(
                        f"Validation performance didn't improve for {self.args.early_stop_patience} epochs. Training stops."
                    )
                    if self.local_rank == 0
                    else None
                )
                early_stop_flag += 1

            if self.args.use_distributed:
                dist.broadcast(early_stop_flag, src=0)
            if early_stop_flag.item() > 0:
                break

        training_time = time.time() - start_time
        metrics = [
            (training_time / 60),
            best_loss,
            best_epoch_number,
        ]

        if self.local_rank == 0:
            self.logger.info(
                f"Total training time: {(training_time / 60):.4f}min, best validation loss: {best_loss:.4f}, epoch number: {best_epoch_number}"
            )
            wandb.log(
                {
                    "Total_training_time": (training_time / 60),
                    "Best_validation_loss": best_loss,
                    "Best_epoch_number": best_epoch_number,
                    "Best model path": self.best_path,
                }
            )
            # Write metrics to Excel
            notes = self.args.notes
            log_dir = self.args.log_dir
            write_train_metrics_to_excel(metrics, notes, self.best_path)

        # Sampling
        dist.barrier() if self.args.use_distributed else None

        if self.args.train_sampling:
            self.sampling(train_metrics=metrics)

        (
            dist.destroy_process_group()
            if self.args.use_distributed and not self.args.hyperopt
            else None
        )

        return best_loss, self.best_path

    def sampling(self, train_metrics=None):
        seed_everything(self.args.seed)
        self.state = restore_checkpoint(self.best_path, self.state, self.device)
        self.ema.copy_to(self.dif_model.parameters())
        predictions_lists = []
        best_predictions_lists = []
        labels_lists = []
        all_sources = []
        all_labels = []
        all_pos_ws = []
        all_pos_ds = []
        all_indices = []
        with torch.no_grad():
            for batch_index, (source, label, pos_w, pos_d, idx) in enumerate(
                self.test_loader
            ):
                if batch_index >= self.args.sampling_num_batches_to_process:
                    break
                # get the first batch of the test data

                source = source[..., : self.args.dataset.dif_model.AGCRN.input_dim].to(
                    self.device
                )
                label = label[..., : self.args.dataset.dif_model.AGCRN.output_dim].to(
                    self.device
                )
                pos_w = pos_w.to(self.device)
                pos_d = pos_d.to(self.device)

                all_sources.append(source)
                all_labels.append(label)
                all_pos_ws.append(pos_w)
                all_pos_ds.append(pos_d)
                all_indices.append(idx)
            all_sources = torch.cat(all_sources, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_pos_ws = torch.cat(all_pos_ws, dim=0)
            all_pos_ds = torch.cat(all_pos_ds, dim=0)
            all_indices = torch.cat(all_indices, dim=0)

            if self.args.random_sampling:
                random_indices = torch.randperm(all_sources.size(0))[
                    : self.args.sampling_first_batch
                ]
            else:
                random_indices = [0]

            sampled_sources = all_sources[random_indices]
            sampled_labels = all_labels[random_indices]
            sampled_pos_ws = all_pos_ws[random_indices]
            sampled_pos_ds = all_pos_ds[random_indices]
            sampled_indices = all_indices[random_indices]

            self.logger.info(f"Random indices: {random_indices}")

            # Retrieve raw sequences using sampled_start_indices
            raw_sequences = []
            for indices in sampled_indices:
                raw_sequences.append(self.original_data[indices])
            raw_sequences = np.concatenate(raw_sequences, axis=0)
            np.save(
                f"{self.args.log_dir}/raw_sequences.npy",
                raw_sequences,
            )

            source = sampled_sources
            label = sampled_labels
            pos_w = sampled_pos_ws
            pos_d = sampled_pos_ds
            psw_w_nsamples = (
                pos_w.unsqueeze(1)
                .repeat(1, self.args.n_samples, 1)
                .reshape(-1, pos_w.shape[1])
            )

            psw_d_nsamples = (
                pos_d.unsqueeze(1)
                .repeat(1, self.args.n_samples, 1)
                .reshape(-1, pos_d.shape[1])
            )

            adj = self.adj

            inputs = label.permute(0, 3, 2, 1).contiguous()  # (B, D, N, T)

            cond = source.permute(0, 3, 2, 1).contiguous()
            cond_n_samples = (
                cond.unsqueeze(1)
                .repeat(1, self.args.n_samples, 1, 1, 1)
                .reshape(-1, *cond.shape[1:])
            )  # (B*n_samples, D, N, T)

            guider_fn = self.guider_fn(source, adj)

            predictions, best_predictions = self.sampling_fn(
                self.dif_model,
                guider_fn,
                adj,
                cond_n_samples,
                psw_w_nsamples,
                psw_d_nsamples,
                source.permute(0, 3, 2, 1).contiguous(),
                self.args,
                self,
                self.inverse_transform,
                label,
                batch_index,
                len(self.test_loader),
            )  # (B*n_samples, D, N, 2T)
            predictions_samples = (
                predictions.reshape(-1, self.args.n_samples, *predictions.shape[1:])[
                    :, :, :, :, -self.args.dataset.dif_model.AGCRN.T_p :
                ]
                .permute(0, 1, 4, 3, 2)
                .contiguous()
            )  # (B, n_samples, T, N, D)
            predictions_lists.append(predictions_samples)

            best_predictions_samples = (
                best_predictions.reshape(
                    -1, self.args.n_samples, *best_predictions.shape[1:]
                )[:, :, :, :, -self.args.dataset.dif_model.AGCRN.T_p :]
                .permute(0, 1, 4, 3, 2)
                .contiguous()
            )
            best_predictions_lists.append(best_predictions_samples)

            labels_lists.append(label)

            labels = torch.cat(labels_lists, dim=0)  # (B, T, N, D)
            predictions = torch.cat(predictions_lists, dim=0)  # (B, n_samples, T, N, D)
            best_predictions = torch.cat(best_predictions_lists, dim=0)
            np.save(
                f"{self.args.log_dir}/best_predictions_pre_inverse.npy",
                best_predictions.cpu().numpy(),
            )
            np.save(
                f"{self.args.log_dir}/predictions_pre_inverse.npy",
                predictions.cpu().numpy(),
            )
            np.save(
                f"{self.args.log_dir}/labels_pre_inverse.npy",
                labels.cpu().numpy(),
            )

            if self.args.use_distributed:
                gathered_outputs, gathered_labels = self.gather_data(
                    predictions, labels
                )
                gathered_best_outputs, _ = self.gather_data(best_predictions, labels)

            else:
                gathered_outputs = predictions
                gathered_best_outputs = best_predictions
                gathered_labels = labels

            # Inverse transform
            predictions_inv, labels_inv = self.inverse_transform(
                gathered_outputs, gathered_labels
            )
            predictions_det = torch.mean(predictions_inv, dim=1)  # (B, T, N, D)
            predictions_prob = predictions_inv  # (B, n_samples, T, N, D)
            best_predictions_inv, _ = self.inverse_transform(
                gathered_best_outputs, gathered_labels
            )
            best_predictions_det = torch.mean(best_predictions_inv, dim=1)
            best_predictions_prob = best_predictions_inv

            np.save(
                f"{self.args.log_dir}/best_predictions.npy",
                best_predictions_prob.cpu().numpy(),
            )

            labels = labels_inv
            # Save labels
            np.save(
                f"{self.args.log_dir}/labels.npy",
                labels.cpu().numpy(),
            )

            # Metrics
            mae, rmse, mape, crps, mis = All_Metrics(
                predictions_det, predictions_prob, labels, None, 0
            )
            mae_best, rmse_best, mape_best, crps_best, mis_best = All_Metrics(
                best_predictions_det, best_predictions_prob, labels, None, 0
            )

            if self.local_rank == 0:
                self.logger.info(
                    f"**********Test Metrics: MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape * 100:.2f}%, CRPS: {crps:.4f}, MIS: {mis:.4f}"
                )
                self.logger.info(
                    f"**********Best Metrics: MAE: {mae_best:.4f}, RMSE: {rmse_best:.4f}, MAPE: {mape_best * 100:.2f}%, CRPS: {crps_best:.4f}, MIS: {mis_best:.4f}"
                )

                write_all_to_sqlite(
                    train_metrics=train_metrics,
                    sample_metrics=[
                        mae.cpu().numpy(),
                        rmse.cpu().numpy(),
                        mape.cpu().numpy(),
                        crps,
                        mis,
                    ],
                    best_sample_metrics=[
                        mae_best.cpu().numpy(),
                        rmse_best.cpu().numpy(),
                        mape_best.cpu().numpy(),
                        crps_best,
                        mis_best,
                    ],
                    notes=self.args.notes,
                    log_dir=(
                        self.best_path
                        if not self.args.test_only
                        else f"{self.args.log_dir}/v011_run_sde.log"
                    ),
                    db_path=self.args.db_path,
                )
            # dist.destroy_process_group()
            dist.barrier() if self.args.use_distributed else None
            rank = dist.get_rank() if self.args.use_distributed else 0
            self.logger.info(f"Rank {rank} finished sampling")

            return mae_best

    def guider_fn(self, source, adj):
        def exe(perturbed_data, t):
            if self.args.guider_scale is not None:
                with torch.enable_grad():
                    perturbed_data = perturbed_data.detach().requires_grad_(True)
                    future_hat, _ = self.agcrn(source)
                    future_hat = future_hat.permute(0, 3, 2, 1).contiguous()
                    future_hat_n_samples = (
                        future_hat.unsqueeze(1)
                        .repeat(1, self.args.n_samples, 1, 1, 1)
                        .reshape(-1, *future_hat.shape[1:])
                    )  # (B*n_samples, D, N, T)

                    loss = perturbed_data - future_hat_n_samples
                    norm = torch.linalg.norm(loss)
                    grad = -torch.autograd.grad(norm, perturbed_data)[0]
                    return grad * self.args.guider_scale
            elif self.args.guider_model and self.args.guider_scale is not None:
                with torch.enable_grad():
                    perturbed_data = perturbed_data.detach().requires_grad_(True)
                    t = t * (self.args.dataset.dif_model.model.num_scales - 1)
                    predictions = self.guider(
                        perturbed_data,
                        t,
                        adj,
                    )
                    future_hat, _ = self.agcrn(source)
                    future_hat = future_hat.permute(0, 3, 2, 1).contiguous()
                    future_hat_n_samples = (
                        future_hat.unsqueeze(1)
                        .repeat(1, self.args.n_samples, 1, 1, 1)
                        .reshape(-1, *future_hat.shape[1:])
                    )  # (B*n_samples, D, N, T)

                    loss = predictions - future_hat_n_samples
                    norm = torch.linalg.norm(loss)
                    grad = -torch.autograd.grad(norm, perturbed_data)[0]
                    return grad * self.args.guider_scale

            else:
                return 0

        return exe

    def inverse_transform(self, outputs, labels):
        for i in range(outputs.shape[1]):
            outputs[:, i, ...] = self.scaler.inverse_transform(outputs[:, i, ...])
        labels = self.scaler.inverse_transform(labels)
        return outputs, labels

    def gather_data(self, outputs, labels):
        """Gathers outputs and labels from all processes."""
        predictions_list = [
            torch.zeros_like(outputs) for _ in range(dist.get_world_size())
        ]
        labels_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
        dist.all_gather(predictions_list, outputs)
        dist.all_gather(labels_list, labels)
        return torch.cat(predictions_list, dim=0), torch.cat(labels_list, dim=0)
