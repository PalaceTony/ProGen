import torch
import os
import time
import copy

import wandb
import os
import torch
import torch.utils.data
import torch.distributed as dist
import torch.nn.functional as F


class Trainer(object):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        train_sampler,
        val_sampler,
        device,
        local_rank,
        args,
        logger,
        agcrn,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        self.agcrn = agcrn
        if self.args.num_batches_to_process == "inf":
            self.args.num_batches_to_process = float("inf")
        else:
            self.args.num_batches_to_process = int(self.args.num_batches_to_process)

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25
            * t**2
            * (self.args.dif_model.model.beta_max - self.args.dif_model.model.beta_min)
            - 0.5 * t * self.args.dif_model.model.beta_min
        )
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_index, (source, label) in enumerate(self.train_loader):
            if batch_index >= self.args.num_batches_to_process:
                break
            self.optimizer.zero_grad()

            source = source[..., : self.args.dif_model.AGCRN.input_dim].to(self.device)
            label = label[..., : self.args.dif_model.AGCRN.output_dim].to(self.device)

            t = (
                torch.rand(label.shape[0], device=self.device)
                * (1 - self.args.dif_model.model.eps)
                + self.args.dif_model.model.eps
            )
            z = torch.randn_like(label.permute(0, 3, 2, 1).contiguous())

            t_steps = t * (self.args.dif_model.model.num_scales - 1)

            mean, std = self.marginal_prob(label.permute(0, 3, 2, 1).contiguous(), t)
            perturbed_data = mean + std[:, None, None, None] * z

            future_agcrn, adj = self.agcrn(source)
            inputs = perturbed_data
            furture_hat = self.model(inputs, t_steps, adj)

            loss = F.mse_loss(
                furture_hat,
                future_agcrn.permute(0, 3, 2, 1).contiguous(),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.9)
            self.optimizer.step()

            num_batches = (
                self.args.num_batches_to_process
                if self.args.num_batches_to_process != float("inf")
                else len(self.train_loader)
            )

            if self.local_rank == 0:
                if (
                    batch_index + 1
                ) % 10 == 0:  # Adjust the modulus value depending on how frequently you want updates
                    self.logger.info(
                        f"Epoch {epoch}, Train Batch {batch_index+1}/{num_batches}: Current Batch Loss = {loss.item():.4f}"
                    )

            total_loss += loss.item()

        if self.args.use_distributed:
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            torch.distributed.all_reduce(
                total_loss_tensor, op=torch.distributed.ReduceOp.SUM
            )
            total_loss = total_loss_tensor.item()
            num_batches = (
                self.args.num_batches_to_process
                if self.args.num_batches_to_process != float("inf")
                else len(self.train_loader)
            )
            train_epoch_loss = total_loss / (
                (num_batches * torch.distributed.get_world_size())
            )
        else:
            num_batches = (
                self.args.num_batches_to_process
                if self.args.num_batches_to_process != float("inf")
                else len(self.train_loader)
            )
            train_epoch_loss = total_loss / (num_batches)

        if self.local_rank == 0:
            # Only the master process computes and logs the average loss
            self.logger.info(
                "**********Train Epoch {}: Train Loss: {:.4f}".format(
                    epoch,
                    train_epoch_loss,
                )
            )

        return train_epoch_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_index, (source, label) in enumerate(self.val_loader):
                if batch_index >= self.args.num_batches_to_process:
                    break
                source = source[..., : self.args.dif_model.AGCRN.input_dim].to(
                    self.device
                )
                label = label[..., : self.args.dif_model.AGCRN.output_dim].to(
                    self.device
                )

                t = (
                    torch.rand(label.shape[0], device=self.device)
                    * (1 - self.args.dif_model.model.eps)
                    + self.args.dif_model.model.eps
                )
                z = torch.randn_like(label.permute(0, 3, 2, 1).contiguous())

                t_steps = t * (self.args.dif_model.model.num_scales - 1)

                mean, std = self.marginal_prob(
                    label.permute(0, 3, 2, 1).contiguous(), t
                )
                perturbed_data = mean + std[:, None, None, None] * z

                future_agcrn, adj = self.agcrn(source)
                inputs = perturbed_data
                furture_hat = self.model(inputs, t_steps, adj)

                loss = F.mse_loss(
                    furture_hat,
                    future_agcrn.permute(0, 3, 2, 1).contiguous(),
                )

                num_batches = (
                    self.args.num_batches_to_process
                    if self.args.num_batches_to_process != float("inf")
                    else len(self.val_loader)
                )
                if self.local_rank == 0:
                    if (
                        batch_index + 1
                    ) % 10 == 0:  # Adjust the modulus value depending on how frequently you want updates
                        self.logger.info(
                            f"Epoch {epoch}, Val Batch {batch_index+1}/{num_batches}: Current Batch Loss = {loss.item():.4f}"
                        )

                total_loss += loss.item()

        if self.args.use_distributed:
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            torch.distributed.all_reduce(
                total_loss_tensor, op=torch.distributed.ReduceOp.SUM
            )
            total_loss = total_loss_tensor.item()
            num_batches = (
                self.args.num_batches_to_process
                if self.args.num_batches_to_process != float("inf")
                else len(self.val_loader)
            )
            val_epoch_loss = total_loss / (
                (num_batches * torch.distributed.get_world_size())
            )
        else:
            num_batches = (
                self.args.num_batches_to_process
                if self.args.num_batches_to_process != float("inf")
                else len(self.val_loader)
            )
            val_epoch_loss = total_loss / (num_batches)

        if self.local_rank == 0:
            # Only the master process computes and logs the average loss
            self.logger.info(
                "**********Validate Epoch {}: Validate Loss: {:.4f}".format(
                    epoch,
                    val_epoch_loss,
                )
            )

        return val_epoch_loss

    def train(self):
        best_model = None
        best_loss = float("inf")
        not_improved_count = 0
        start_time = time.time()

        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(1, self.args.epochs + 1):
            self.train_sampler.set_epoch(epoch) if self.args.use_distributed else None
            train_epoch_loss = self.train_epoch(epoch)
            self.scheduler.step()
            if self.local_rank == 0:
                wandb.log(
                    {
                        "Train_loss_Epoch": train_epoch_loss,
                        "epoch": epoch,
                    }
                )
            self.val_sampler.set_epoch(epoch) if self.args.use_distributed else None
            val_loss = self.val_epoch(epoch)
            if self.local_rank == 0:
                wandb.log(
                    {
                        "Validate_loss_Epoch": val_loss,
                        "epoch": epoch,
                    }
                )
            early_stop_flag = torch.tensor(0, device="cuda")
            if self.local_rank == 0:
                if val_loss < best_loss:
                    not_improved_count = 0
                    best_loss = val_loss
                    best_epoch_number = epoch
                    best_model = copy.deepcopy(self.model.state_dict())
                    self.logger.info("Saving current best model to " + self.best_path)
                    torch.save(best_model, self.best_path)  # Save the best model
                else:
                    not_improved_count += 1

                if (
                    self.args.early_stop
                    and not_improved_count == self.args.early_stop_patience
                ):
                    self.logger.info(
                        f"Validation performance didn't improve for {self.args.early_stop_patience} epochs. Training stops."
                    )
                    early_stop_flag += 1

            if self.args.use_distributed:
                dist.broadcast(early_stop_flag, src=0)

            if early_stop_flag.item() > 0:
                break

        if self.args.use_distributed:
            dist.barrier()

        if self.local_rank == 0:
            training_time = time.time() - start_time
            self.logger.info(
                f"Total training time: {(training_time / 60):.4f}min, best validation loss: {best_loss:.4f}, epoch number: {best_epoch_number}"
            )

        torch.distributed.destroy_process_group() if self.args.use_distributed else None
