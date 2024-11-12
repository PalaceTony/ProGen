import torch
import hydra
import logging
import wandb
import os
import torch.distributed as dist

from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf

from v113_trainer_guider_unet import Trainer
from lib.utils import seed_everything, set_local_rank, setup_logging
from lib.dataloader import get_dataloader
from score_sde_spatiotemporal.ddpm_guider import DDPM as UGnet
from AGCRN import AGCRN
from collections import OrderedDict


@hydra.main(
    version_base=None,
    config_path="../../configuration/modules",
    config_name="guider_train",
)
def main(args: DictConfig) -> None:
    use_distributed = args.get("use_distributed", True)
    if use_distributed:
        # Initialize the distributed environment
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        set_local_rank(local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device if torch.cuda.is_available() else "cpu"
        local_rank = 0

    # Logging and seed
    setup_logging()
    logger = logging.getLogger("ugnet")
    seed_everything(args.seed)

    args.device = device_str
    logging.info(f"Using device: {args.device}")
    device = torch.device(args.device)

    agcrn = AGCRN(args.dif_model.AGCRN).to(device)
    dif_state_dict = torch.load(args.agcrn)
    new_state_dict = OrderedDict()
    if args.use_distributed:
        for k, v in dif_state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove `module.` prefix
            new_state_dict[name] = v
    else:
        new_state_dict = dif_state_dict

    agcrn.load_state_dict(new_state_dict)
    for param in agcrn.parameters():
        param.requires_grad = False

    # Model
    model = UGnet(args.dif_model).to(device)
    model = DDP(model, device_ids=[local_rank]) if args.use_distributed else model
    if local_rank == 0:
        logging.info(
            f"Total number of parameters: {sum(p.numel() for p in model.parameters())}"
        )

    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False,
    )

    scheduler = StepLR(optimizer, step_size=10, gamma=1)

    # Data loading
    (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        train_sampler,
        val_sampler,
        _,
    ) = get_dataloader(
        args,
        local_rank,
        normalizer=args.normalizer,
        single=False,
        use_distributed=args.use_distributed,
    )

    # Wandb setup
    if local_rank == 0:
        wandb_dir = args.wandb_dir
        os.environ["WANDB_MODE"] = args.wandb_mode
        wandb.init(
            project=args.wandb_project_name,
            name=args.run_name,
            dir=wandb_dir,
            notes=args.notes,
            settings=wandb.Settings(code_dir=wandb_dir),
            save_code=True,
        )
        wandb.config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Training
    trainer = Trainer(
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
    )
    trainer.train()

    # Finish Wandb
    if local_rank == 0:
        wandb.run.finish()


if __name__ == "__main__":
    main()
