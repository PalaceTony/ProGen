import torch.nn as nn
from score_sde_spatiotemporal.ddpm_guider import DDPM as UGnet
from AGCRN import AGCRN as AGCRN


class diffusion(nn.Module):
    def __init__(self, args, device):
        super(diffusion, self).__init__()
        self.eps_model = UGnet(args).to(device)
        self.agcrn_model = AGCRN(args.AGCRN).to(device)

    def forward(self):
        raise NotImplementedError(
            "This CombinedModel does not implement a forward method. "
            "Use individual model forward methods or handle externally."
        )
