"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""

import abc
import torch
import numpy as np


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t, adj, args, st_version=True):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t, adj, args, st_version)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, guider, adj):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)

                score = score_fn(x, t) + guider(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t, guider, adj, args, st_version=True):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, adj, args, st_version)
                G = G[:, None, None, None] if len(G.shape) == 1 else G
                rev_f = f - G**2 * (score_fn(x, t) + guider(x, t)) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape, n_sample):
        return torch.randn(n_sample, *shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t, adj, args, st_version=True):
        if st_version:
            beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
            adj = torch.from_numpy(adj).to(torch.float32).to(x.device)
            neighbors_sum = torch.einsum("ij,bjkl->bikl", adj, x)
            neighbors_mean = neighbors_sum.mean(
                dim=1, keepdim=True
            )  # Shape: [1920, 1, 170, 12]

            drift = (
                -0.5
                * beta_t[:, None, None, None]
                * (x - args.dataset.neighbors_sum_c * neighbors_mean)
            )

            # Diffusion term

            discount = 1.0 - torch.exp(
                -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
            )

            if args.dataset.spatial_discount_type == "type1":
                spatial_discount = torch.exp(
                    -args.dataset.neighbors_sum_c2
                    * neighbors_mean
                    * torch.log(t + 1)[:, None, None, None]
                )

            elif args.dataset.spatial_discount_type == "type2":
                spatial_discount = 1.0 - torch.exp(
                    -args.dataset.neighbors_sum_c2
                    * neighbors_mean
                    * torch.log(t + 1)[:, None, None, None]
                )
            elif args.dataset.spatial_discount_type == "type3":
                spatial_discount = 1.0 - torch.exp(
                    -args.dataset.neighbors_sum_c2
                    * neighbors_mean
                    * torch.sqrt(t)[:, None, None, None]
                )
            elif args.dataset.spatial_discount_type == "type4":
                spatial_discount = 1.0 - torch.exp(
                    -args.dataset.neighbors_sum_c2
                    * neighbors_mean
                    * (torch.log(t + 1) + torch.sqrt(t))[:, None, None, None]
                )
            elif args.dataset.spatial_discount_type == "type5":
                spatial_discount = torch.exp(
                    -args.dataset.neighbors_sum_c2
                    * neighbors_mean
                    * (t**0.5)[:, None, None, None]
                )

            spatial_discount = torch.clamp(spatial_discount, min=1e-10)

            discount = discount[:, None, None, None] * spatial_discount

            diffusion = torch.sqrt(beta_t[:, None, None, None] * discount)
            return drift, diffusion
        else:
            beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
            drift = -0.5 * beta_t[:, None, None, None] * x
            discount = 1.0 - torch.exp(
                -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
            )
            diffusion = torch.sqrt(beta_t * discount)
            return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape, n_sample):
        return torch.randn(n_sample, *shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0


class stVPSDE(SDE):
    def __init__(
        self,
        beta_min=0.1,
        beta_max=20,
        N=1000,
        scale1=0.1,
        scale2=0.2,
        scale3=0.3,
        scale4=0.4,
        scale5=0.5,
        adjacency=None,
        history=None,
        future_hat=None,
    ):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        self.scale4 = scale4
        self.scale5 = scale5
        self.adjacency = adjacency
        self.history = history
        self.future_hat = future_hat

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = (
            -0.5 * beta_t[:, None, None, None] * x
            + self.scale1 * (torch.einsum("ij,bfjk->bfik", self.adjacency, x))
            + self.scale2 * self.history
            + self.scale3 * self.future_hat
        )

        discount = 1.0 - torch.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
        )
        diffusion = torch.sqrt(beta_t * discount)[:, None, None, None] * torch.exp(
            self.scale4 * torch.einsum("ij,bfjk->bfik", self.adjacency, x)
            + self.scale5 * self.history
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape, n_sample):
        return torch.randn(n_sample, *shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device
            )
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)
        ) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            self.discrete_sigmas[timestep - 1].to(t.device),
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G
