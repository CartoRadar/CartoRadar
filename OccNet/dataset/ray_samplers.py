import torch

# point sampling for rays for the rendering
class UniformRaySampler:
    def __init__(self):
        "print('Initializing a uniform ray sampler')"

    def get_samples(self, rays, N_samples, perturb):
        """
        Args:
            rays: (N_rays, 2(near, far))
            N_samples: number of samples
            perturb: whether we want to perturb the z_vals
        Returns:
            z_vals: (N_rays, N_samples)
        """
        N_rays = rays.shape[0]
        near = rays[:, -2:-1]
        far = rays[:, -1:]
        with torch.no_grad():
            z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
            # z_steps = torch.logspace(-4, 0, N_samples, device=rays.device)       # (N_samples)
            z_vals = near * (1 - z_steps) + far * z_steps
            z_vals = z_vals.expand(N_rays, N_samples)

            if perturb > 0:  # perturb z_vals
                # (N_rays, N_samples-1) interval mid points
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                # get intervals between samples
                upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
                lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)
                perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
                z_vals = lower + (upper - lower) * perturb_rand

        return z_vals  # (N_rays, N_samples)
