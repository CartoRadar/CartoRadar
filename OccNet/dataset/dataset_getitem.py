import torch

# register the getitem function
def obtain_getitem(model_name):
    if model_name == "RadarOccNerf":
        return radar_getitem
    else:
        raise ValueError()

def radar_getitem(
    indices,
    global_pos=None,
    global_depth=None,
    global_dir=None,
    dataset=None,
    idx=None,
    conf=None,
):
    """
    Args:
        indices: indices of the samples
        global_pos: global position of the samples
        global_depth: global depth of the samples
        global_dir: global direction of the samples
        dataset: dataset
        idx: idx
        conf: confidence
    Returns:
        dict: dictionary of the sample points"""
    if indices is not None:
        depth = global_depth[indices, :]
        depth = depth[(valid_mask := (depth > dataset.scaled_ray_range[0]).squeeze(-1))]
        pos = global_pos[indices, :][valid_mask]
        dir = global_dir[indices, :][valid_mask]
        conf = conf[indices, :][valid_mask]
    else:
        depth = global_depth
        depth = depth[(valid_mask := (depth > dataset.scaled_ray_range[0]).squeeze(-1))]
        pos = global_pos[valid_mask]
        dir = global_dir[valid_mask]
        conf = conf[valid_mask]


    pts, z_vals = tensor_sample_pts(
        pos,
        dir,
        depth,
        dataset.scaled_ray_range[0],
        dataset.scaled_ray_range[1],
        dataset.ray_pts,
    )
    labels, weights, labels1, weights1 = tensor_get_label_weight(
        depth,
        conf,
        z_vals,
        idx,
        dataset.ray_pts,
        dataset.scaled_ray_range[0],
        dataset.scaled_ray_range[1],
        dataset.scale_factor,
        dataset.use_uncertainty,
        input_epsilon = dataset.input_epsilon,
    )

    return {
        "pts": pts.reshape(-1, 3),
        "labels": labels.reshape(-1, 1),
        "weights": weights.reshape(-1, 1),
        "depths": depth.reshape(-1, 1),
        "z_vals": z_vals.reshape(-1, 1),
        "labels1": labels1.reshape(-1, 1),
        "weights1": weights1.reshape(-1, 1),
    }


def laplace_cdf(x, mu, b):
    """compute the laplace cdf values"""
    cdf = torch.zeros_like(x)
    left_inds = x < mu
    right_inds = ~left_inds
    mu = mu.expand(x.shape)
    b = b.expand(x.shape)

    cdf[left_inds] = 0.5 * torch.exp((x[left_inds] - mu[left_inds]) / b[left_inds])
    cdf[right_inds] = 1 - 0.5 * torch.exp((mu[right_inds] - x[right_inds]) / b[right_inds])
    return cdf


def tensor_get_label_weight(depth, conf, z_vals, idx, ray_pts, near, far, scale_factor, use_uncertainty=True, input_epsilon=0.004):
    """
    Args:
        depth: depth
        conf: confidence
        z_vals: z values
        idx: idx
        ray_pts: ray points
        near: near
        far: far
        scale_factor: scale factor
        use_uncertainty: use uncertainty
        input_epsilon: input epsilon
    Returns:
        tuple: tuple of labels and weights"""
    depth = depth.view(-1)
    pixel_num = depth.shape[0]
    labels0 = torch.zeros((pixel_num, ray_pts), dtype=torch.float32, device=depth.device)
    labels1 = torch.ones_like(labels0)

    # new weight place holder
    weights0 = torch.ones((pixel_num, ray_pts), dtype=torch.float32, device=depth.device) * 6
    weights1 = torch.full_like(weights0, 200 * 6)

    # view each point as independent
    if use_uncertainty:
        epsilon = input_epsilon / scale_factor
        depth = depth.view(-1, 1)

        epsilon_plus_cdf = laplace_cdf(z_vals + epsilon, depth, conf)
        epsilon_minus_cdf = laplace_cdf(z_vals - epsilon, depth, conf)
        weights0 *= 1 - epsilon_plus_cdf  # prob for x_i = 0
        weights1 *= epsilon_plus_cdf - epsilon_minus_cdf  # prob for x_i = 1

    return labels0, weights0, labels1, weights1


def tensor_sample_pts(pos, dir, depth, scaled_near, scaled_far, ray_pts, perturb=False):
    """
    Args:
        pos: position
        dir: direction
        depth: depth
        scaled_near: scaled near
        scaled_far: scaled far
        ray_pts: ray points
        perturb: perturb
    Returns:
        tuple: tuple of sample points and z values"""
    near = torch.ones_like(pos[:, -1:]) * scaled_near
    far = torch.ones_like(pos[:, -1:]) * scaled_far
    ray_number = pos.shape[0]

    with torch.no_grad():
        # add jitter tot he sampling points
        t_vals = torch.arange(ray_pts + 1, device=pos.device, dtype=torch.float32)[None, ...]
        if perturb:
            t_rand = torch.rand([ray_number, ray_pts + 1], device=pos.device, dtype=torch.float32)
            t_vals = t_vals + t_rand
        t_vals[:, -2] = ray_pts
        t_vals = t_vals[:, :-1] / ray_pts
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

    xyz_samples = pos[:, None, :] + dir[:, None, :] * z_vals[:, :, None]
    return xyz_samples.reshape(-1, 3), z_vals
