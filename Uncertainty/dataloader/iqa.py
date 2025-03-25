import torch
import torch.nn.functional as F
import math
from pyiqa.utils.color_util import to_y_channel


def masked_psnr(x, y, mask, test_y_channel=False, data_range=1.0, eps=1e-8, color_space='yiq'):
    r"""Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        mask: A mask tensor. Shape :math:`(N, 1, H, W)`.
        test_y_channel (Boolean): Convert RGB image to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.
        data_range: Maximum value range of images (default 1.0).

    Returns:
        PSNR Index of similarity betwen two images.
    """

    if (x.shape[1] == 3) and test_y_channel:
        # Convert RGB image to YCbCr and use Y-channel
        x = to_y_channel(x, data_range, color_space)
        y = to_y_channel(y, data_range, color_space)

    mse = (x - y) ** 2
    mse = (mse * mask).sum() / mask.sum()
    score = 10 * torch.log10(data_range**2 / (mse + eps))

    return score


def masked_ssim(img1, img2, mask, window_size=11, sigma=1.5, data_range=1.0):
    channel = img1.size()[1]
    window = create_window(window_size, channel, sigma)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    # Mask the ssim map
    ssim_map *= mask

    return ssim_map.sum() / mask.sum()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    mask = torch.ones(1, 1, 256, 256)
    print(masked_ssim(x, y, mask))
