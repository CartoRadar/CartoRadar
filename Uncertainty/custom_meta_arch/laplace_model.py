import pyiqa
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from detectron2.config import configurable
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.structures import ImageList


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, padding=0)


def DownsampleAzimuth(dim):
    return nn.Conv2d(dim, dim, (1, 4), (1, 2), padding=0)


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, padding=3)


def UpsampleAzimuth(dim):
    return nn.ConvTranspose2d(dim, dim, (1, 4), (1, 2), padding=(0, 3))


def pad(x, padding, circular=True):
    if circular:
        x = F.pad(x, (padding, padding, 0, 0), 'circular')
    else:
        x = F.pad(x, (padding, padding, 0, 0), 'constant')
    x = F.pad(x, (0, 0, padding, padding), 'constant')
    return x


def pad_azimuth(x, padding, circular=True):
    if circular:
        return F.pad(x, (padding, padding, 0, 0), 'circular')
    else:
        return F.pad(x, (padding, padding, 0, 0), 'constant')


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, groups=8, circular=True):
        super().__init__()
        self.padding = padding
        self.circular = circular
        self.proj = nn.Conv2d(in_channels, out_channels, kernel, padding=0)
        self.norm = nn.Identity() if groups is None else nn.GroupNorm(groups, out_channels, eps=1e-4)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = pad(x, padding=self.padding, circular=self.circular)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, circular=True):
        """The basic res block for the network.
        Args:
            groups: can be number or None. When it is None, means disable norm. Can also be a tuple
                to specify the settings for the two ConvBlock
        """
        super().__init__()
        try:
            groups1, groups2 = groups
        except:
            groups1 = groups2 = groups

        self.block1 = ConvBlock(in_channels, out_channels, groups=groups1, circular=circular)
        self.block2 = ConvBlock(out_channels, out_channels, groups=groups2, circular=circular)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        residual = self.res_conv(x)
        out = h + residual
        return out


class UnetBackbone(nn.Module):
    """This is the backbone for our unet model.

    1. It uses basic blocks instead of bottleneck block (but ativation first then addition)
    2. Conv2d (k=4x4, stride=2, padding=0) for downsample
    3. downsample happens at the end of each layer
    4. has stem connection
    5. only downsample azimuth (width) for layers more than 3
    6. Group Norm, SiLU, learnable bias
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_down_elev = 3  # after this layer only down azimuth
        dim = cfg.MODEL.BACKBONE.STEM_OUT_CHANNELS  # stem output channel
        dim_copy = 64
        dim_mults = cfg.MODEL.BACKBONE.DIM_MULTS
        num_blocks_per_down = cfg.MODEL.BACKBONE.NUM_BLOCKS_PER_DOWN
        resnet_groups = 8
        init_kernel_size = 7
        self.circular = cfg.MODEL.CIRCULAR_DEPTH
        channel_in = 256

        assert (init_kernel_size % 2) == 1
        self.init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv2d(channel_in, dim, init_kernel_size, padding=0)

        # dimensions
        dims = [dim, *map(lambda m: dim_copy * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        # block type
        block_klass = partial(BasicBlock, groups=resnet_groups, circular=self.circular)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            azimuth_only = ind >= self.num_down_elev and not is_last
            if is_last:
                downsample = nn.Identity()
            elif azimuth_only:
                downsample = DownsampleAzimuth(dim_out)
            else:
                downsample = Downsample(dim_out)

            num_blocks = nn.ModuleList([])
            num_blocks.append(block_klass(dim_in, dim_out))
            for _ in range(num_blocks_per_down[ind] - 1):
                num_blocks.append(block_klass(dim_out, dim_out))
            num_blocks.append(downsample)
            self.downs.append(num_blocks)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        outputs = {}

        x = pad(x, padding=self.init_padding, circular=self.circular)
        x = self.init_conv(x)
        outputs['stem'] = x
        for i, down_blocks in enumerate(self.downs):
            for idx, block in enumerate(down_blocks[:-1]):
                x = block(x)
            outputs[f'res{i+2}'] = x
            is_last = (i + 1) >= len(self.downs)
            azimuth_only = i >= self.num_down_elev and not is_last
            if azimuth_only:
                x = pad_azimuth(x, padding=1, circular=self.circular)
            elif not is_last:
                x = pad(x, padding=1, circular=self.circular)
            x = down_blocks[-1](x)
            if isinstance(x, tuple):
                x = x[0]

        return outputs


class DepthHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_down_elev = 3  # after this layer only down azimuth
        dim = cfg.MODEL.BACKBONE.STEM_OUT_CHANNELS  # stem output channel
        dim_copy = 64
        dim_mults = cfg.MODEL.BACKBONE.DIM_MULTS
        resnet_groups = 8

        self.final_res = len(dim_mults) + 1
        self.circular = cfg.MODEL.CIRCULAR_DEPTH

        # dimensions
        dims = [dim, *map(lambda m: dim_copy * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        block_klass = partial(BasicBlock, groups=resnet_groups, circular=self.circular)
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        self.ups = nn.ModuleList([])
        self.ups_var = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            azimuth_only = ind < (num_resolutions - self.num_down_elev - 1) and not is_last

            if is_last:
                upsample = nn.Identity()
            elif azimuth_only:
                upsample = UpsampleAzimuth(dim_in)
            else:
                upsample = Upsample(dim_in)
            self.ups.append(nn.ModuleList([block_klass(dim_out * 2, dim_in), block_klass(dim_in, dim_in), upsample]))

            if is_last:
                upsample = nn.Identity()
            elif azimuth_only:
                upsample = UpsampleAzimuth(dim_in)
            else:
                upsample = Upsample(dim_in)
            self.ups_var.append(
                nn.ModuleList([block_klass(dim_out * 2, dim_in), block_klass(dim_in, dim_in), upsample])
            )

        # final conv
        self.final_conv_depth = nn.Sequential(block_klass(dim * 2, dim), nn.Conv2d(dim, 1, 1))
        self.final_conv_var = nn.Sequential(block_klass(dim * 2, dim), nn.Conv2d(dim, 1, 1))
        self.final_conv_oor = nn.Sequential(block_klass(dim * 2, dim), nn.Conv2d(dim, 1, 1))  # out-of-range

        # loss
        self.bce_loss_func = partial(F.binary_cross_entropy_with_logits, reduction="none")
        self.l1_loss_func = nn.L1Loss(reduction="none")
        self.percep_loss_func = [
            pyiqa.create_metric(
                'lpips',
                device=torch.device('cuda'),
                as_loss=True,
                net='vgg',
                eval_mode=True,
                pnet_tune=False,
            )
        ]  # do not register into model

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (1xHxW depth predictions, {})
        """
        y, features = self.layers(features)

        if self.training:
            return self.losses(y, targets)
        else:
            return y, features

    def layers(self, features):
        x = features[f'res{self.final_res}']
        h = [features['stem']]
        h.extend([features[f'res{i}'] for i in range(2, self.final_res + 1)])

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        x_mean = x
        for i, (block1, block2, upsample) in enumerate(self.ups):
            x_mean = torch.cat((x_mean, h[-1 - i]), dim=1)
            x_mean = block1(x_mean)
            x_mean = block2(x_mean)
            is_last = (i + 1) >= len(self.ups)
            azimuth_only = i < (len(self.ups) - self.num_down_elev - 1) and not is_last
            if azimuth_only:
                x_mean = pad_azimuth(x_mean, padding=1, circular=self.circular)
            elif not is_last:
                x_mean = pad(x_mean, padding=1, circular=self.circular)
            x_mean = upsample(x_mean)

        x_mean = torch.cat((x_mean, h[0]), dim=1)
        ret_mean = self.final_conv_depth(x_mean)
        ret_oor = self.final_conv_oor(x_mean)

        x_var = x
        for i, (block1, block2, upsample) in enumerate(self.ups_var):
            x_var = torch.cat((x_var, h[-1 - i]), dim=1)
            x_var = block1(x_var)
            x_var = block2(x_var)
            is_last = (i + 1) >= len(self.ups_var)
            azimuth_only = i < (len(self.ups_var) - self.num_down_elev - 1) and not is_last
            if azimuth_only:
                x_var = pad_azimuth(x_var, padding=1, circular=self.circular)
            elif not is_last:
                x_var = pad(x_var, padding=1, circular=self.circular)
            x_var = upsample(x_var)

        x_var = torch.cat((x_var, h[0]), dim=1)
        ret_var = self.final_conv_var(x_var)

        return (ret_mean, ret_var, ret_oor), features

    def losses(self, predictions, targets):
        """This is the regression for in-range points + classification problem.
        Also, incorporate the laplace logb uncertainty model and use rf reflection power
        as the porxy for uncertainty label.
        """
        target_depth = targets
        pred_mean, pred_var, pred_oor = predictions

        # get masks and the out-of-range prediction and label
        mask_valid = target_depth > 0
        mask_oor = target_depth > 0.96  # out-of-range
        mask_inr = mask_valid & (~mask_oor)  # in-range
        weight_oor = torch.zeros_like(target_depth)
        weight_oor[mask_oor] = 0.98
        weight_oor[mask_inr] = 0.02
        target_oor = mask_oor.to(torch.float32)

        # the laplacian likelihood
        pred_mean = pred_mean * mask_inr
        pred_var = pred_var * mask_inr
        target_depth *= mask_inr
        l1_depth_error = self.l1_loss_func(pred_mean, target_depth)
        loss_depth = (l1_depth_error * torch.exp(-pred_var) + pred_var).sum() / mask_inr.sum()  # laplace
        loss_percep = self.percep_loss_func[0](pred_mean, target_depth)

        # classification for in-range and our-of-range
        loss_cls = self.bce_loss_func(10 * (pred_oor - 0.96), target_oor, weight_oor).sum() / mask_valid.sum()

        losses = {"loss_depth": loss_depth, "loss_oor_cls": loss_cls, "loss_percep": loss_percep * 0.1}
        return losses


@META_ARCH_REGISTRY.register()
class LaplaceModel(nn.Module):
    """This is the model for add noise for input to get uncertainty.
    Make sure variables are independent by no consecutive conv block.
    Instead of running multiple forward inferences (sampling), this model adapts a mean-
    field-network approach to compute the uncertainty in one forward.
    """

    @configurable
    def __init__(self, *, backbone: nn.Module, depth_head: nn.Module):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            depth_head: a depth estimation head that reconstructs depth maps using backbone features.
            sn_head: a surface normal estimation head that estimates normals using backbone features.
        """
        super().__init__()
        self.backbone = backbone
        self.depth_head = depth_head

        self.dummy_param = nn.Parameter(torch.empty(0))

    @classmethod
    def from_config(cls, cfg):
        backbone = UnetBackbone(cfg)
        depth_head = DepthHead(cfg)
        return {"backbone": backbone, "depth_head": depth_head}

    @property
    def device(self):
        return self.dummy_param.device

    @staticmethod
    def _postprocess(depth_results):
        """post processing of the inference results"""
        processed_results = []
        mean_result, var_result, oor_result = depth_results

        oor_result = torch.sigmoid(10 * (oor_result[0] - 0.96))
        processed_results.append({"depth": mean_result[0], "var": var_result[0], "oor_cls": oor_result})

        return processed_results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], one_forward=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * depth: Tensor, groundtruth depth in (1, H, W) format
                * sn: Tensor, groundtruth normal in (3, H, W) format
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                * sem_seg: semantic segmentation ground truth

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            loss dictionary
        """
        if not self.training:
            return self.inference(batched_inputs, one_forward)

        # images
        rf = [x["image"].to(self.device) for x in batched_inputs]
        rf = ImageList.from_tensors(rf).tensor

        # depth
        gt_depth = [x["depth"].to(self.device) for x in batched_inputs]
        gt_depth = ImageList.from_tensors(gt_depth).tensor

        # 1. backbone forward
        features = self.backbone(rf)

        # 2. depth and sn forward
        depth_losses = self.depth_head(features, gt_depth)

        losses = {}
        losses.update(depth_losses)
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], one_forward=False):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`

        Returns:
            When do_postprocess=True
            list[dict]:
                Each dict is the output for one input image.
                    1. The dict contains a key "instances" whose value is a :class:`Instances`.
                    The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
                    2. The dict contains a key "sem_seg" whose value is a
                    Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        assert not self.training

        # images
        rf = [x["image"].to(self.device) for x in batched_inputs]
        rf = ImageList.from_tensors(rf).tensor

        # initial variance
        if one_forward:
            init_uncertain = [x["init_uncertain"].to(self.device) for x in batched_inputs]
            init_uncertain = ImageList.from_tensors(init_uncertain).tensor
            rf = torch.cat((rf, init_uncertain))

        # Pass through backbone and head
        features = self.backbone(rf)
        depth_results, features = self.depth_head(features, None)

        return depth_results, features

    def batch_inference(self, batched_inputs: Dict[str, torch.Tensor]):
        """Run inference on the given inputs in batch.
        Args:
            batched_inputs Dict[Tensor]: a dictionary of input tensor
        Returns:
            depth_results: the results tuple (depth, laplace_u, oor_cls)
            features: A dictionary of features at different location
        """
        assert not self.training

        # images
        rf = batched_inputs["image"].to(self.device)

        # Pass through backbone and head
        features = self.backbone(rf)
        depth_results, features = self.depth_head(features, None)

        return depth_results, features
