"""run multiple distribution and save the result. For the Hybrid version."""

import torch
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from configs import get_cfg
from dataloader import dataset
from engine.trainer import get_trainer_class
from evaluation.uncertainty_metrics import merge_uncertainty_files, analyze_uncertainty
import custom_meta_arch  # need this to register meta-arch


def eval_model(
    model, dataloader, save_name: str, n_sample: int, centers=(10,), widths=(0,), batch_size=16, downsample=5
):
    """Evaluate the model with perturbation, save the middle-results in files for later analysis of uncertainty.
    Args:
        model: the neural network model
        dataloader: the test dataloader
        save_name: the name of the saved temp file
        n_sample: how many random samples
        centers: the Gaussian noise center, x10 value
        widths: the width of the Gaussian noise center, x10 value
        batch_size: the inference batch size
        downsample: downsample factor
    """
    noise_bounds = [(c - w, c + w) for c in centers for w in widths]
    abs_error = {0: []}
    sampled_uncertainty = {N: [] for N in noise_bounds}
    lapace_uncertainty = {N: [] for N in noise_bounds}
    lapace_uncertainty[0] = []

    with torch.no_grad():
        for i, inputs in tqdm(enumerate(dataloader), desc=save_name, total=len(dataloader)):
            if i % downsample != 0:
                continue

            gt_depth = inputs[0]['depth'].cuda() * 10
            rf_image = inputs[0]['image'].cuda().expand(batch_size, -1, -1, -1)

            # pred deoth without noise
            outputs = model(inputs)[0]
            pred_depth = outputs[0] * 10
            pred_var = torch.exp(outputs[1]) * 10

            # save the prediction points
            valid_mask = (gt_depth > 0) & (gt_depth < 9.6)
            abs_error[0].append(torch.abs(gt_depth[valid_mask] - pred_depth[0][valid_mask]).cpu())
            lapace_uncertainty[0].append(pred_var[0][valid_mask].cpu())

            # inference with noise
            for noise_bound in noise_bounds:
                noise_sigmas = torch.linspace(*noise_bound, n_sample, device='cuda').view(-1, 1, 1, 1) / 10
                pred_depths = []
                pred_vars = []

                for batch_ind in range(0, n_sample, batch_size):
                    inputs = {
                        'image': rf_image
                        + torch.randn_like(rf_image) * noise_sigmas[batch_ind : batch_ind + batch_size]
                    }
                    outputs = model.batch_inference(inputs)[0]
                    pred_depths.append(outputs[0])
                    pred_vars.append(outputs[1])

                pred_depths = torch.cat(pred_depths) * 10
                pred_vars = torch.exp(torch.cat(pred_vars)) * 10
                pred_vars = 2 * pred_vars**2  # 2b^2

                sampled_uncertainty[noise_bound].append(torch.std(pred_depths, dim=0)[valid_mask].cpu())
                expected_var = torch.mean(pred_vars, dim=0)[valid_mask]
                lapace_uncertainty[noise_bound].append(expected_var.cpu())

    abs_error = {k: torch.cat(v) for k, v in abs_error.items()}
    sampled_uncertainty = {k: torch.cat(v) for k, v in sampled_uncertainty.items()}
    lapace_uncertainty = {k: torch.cat(v) for k, v in lapace_uncertainty.items()}

    final_result = {
        'abs_error': abs_error,
        'sampled_uncertainty': sampled_uncertainty,
        'lapace_uncertainty': lapace_uncertainty,
    }
    torch.save(final_result, save_name)


def analyze_result(n_sample: int):
    """Analyze the uncertainty result
    Args:
        n_sample: how many random samples used for inference
    """
    assert n_sample in {16, 32}, "Incorrect arguments"

    print(f"Analyzing results for OursH-{n_sample}")
    result_paths = Path("./temp").glob(f"laplace_*_N{n_sample}.pth")
    error_uctt = merge_uncertainty_files(sorted(result_paths))
    l1_err_result, uq_std_result, uq_mean_result, uq_mix_result = analyze_uncertainty(error_uctt, utype='laplace')

    # print results
    nll = uq_mix_result[(10, 10)]["nll_laplace"]
    ause = uq_mix_result[(10, 10)]["ause"] * 10
    print(f"  OursH-{n_sample}: NLL = {nll:.3f}, AUSE = {ause:.3f}")


def main():
    log_paths = sorted(Path("./logs").glob("laplace_*"))
    Path("./temp").mkdir(parents=True, exist_ok=True)
    is_data_registered = False

    # load model and evaluate
    for log_path in log_paths:
        # get and merge config file
        cfg = get_cfg()
        cfg.merge_from_file(log_path / "config.yaml")
        cfg.MODEL.PIXEL_MEAN = [0 for _ in range(256)]
        cfg.MODEL.PIXEL_STD = [1 for _ in range(256)]
        cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        cfg.OUTPUT_DIR = "./logs/test"

        # load model checkpoints
        model = build_model(cfg)
        DetectionCheckpointer(model).load(str(log_path / "model_final.pth"))
        model.eval()
        print("Model loaded")

        # Dataloader, meta data and evaluator
        if not is_data_registered:
            dataset.register_dataset(cfg)
            is_data_registered = True
        trainer = get_trainer_class(cfg)(cfg)
        dataloader = trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
        print("Data loaded")

        # evaluate for N=16 and N=32
        eval_model(model, dataloader, f"./temp/{log_path.name}_N16.pth", n_sample=16)
        eval_model(model, dataloader, f"./temp/{log_path.name}_N32.pth", n_sample=32)

    analyze_result(n_sample=16)
    analyze_result(n_sample=32)


if __name__ == '__main__':
    main()
