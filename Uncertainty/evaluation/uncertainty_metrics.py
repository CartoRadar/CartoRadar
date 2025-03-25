"""Metrics for the uncertainty evaluation"""

import numpy as np
import torch
from typing import Dict, List


def UQ_metrics(abs_error: torch.Tensor, uncertainty_b: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute the UQ metrics.
    For calibration curves, we assume the distribution is Laplacian
    Args:
        abs_error: 1D tensor, the absolute depth error, in meter
        uncertainty_b: 1D tensor, the predicted uncertainty, or the std of a distribution
    Returns:
        results: a dictionary containing all the metrics and intermediate results
    """
    # sort variance and error
    sorted_var, sort_inds = torch.sort(uncertainty_b)
    sorted_err = abs_error[sort_inds]
    rmse_func = lambda x: torch.sqrt(torch.mean(x**2, dim=-1))

    # the sparsification plot
    end_inds = torch.flip(torch.linspace(0, len(sorted_err), 51).int(), (-1,))
    end_inds = end_inds[:-1]  # remove the last one
    fractions = torch.linspace(0, 1, 51)[:-1]
    rmse_our = torch.tensor([rmse_func(sorted_err[0:ind]) for ind in end_inds])
    #
    sorted_ora_err, _ = torch.sort(sorted_err)
    rmse_oracle = torch.tensor([rmse_func(sorted_ora_err[0:ind]) for ind in end_inds])
    ause = torch.trapz(rmse_our, fractions) - torch.trapz(rmse_oracle, fractions)

    # negative log-likelihood (NLL)
    nll_laplace = torch.log(2**0.5 * uncertainty_b) + 2**0.5 * abs_error / uncertainty_b
    nll_laplace = torch.mean(nll_laplace)

    results = {
        # sparsification curve
        'fractions': fractions,
        'rmse_oracle': rmse_oracle,
        'rmse_our': rmse_our,
        'ause': ause,
        # negative log-likelihood (NLL)
        'nll_laplace': nll_laplace,
    }
    return results


def depth_metrics(abs_error: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute the depth metrics.
    For calibration curves, we assume the distribution is Laplacian
    Args:
        abs_error: 1D tensor, the absolute depth error, in meter
    Returns:
        results: a dictionary containing all the metrics and intermediate results
    """
    results = {
        'l1_mean': torch.mean(abs_error),
        'l1_median': torch.median(abs_error),
        'l1_80': torch.tensor(np.quantile(abs_error, 0.8), dtype=torch.float32),
        'l1_90': torch.tensor(np.quantile(abs_error, 0.9), dtype=torch.float32),
        'l1_95': torch.tensor(np.quantile(abs_error, 0.95), dtype=torch.float32),
    }
    return results


def analyze_uncertainty(error_uctt: Dict[str, torch.Tensor], utype: str):
    """Analyze the uncertainty inference data.
    Param:
        error_uctt: the dictionary containing the model evaluation results generated from the
            scripts/eval_sampling_uncertainty.py or scripts/eval_laplacian_uncertainty.py
        utype: should be one of {ensemble, dropout, gauss, laplace}
    """
    if utype == 'laplace':
        # Laplacian and mix uncertainty
        abs_error = error_uctt['abs_error']
        sampled_uncertainty = error_uctt['sampled_uncertainty']  # std
        lapace_uncertainty = error_uctt['lapace_uncertainty']  # {0:b, others:var}
        lapace_uncertainty[0] = 2 * lapace_uncertainty[0] ** 2

        l1_err_result = {k: depth_metrics(v) for k, v in abs_error.items()}
        uq_std_result = {k: UQ_metrics(abs_error[0], v) for k, v in sampled_uncertainty.items()}
        uq_mean_result = {k: UQ_metrics(abs_error[0], torch.sqrt(v)) for k, v in lapace_uncertainty.items()}
        uq_mix_result = {
            k: UQ_metrics(abs_error[0], torch.sqrt(sampled_uncertainty[k] ** 2 + lapace_uncertainty[k]))
            for k in sampled_uncertainty.keys()
        }
    elif utype == 'depth':
        abs_error = error_uctt['abs_error']
        sampled_uncertainty = error_uctt['sampled_uncertainty']  # std

        l1_err_result = l1_err_result = {k: depth_metrics(v) for k, v in abs_error.items()}
        uq_std_result = {k: UQ_metrics(abs_error[0], v) for k, v in sampled_uncertainty.items()}
        uq_mean_result = uq_mix_result = {}
    else:
        raise NameError(f'the utype={utype} is not supported.')

    return l1_err_result, uq_std_result, uq_mean_result, uq_mix_result


def merge_uncertainty_files(file_names: List[str]) -> Dict[str, torch.Tensor]:
    """Merge separate model evaluation results (different buildings) into a whole one
    for the same uncertainty method.
    Args:
        file_names: A list of results names that need to be merged
    Returns:
        merged: the merged evaluation result
    """
    files = [torch.load(file_name) for file_name in file_names]
    merged = {}

    for k0, v0 in files[0].items():
        merged[k0] = {}
        for k1 in v0:
            tensors = [file[k0][k1] for file in files]
            merged[k0][k1] = torch.cat(tensors)
    return merged
