import numpy as np
import os, json
import glob
import configargparse


def main(online=False):

    folder_path = './output'
    if online:
        category = 'online'
    else:
        category = 'offline'
    subfolders = [
        name for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name)) and (category in name)
    ]

    subfolders = sorted(subfolders)

    assert len(subfolders) == 14

    names = []
    acc_results = []
    comp_results = []
    pre_results = []
    rec_results = []
    fs_results = []
    ate_results = []
    rpe_results = []

    for subfolder in subfolders:
        names_before = subfolder.split('_online')[0]
        json_paths = glob.glob(os.path.join(os.path.join(folder_path, subfolder), "*result.json"))
        if len(json_paths) == 0:
            continue
        with open(json_paths[0], 'r') as file:
            data = json.load(file)

        acc = data['global_alignment']['mean accuracy (m)']
        comp = data['global_alignment']['mean completeness (m)']
        pre = data['global_alignment']['Precision']
        rec = data['global_alignment']['Recall']
        fs = data['global_alignment']['F-score']
        ate = data['ATE(MEAN: cm)']
        rpe = data['RPE_ROT(MEAN: deg)']

        names.append(names_before)
        acc_results.append(acc)
        comp_results.append(comp)
        pre_results.append(pre)
        rec_results.append(rec)
        fs_results.append(fs)
        ate_results.append(ate)
        rpe_results.append(rpe)

    assert len(names) == 14

    print(f'ATE: {np.mean(ate_results):.2f}')
    print(f'RPE: {np.mean(rpe_results):.2f}')
    print(f'Acc: {np.mean(acc_results)*100:.2f}')
    print(f'Comp: {np.mean(comp_results)*100:.2f}')
    print(f'F-Score: {np.mean(fs_results)*100:.2f}%')


def parse_arguments():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--online", action="store_true", help="choose to ignore log")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    main(online=args.online)
