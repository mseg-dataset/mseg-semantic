#!/usr/bin/python3

"""After inference with many models and many datasets, generate summary tables of results.

Results are expected in the following folder structure, 

    {RESULTS_BASE_ROOT}/{M}/{M}/{D}/{RESOLUTION}
    where "M" is a model name, and "D" is a dataset name, e.g.

    {RESULTS_BASE_ROOT}/camvid-11-1m/camvid-11-1m/camvid-11/360/ss/results.txt 

Note: models trained on a single training dataset are trained in the universal taxonomy, to avoid
having to hand-specify 7 * 6 = 42 train to test taxonomy mappings.
"""

import argparse
import os
from enum import Enum
from typing import List

import numpy as np

# import scipy.stats.hmean as hmean
from scipy.stats.mstats import gmean

ROW_LEFT_JUSTIFY_OFFSET = 50


class PrintOutputFormat(str, Enum):
    """syntax for STDOUT table formatting."""

    LaTeX: str = "LaTeX"
    MARKDOWN: str = "MARKDOWN"


RESULTS_BASE_ROOT = "/srv/scratch/jlambert30/MSeg/pretrained-semantic-models"

# zero_shot_datasets
ZERO_SHOT_DATASETS = ["voc2012", "pascal-context-60", "camvid-11", "wilddash-19", "kitti-19", "scannet-20"]

training_datasets = [
    "coco-panoptic-133_universal",
    "ade20k-150_universal",
    "mapillary-public65_universal",
    "idd-39_universal",
    "bdd_universal",
    "cityscapes-19_universal",
    "sunrgbd-37_universal",
]

# universal taxonomy models
UNIVERSAL_TAX_MODEL_FNAMES = [
    "coco-panoptic-133-1m",
    "ade20k-150-1m",
    "mapillary-65-1m",
    "idd-39-1m",
    "bdd-1m",
    "cityscapes-19-1m",
    "sunrgbd-37-1m",
    "mseg-1m",
    "mseg-unrelabeled-1m",
    "mseg-mgda-1m",
    "mseg-3m-480p",
    "mseg-3m-720p",
    "mseg-3m",
]

UNIVERSAL_TAX_MODEL_PRETTYPRINT_NAMES = [
    "COCO",
    "ADE20K",
    "Mapillary",
    "IDD",
    "BDD",
    "Cityscapes",
    "SUN RGBD",
    "MSeg-1m",
    "MSeg-1m-w/o relabeling",
    "MSeg-MGDA-1m",
    "MSeg-3m-480p",
    "MSeg-3m-720p",
    "MSeg-3m-1080p",
]
# 'naive'

# oracle taxonomy model filenames
ORACLE_MODELS = ["voc2012-1m", "pascal-context-60-1m", "camvid-11-1m", "kitti-19-1m", "scannet-20-1m"]
# formal names for models above
ORACLE_NAMES = ["VOC Oracle", "PASCAL Context Oracle", "Camvid Oracle", "KITTI Oracle", "ScanNet Oracle"]

# oracle-trained datasets
ORACLE_DATASETS = ["voc2012", "pascal-context-60", "camvid-11", "kitti-19", "scannet-20"]


VERBOSE = False


def parse_result_file(result_file: str) -> float:
    """Load mIoU from .txt result file."""
    if not os.path.isfile(result_file):
        if VERBOSE:
            print(result_file + " does not exist!")
        return 100000

    with open(result_file, "r") as f:
        tmp = f.readlines()[0]
        miou = tmp.split("/")[2].split(" ")[-1]
        miou = float(miou) * 100
        # miou = "{:.1f}".format(miou)
    return miou


def parse_folder(folder: str, resolution: str, scale: str) -> float:
    """Scrape results from resolution-specific .txt files in subfolders.

    # folder containing subfolders as 360/720/1080

    Args:
        folder: path to folder.
        resolution: either "360", "720", "1080", or "max", which represents the best result
            over all 3 aforementioned resolutions.
        scale: string representing inference scale option,
            either 'ss' or 'ms' (single-scale or multi-scale)

    Returns:
        mIoU at this resolution.
    """
    mious = []
    resolutions = ["360", "720", "1080"]
    for b in resolutions:
        result_file = os.path.join(folder, b, scale, "results.txt")
        # parse_file
        mious.append(parse_result_file(result_file))

    if resolution == "max":
        max_miou = max(mious)
        return max_miou

    else:
        val_idx = resolutions.index(resolution)
        return mious[val_idx]


def harmonic_mean(x: np.ndarray) -> float:
    """Compute the harmonic mean of a set of numbers.

    1. Take the reciprocal of all numbers in the dataset
    2. Find the arithmetic mean of those reciprocals
    3. Take the reciprocal of that number
    """
    n = x.shape[0]
    mean = np.sum(1 / x) / n
    return 1 / mean


def arithmetic_mean(x: np.ndarray) -> float:
    """Compute the arithmetic mean of a set of numbers."""
    return np.mean(x)


def geometric_mean(x: np.ndarray) -> float:
    """Compute the geometric mean of a set of numbers."""
    n = x.shape[0]
    prod = np.prod(x)
    return prod ** (1 / n)


def dump_results_latex(name: str, results: List[float]) -> None:
    """Dump a table to STDOUT in LaTeX syntax."""
    results = ["{:.1f}".format(r).rjust(5) for r in results]
    results = ["$ " + r + " $" for r in results]
    print(name.rjust(ROW_LEFT_JUSTIFY_OFFSET), " & ", " & ".join(results) + "\\\\")


def dump_results_markdown(name: str, results: List[float]) -> None:
    """Dump a table to STDOUT in Markdown syntax."""
    results = ["{:.1f}".format(r).rjust(5) for r in results]
    results = ["| " + r + "" for r in results]
    print(name.rjust(ROW_LEFT_JUSTIFY_OFFSET), "  ", " ".join(results) + "|")


def collect_naive_merge_results_at_res(
    resolution: str, scale: str, output_format: PrintOutputFormat, mean_type: str = "harmonic"
) -> None:
    """For all test datasets, aggregate the results of the naive-merge model at a specific evaluation resolution
    (from resolution-specific .txt files in subfolders).

    Args:
        resolution: either "360", "720", "1080", or "max", which represents the best result
            over all 3 aforementioned resolutions.
        scale: string representing inference scale option,
            either 'ss' or 'ms' (single-scale or multi-scale)
        output_format: syntax for STDOUT result table formatting.
    """
    print(" " * 60, (" " * 5).join(ZERO_SHOT_DATASETS), " " * 10 + "mean")
    m = "mseg-naive-baseline-1m"

    results = []
    for d in ZERO_SHOT_DATASETS:
        folder = f"{RESULTS_BASE_ROOT}/{m}/{m}/{d}"
        miou = parse_folder(folder, resolution, scale)
        results.append(miou)

    tmp_results = np.array(results)
    if mean_type == "harmonic":
        results.append(harmonic_mean(tmp_results))

    if output_format == PrintOutputFormat.LaTeX:
        dump_results_latex("Naive Merge", results)
    elif output_format == PrintOutputFormat.MARKDOWN:
        dump_results_markdown("Naive Merge", results)


def collect_oracle_results_at_res(resolution: str, scale: str, output_format: PrintOutputFormat) -> None:
    """For all test datasets (except WildDash), aggregate the results of the corresponding
    oracle models at a specific evaluation resolution (from resolution-specific .txt files in subfolders).

    Args:
        resolution: either "360", "720", "1080", or "max", which represents the best result
            over all 3 aforementioned resolutions.
        scale: string representing inference scale option,
            either 'ss' or 'ms' (single-scale or multi-scale)
        output_format: syntax for STDOUT result table formatting.
    """
    results = []
    print(" " * 60, (" " * 5).join(o_datasets), " " * 10 + "mean")
    for m, name, d in zip(ORACLE_MODELS, ORACLE_NAMES, ORACLE_DATASETS):
        folder = f"{RESULTS_BASE_ROOT}/{m}/{m}/{d}"
        miou = parse_folder(folder, resolution, scale)
        results.append(miou)

    if output_format == PrintOutputFormat.LaTeX:
        dump_results_latex("Oracle", results)
    elif output_format == PrintOutputFormat.MARKDOWN:
        dump_results_markdown("Oracle", results)


def collect_results_at_res(
    datasets: List[str], resolution: str, scale: str, output_format: PrintOutputFormat, mean_type: str = "harmonic"
) -> None:
    """Collect results from inference at a single evaluation resolution.

    In the result table, each row will represent a single model. Columns represent different evaluation datasets.
    """
    print(" " * 60, (" " * 5).join(datasets), " " * 10 + "mean")
    for m, name in zip(UNIVERSAL_TAX_MODEL_FNAMES, UNIVERSAL_TAX_MODEL_PRETTYPRINT_NAMES):
        results = []
        for d in datasets:

            # rename
            if ("mseg" in m) and ("unrelabeled" not in m) and (d in training_datasets):
                d += "_relabeled"

            folder = f"{RESULTS_BASE_ROOT}/{m}/{m}/{d}"
            miou = parse_folder(folder, resolution, scale)
            results.append(miou)

        tmp_results = np.array(results)
        if mean_type == "harmonic":
            # results.append([len(tmp_results) / np.sum(1.0/np.array(tmp_results))])
            results.append(harmonic_mean(tmp_results))
        elif mean_type == "arithmetic":
            results.append(arithmetic_mean(tmp_results))
        elif mean_type == "geometric":
            results.append(geometric_mean(tmp_results))
        else:
            print("Unknown mean type")
            exit()
        if output_format == PrintOutputFormat.LaTeX:
            dump_results_latex(name, results)
        elif output_format == PrintOutputFormat.MARKDOWN:
            dump_results_markdown(name, results)


def collect_zero_shot_results(scale: str, output_format: PrintOutputFormat) -> None:
    """Collect the results of zero-shot cross-dataset generalization experiments."""
    # 'ms' vs. 'ss'
    for resolution in ["360", "720", "1080", "max"]:  #  '480', '2160',
        print(f"At resolution {resolution}")
        collect_results_at_res(ZERO_SHOT_DATASETS, resolution, scale, output_format)


def collect_naive_merge_results(scale: str, output_format: PrintOutputFormat) -> None:
    """Collect results of our single naive-merge model."""
    for resolution in ["360", "720", "1080", "max"]:
        print(f"At resolution {resolution}")
        collect_naive_merge_results_at_res(resolution, scale, output_format)


def collect_oracle_results(scale: str, output_format: PrintOutputFormat) -> None:
    """Collect the results of oracle-trained models.

    `Oracle' means trained on train split of test dataset, tested on test split of same test dataset.
    """
    # 'ms' vs. 'ss'
    for resolution in ["360", "720", "1080", "max"]:  #  '480', '2160',
        print(f"At resolution {resolution}")
        collect_oracle_results_at_res(resolution, scale, output_format)


def collect_training_dataset_results(scale: str, output_format: PrintOutputFormat) -> None:
    """ """
    # 'ss' only
    for resolution in ["360", "720", "1080", "max"]:  #  '480', '2160',
        print(f"At resolution {resolution}")
        collect_results_at_res(training_datasets, resolution, scale, output_format)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime",
        required=True,
        type=str,
        help="Testing regime -- either `zero_shot`, `oracle`, `training_datasets`, or `naive_merge' ",
        choices=["zero_shot", "oracle", "training_datasets", "naive_merge"],
    )
    parser.add_argument(
        "--scale", required=True, type=str, help="ss (single-scale) or ms (multi-scale)", choices=["ss", "ms"]
    )
    parser.add_argument(
        "--output_format",
        required=True,
        type=str,
        help="syntax for STDOUT result table formatting (latex or markdown)",
        choices=["latex", "markdown"],
    )
    args = parser.parse_args()
    print(args)

    if args.output_format == "latex":
        output_format = PrintOutputFormat.LaTeX

    elif args.output_format == "markdown":
        output_format = PrintOutputFormat.MARKDOWN

    if args.regime == "zero_shot":
        collect_zero_shot_results(args.scale, output_format)

    elif args.regime == "oracle":
        collect_oracle_results(args.scale, output_format)

    elif args.regime == "training_datasets":
        collect_training_dataset_results(args.scale, output_format)

    elif args.regime == "naive_merge":
        collect_naive_merge_results(args.scale, output_format)

    else:
        print("Unknown testing regime")
