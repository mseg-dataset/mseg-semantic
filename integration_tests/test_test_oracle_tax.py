#!/usr/bin/python3

from pathlib import Path
from types import SimpleNamespace

from mseg_semantic.scripts.collect_results import parse_result_file
from mseg_semantic.tool.test_oracle_tax import test_oracle_taxonomy_model

REPO_ROOT_ = Path(__file__).resolve().parent.parent


# Replace this variables with your own path to run integration tests.
INTEGRATION_TEST_OUTPUT_DIR = "/srv/scratch/jlambert30/MSeg/mseg-semantic/integration_test_data"
# Copy the mseg-3m-1080p model there
CAMVID_MODEL_PATH = f"{INTEGRATION_TEST_OUTPUT_DIR}/camvid-11-1m.pth"


def test_evaluate_oracle_tax_model() -> None:
    """Ensure oracle model testing script works correctly.

    base_sizes=(
            #360
            720
            #1080

    python -u mseg_semantic/tool/test_oracle_tax.py --config=${config_fpath}
            dataset ${dataset_name} model_path ${model_fpath} model_name ${model_name}
    """
    base_size = 1080
    d = {
        "dataset": "camvid-11",
        "config": f"{REPO_ROOT_}/mseg_semantic/config/test/default_config_${base_size}_ss.yaml",
        "model_path": CAMVID_MODEL_PATH,
        "model_name": "mseg-3m-1080p",
        "input_file": "default",
        "base_size": base_size,
        "test_h": 713,
        "test_w": 713,
        "scales": [1.0],
        "save_folder": "default",
        "arch": "hrnet",
        "index_start": 0,
        "index_step": 0,
        "workers": 16,
        "has_prediction": False,
        "split": "val",
        "vis_freq": 20,
    }
    args = SimpleNamespace(**d)
    use_gpu = True
    test_oracle_taxonomy_model(args, use_gpu)

    # Ensure that results match paper
    result_file_path = INTEGRATION_TEST_OUTPUT_DIR
    result_file_path += f"/camvid-11-1m/camvid-11/{base_size}/ss/results.txt"
    assert Path(result_file_path).exists()
    mIoU = parse_result_file(result_file_path)

    print(f"mIoU: {mIoU}")
    # single-scale result
    assert mIoU == 78.79

    OKGREEN = "\033[92m"
    ENDC = "\033[0m"
    print(OKGREEN + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>" + ENDC)
    print(OKGREEN + "Oracle model evalution passed successfully" + ENDC)
    print(OKGREEN + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>" + ENDC)


if __name__ == "__main__":
    test_evaluate_oracle_tax_model()
