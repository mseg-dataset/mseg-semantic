#!/usr/bin/python3

import os
from pathlib import Path
from types import SimpleNamespace

import mseg_semantic.tool.universal_demo as universal_demo

REPO_ROOT_ = Path(__file__).resolve().parent.parent

# Replace this variables with your own path to run integration tests.
INTEGRATION_TEST_OUTPUT_DIR = "/srv/scratch/jlambert30/MSeg/mseg-semantic/integration_test_data"
# Copy the mseg-3m-1080p model there
MSEG_3M_1080p_MODEL_PATH = f"{INTEGRATION_TEST_OUTPUT_DIR}/mseg-3m.pth"


def test_run_universal_demo() -> None:
    """
    Ensure demo script works correctly
    base_sizes=(
            #360
            720
            #1080

    python -u mseg_semantic/tool/test_universal_tax.py --config=${config_fpath}
            dataset ${dataset_name} model_path ${model_fpath} model_name ${model_name}
    """
    for base_size in [360, 720, 1080]:
        # Args that would be provided in command line and in config file
        d = {
            "config": f"{REPO_ROOT_}/mseg_semantic/config/test/default_config_${base_size}_ms.yaml",
            #'model_path': f'{_ROOT}/pretrained-semantic-models/${model_name}/${model_name}.pth',
            "model_path": MSEG_3M_1080p_MODEL_PATH,
            "input_file": f"{REPO_ROOT_}/tests/test_data/demo_images",
            "model_name": "mseg-3m",
            "dataset": "default",
            "base_size": base_size,
            "test_h": 713,
            "test_w": 713,
            "scales": [1.0],
            "save_folder": "default",
            "arch": "hrnet",
            "index_start": 0,
            "index_step": 0,
            "workers": 16,
        }
        args = SimpleNamespace(**d)
        use_gpu = True
        print(args)
        universal_demo.run_universal_demo(args, use_gpu)

        # assert result files exist
        results_dir = f"{REPO_ROOT_}/temp_files/mseg-3m_default_universal_ss/{base_size}/gray"
        fnames = [
            "242_Maryview_Dr_Webster_0000304.png",
            "bike_horses.png",
            "PrivateLakefrontResidenceWoodstockGA_0000893.png",
        ]
        for fname in fnames:
            gray_fpath = f"{results_dir}/{fname}"
            print(gray_fpath)
            assert Path(gray_fpath).exists()
            os.remove(gray_fpath)


if __name__ == "__main__":
    test_run_universal_demo()
