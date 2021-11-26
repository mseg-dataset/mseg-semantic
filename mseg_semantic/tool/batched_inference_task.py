"""
Script to maximize throughput to obtain label maps for an entire dataset.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import mseg.utils.dir_utils as dir_utils

import mseg_semantic.utils.logger_utils as logger_utils
from mseg_semantic.tool.inference_task import InferenceTask
from mseg_semantic.tool.mseg_dataloaders import create_test_loader
from mseg_semantic.utils.avg_meter import AverageMeter
from mseg_semantic.utils.img_path_utils import get_unique_stem_from_last_k_strs


_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logger_utils.get_logger()


def pad_to_crop_sz_batched(
    batch: torch.Tensor, crop_h: int, crop_w: int, mean: float, std: float
) -> Tuple[torch.Tensor, int, int]:
    """preserves NCHW

    Args:
        batch
        crop_h
        crop_w
        mean: RGB mean pixel values
        std: RGB standard deviation of pixel values

    Returns:
        padded_batch
        pad_h_half
        pad_w_half
    """
    n, _, orig_h, orig_w = batch.shape
    pad_h = max(crop_h - orig_h, 0)
    pad_w = max(crop_w - orig_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    padded_batch = torch.zeros((n, 3, crop_h, crop_w))

    start_h = pad_h_half
    # may not be exactly pad_h_half * 2 because of odd num
    end_h = crop_h - (pad_h - pad_h_half)

    start_w = pad_w_half
    # may not be exactly pad_w_half * 2 because of odd num
    end_w = crop_w - (pad_w - pad_w_half)

    padded_batch[:, :, start_h:end_h, start_w:end_w] = batch
    return padded_batch, pad_h_half, pad_w_half


class BatchedInferenceTask(InferenceTask):
    """Subclass of InferenceTask that performs inference over batches from a dataset, instead
    of processing each frame one-by-one. Uses Pytorch (not OpenCV) for all interpolation/padding ops.
    """

    def execute(self) -> None:
        """ """
        logger.info(">>>>>>>>>>>>>> Start inference task >>>>>>>>>>>>>")
        self.model.eval()

        is_dir = os.path.isdir(self.input_file)
        if is_dir:
            # argument is a path to a directory
            self.create_path_lists_from_dir()
            test_loader, self.data_list = create_test_loader(self.args, use_batched_inference=True)
            self.execute_on_dataloader_batched(test_loader)

        else:
            logger.info("Error: Unknown input type")

        logger.info("<<<<<<<<<<< Inference task completed <<<<<<<<<<<<<<")

    def execute_on_dataloader_batched(self, test_loader: torch.utils.data.dataloader.DataLoader):
        """Optimize throughput through the network by batched inference, instead of single image inference"""
        if self.args.save_folder == "default":
            self.args.save_folder = f"{_ROOT}/temp_files/{self.args.model_name}_{self.args.dataset}_universal_{self.scales_str}/{self.args.base_size}"

        os.makedirs(self.args.save_folder, exist_ok=True)
        gray_folder = os.path.join(self.args.save_folder, "gray")
        self.gray_folder = gray_folder

        data_time = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        dir_utils.check_mkdir(self.gray_folder)

        for i, (input, _) in enumerate(test_loader):
            logger.info(f"On batch {i}")
            data_time.update(time.time() - end)

            gray_batch = self.execute_on_batch(input)
            batch_sz = input.shape[0]
            # dump results to disk
            for j in range(batch_sz):
                # determine path for grayscale label map
                image_path, _ = self.data_list[i * self.args.batch_size_val + j]
                if self.args.img_name_unique:
                    image_name = Path(image_path).stem
                else:
                    image_name = get_unique_stem_from_last_k_strs(image_path)
                gray_path = os.path.join(self.gray_folder, image_name + ".png")
                cv2.imwrite(gray_path, gray_batch[j])

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % self.args.print_freq == 0) or (i + 1 == len(test_loader)):
                logger.info(
                    f"Test: [{i+1}/{len(test_loader)}] "
                    f"Data {data_time.val:.3f} (avg={data_time.avg:.3f})"
                    f"Batch {batch_time.val:.3f} (avg={batch_time.avg:.3f})"
                )

    def execute_on_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Only allows for single-scale inference in batch processing mode for now"""
        start = time.time()
        # single-scale, do addition and argmax on CPU, interp back to native resolution
        logits = self.scale_process_cuda_batched(batch, self.args.native_img_h, self.args.native_img_w)
        predictions = torch.argmax(logits, axis=1)
        end = time.time()
        duration = end - start
        print(f"Took {duration:.3f} sec. to run batch")

        predictions = predictions.data.cpu().numpy()
        gray_batch = np.uint8(predictions)

        return gray_batch

    def scale_process_cuda_batched(
        self, batch: torch.Tensor, native_h: int, native_w: int, stride_rate: float = 2 / 3
    ) -> np.ndarray:
        """Note: we scale up the image to fit exactly within the crop size

        Args:
            batch: NCHW tensor
            native_h: image height @ native/raw image resolution, as found originally on disk
            native_w: image width @ native/raw image resolution, as found originally on disk
            stride_rate:
        
        Returns:
            NCHW tensor where dimensions are (N, num_classes, H, W)
        """
        n, _, orig_h, orig_w = batch.shape

        padded_batch, pad_h_half, pad_w_half = pad_to_crop_sz_batched(
            batch, self.crop_h, self.crop_w, self.mean, self.std
        )
        prediction_crops = self.net_process_batched(padded_batch)

        # disregard predictions from padded portion of image
        prediction_crops = prediction_crops[:, :, pad_h_half : pad_h_half + orig_h, pad_w_half : pad_w_half + orig_w]
        prediction_crops = torch.nn.functional.interpolate(
            prediction_crops, size=(native_h, native_w), mode="bilinear", align_corners=True
        )
        return prediction_crops

    def net_process_batched(self, batch: torch.Tensor, flip: bool = True) -> torch.Tensor:
        """Feed input through the network

        In addition to running a crop through the network, we can flip the crop
        horizontally, run both crops through the network, and then average them appropriately.
        Afterwards, we apply softmax, then convert the prediction to the label taxonomy.

        Args:
            batch: NCHW
            flip: whether to average with flipped patch output

        Returns:
            output: Pytorch tensor representing network prediction in evaluation taxonomy
                (not necessarily model taxonomy), (N, num_classes, H, W)
        """
        input = batch.float().cuda()

        if flip:
            # add another example to batch dimension, that is the flipped crop
            input = torch.cat([input, input.flip(3)], 0)
        with torch.no_grad():
            output = self.model(input)
        n, _, h_i, w_i = input.shape
        _, _, h_o, w_o = output.shape

        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode="bilinear", align_corners=True)

        prediction_conversion_req = self.model_taxonomy != self.eval_taxonomy
        if prediction_conversion_req:
            # either (model_taxonomy='naive', eval_taxonomy='test_dataset')
            # Or (model_taxonomy='universal', eval_taxonomy='test_dataset')
            output = self.tc.transform_predictions_test(output, self.args.dataset)

        else:
            # model and eval. taxonomy match, so no conversion needed
            assert self.model_taxonomy in ["universal", "test_dataset"]
            output = self.softmax(output)

        if flip:
            # take back out the flipped crop, correct its orientation, and average result
            split_idx = n // 2
            output = (output[:split_idx] + output[split_idx:].flip(3)) / 2

        return output
