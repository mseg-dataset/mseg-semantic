#!/usr/bin/python3

import logging
import os
import pdb
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

import mseg.utils.names_utils as names_utils
import mseg.utils.resize_util as resize_util
from mseg.utils.dir_utils import check_mkdir, create_leading_fpath_dirs
from mseg.utils.mask_utils_detectron2 import Visualizer
from mseg.taxonomy.taxonomy_converter import TaxonomyConverter
from mseg.taxonomy.naive_taxonomy_converter import NaiveTaxonomyConverter

import mseg_semantic.utils.normalization_utils as normalization_utils
from mseg_semantic.model.pspnet import PSPNet
from mseg_semantic.utils.avg_meter import AverageMeter
from mseg_semantic.utils.cv2_video_utils import VideoWriter, VideoReader
from mseg_semantic.utils import dataset, transform, config
from mseg_semantic.utils.img_path_utils import dump_relpath_txt, get_unique_stem_from_last_k_strs
from mseg_semantic.tool.mseg_dataloaders import create_test_loader
from mseg_semantic.utils.logger_utils import get_logger


"""
Given a specified task, run inference on it using a pre-trained network.
Used for demos, and for testing on an evaluation dataset.

If projecting universal taxonomy into a different evaluation taxonomy,
the argmax comes *after* the linear mapping, so that probabilities can be
summed first.

Note: "base size" should be the length of the shorter side of the desired
inference image resolution. Note that the official PSPNet repo 
(https://github.com/hszhao/semseg/blob/master/tool/test.py) treats
base_size as the longer side, which we found less intuitive given
screen resolution is generally described by shorter side length.

"base_size" is a very important parameter and will
affect results significantly.

There are 4 possible configurations for 
(model_taxonomy, eval_taxonomy):

(1) model_taxonomy='universal', eval_taxonomy='universal'
	Occurs when:
	(a) running demo w/ universal output
	(b) evaluating universal models on train datasets
	in case (b), training 'val' set labels are converted to univ.

(2) model_taxonomy='universal', eval_taxonomy='test_dataset':
	(a) generic zero-shot cross-dataset evaluation case

(3) model_taxonomy='test_dataset', eval_taxonomy='test_dataset'
	(a) evaluating `oracle` model -- trained and tested on same
		dataset (albeit on separate splits)

(4) model_taxonomy='naive', eval_taxonomy='test_dataset':
	Occurs when:
	(a) evaluating naive unified model on test datasets
"""

_ROOT = Path(__file__).resolve().parent.parent.parent


logger = get_logger()


def resize_by_scaled_short_side(image: np.ndarray, base_size: int, scale: float) -> np.ndarray:
    """Equivalent to ResizeShort(), but functional, instead of OOP paradigm, and w/ scale param.

    Args:
        image: Numpy array of shape ()
        scale: scaling factor for image

    Returns:
        image_scaled:
    """
    h, w, _ = image.shape
    short_size = round(scale * base_size)
    new_h = short_size
    new_w = short_size
    # Preserve the aspect ratio
    if h > w:
        new_h = round(short_size / float(w) * h)
    else:
        new_w = round(short_size / float(h) * w)
    image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image_scaled


def pad_to_crop_sz(
    image: np.ndarray, crop_h: int, crop_w: int, mean: Tuple[float, float, float]
) -> Tuple[np.ndarray, int, int]:
    """
    Network input should be at least crop size, so we pad using mean values if
    provided image is too small. No rescaling is performed here.

    We use cv2.copyMakeBorder to copy the source image into the middle of a
    destination image. The areas to the left, to the right, above and below the
    copied source image will be filled with extrapolated pixels, in this case the
    provided mean pixel intensity.

    Args:
        image:
        crop_h: integer representing crop height
        crop_w: integer representing crop width

    Returns:
        image: Numpy array of shape (crop_h x crop_w) representing a
               square image, with short side of square is at least crop size.
        pad_h_half: half the number of pixels used as padding along height dim
        pad_w_half: half the number of pixels used as padding along width dim
    """
    orig_h, orig_w, _ = image.shape
    pad_h = max(crop_h - orig_h, 0)
    pad_w = max(crop_w - orig_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            src=image,
            top=pad_h_half,
            bottom=pad_h - pad_h_half,
            left=pad_w_half,
            right=pad_w - pad_w_half,
            borderType=cv2.BORDER_CONSTANT,
            value=mean,
        )
    return image, pad_h_half, pad_w_half


def imread_rgb(img_fpath: str) -> np.ndarray:
    """
    Returns:
        RGB 3 channel nd-array with shape (H, W, 3)
    """
    bgr_img = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(rgb_img)
    return rgb_img


class InferenceTask:
    def __init__(
        self,
        args,
        base_size: int,
        crop_h: int,
        crop_w: int,
        input_file: str,
        model_taxonomy: str,
        eval_taxonomy: str,
        scales: List[float],
        use_gpu: bool = True,
    ) -> None:
        """
        We always use the ImageNet mean and standard deviation for normalization.
        mean: 3-tuple of floats, representing pixel mean value
        std: 3-tuple of floats, representing pixel standard deviation

        'args' should contain at least 5 fields (shown below).
        See brief explanation at top of file regarding taxonomy arg configurations.

        Args:
            args: experiment configuration arguments
            base_size: shorter side of image
            crop_h: integer representing crop height, e.g. 473
            crop_w: integer representing crop width, e.g. 473
            input_file: could be absolute path to .txt file, .mp4 file, or to a directory full of jpg images
            model_taxonomy: taxonomy in which trained model makes predictions
            eval_taxonomy: taxonomy in which trained model is evaluated
            scales: floats representing image scales for multi-scale inference
            use_gpu: TODO, not supporting cpu at this time
        """
        self.args = args

        # Required arguments:
        assert isinstance(self.args.save_folder, str)
        assert isinstance(self.args.dataset, str)
        assert isinstance(self.args.img_name_unique, bool)
        assert isinstance(self.args.print_freq, int)
        assert isinstance(self.args.num_model_classes, int)
        assert isinstance(self.args.model_path, str)
        self.num_model_classes = self.args.num_model_classes

        self.base_size = base_size
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.input_file = input_file
        self.model_taxonomy = model_taxonomy
        self.eval_taxonomy = eval_taxonomy
        self.scales = scales
        self.use_gpu = use_gpu

        self.mean, self.std = normalization_utils.get_imagenet_mean_std()
        self.model = self.load_model(args)
        self.softmax = nn.Softmax(dim=1)

        self.gray_folder = None  # optional, intended for dataloader use
        self.data_list = None  # optional, intended for dataloader use

        if model_taxonomy == "universal" and eval_taxonomy == "universal":
            # See note above.
            # no conversion of predictions required
            self.num_eval_classes = self.num_model_classes

        elif model_taxonomy == "test_dataset" and eval_taxonomy == "test_dataset":
            # no conversion of predictions required
            self.num_eval_classes = len(names_utils.load_class_names(args.dataset))

        elif model_taxonomy == "naive" and eval_taxonomy == "test_dataset":
            self.tc = NaiveTaxonomyConverter()
            if args.dataset in self.tc.convs.keys() and use_gpu:
                self.tc.convs[args.dataset].cuda()
            self.tc.softmax.cuda()
            self.num_eval_classes = len(names_utils.load_class_names(args.dataset))

        elif model_taxonomy == "universal" and eval_taxonomy == "test_dataset":
            # no label conversion required here, only predictions converted
            self.tc = TaxonomyConverter()
            if args.dataset in self.tc.convs.keys() and use_gpu:
                self.tc.convs[args.dataset].cuda()
            self.tc.softmax.cuda()
            self.num_eval_classes = len(names_utils.load_class_names(args.dataset))

        if self.args.arch == "psp":
            assert isinstance(self.args.zoom_factor, int)
            assert isinstance(self.args.network_name, int)

        # `id_to_class_name_map` only used for visualizing universal taxonomy
        self.id_to_class_name_map = {i: classname for i, classname in enumerate(names_utils.get_universal_class_names())}

        # indicate which scales were used to make predictions
        # (multi-scale vs. single-scale)
        self.scales_str = "ms" if len(args.scales) > 1 else "ss"

    def load_model(self, args) -> nn.Module:
        """Load Pytorch pre-trained model from disk of type torch.nn.DataParallel.

        Note that `args.num_model_classes` will be size of logits output.

        Args:
            args:

        Returns:
            model
        """
        if args.arch == "psp":
            model = PSPNet(
                layers=args.layers,
                classes=args.num_model_classes,
                zoom_factor=args.zoom_factor,
                pretrained=False,
                network_name=args.network_name,
            )
        elif args.arch == "hrnet":
            from mseg_semantic.model.seg_hrnet import get_configured_hrnet

            # note apex batchnorm is hardcoded
            model = get_configured_hrnet(args.num_model_classes, load_imagenet_model=False)
        elif args.arch == "hrnet_ocr":
            from mseg_semantic.model.seg_hrnet_ocr import get_configured_hrnet_ocr

            model = get_configured_hrnet_ocr(args.num_model_classes)
        # logger.info(model)
        model = torch.nn.DataParallel(model)
        if self.use_gpu:
            model = model.cuda()
        cudnn.benchmark = True

        if os.path.isfile(args.model_path):
            logger.info(f"=> loading checkpoint '{args.model_path}'")
            if self.use_gpu:
                checkpoint = torch.load(args.model_path)
            else:
                checkpoint = torch.load(args.model_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            logger.info(f"=> loaded checkpoint '{args.model_path}'")
        else:
            raise RuntimeError(f"=> no checkpoint found at '{args.model_path}'")

        return model

    def execute(self) -> None:
        """
        Execute the demo, i.e. feed all of the desired input through the
        network and obtain predictions. Gracefully handles .txt,
        or video file (.mp4, etc), or directory input.
        """
        logger.info(">>>>>>>>>>>>>> Start inference task >>>>>>>>>>>>>")
        self.model.eval()

        if self.input_file is None and self.args.dataset != "default":
            # evaluate on a train or test dataset
            test_loader, self.data_list = create_test_loader(self.args)
            self.execute_on_dataloader(test_loader)
            logger.info("<<<<<<<<< Inference task completed <<<<<<<<<")
            return

        suffix = self.input_file[-4:]
        is_dir = os.path.isdir(self.input_file)
        is_img = suffix in [".png", ".jpg"]
        is_vid = suffix in [".mp4", ".avi", ".mov"]

        if is_img:
            self.render_single_img_pred()
        elif is_dir:
            # argument is a path to a directory
            self.create_path_lists_from_dir()
            test_loader, self.data_list = create_test_loader(self.args)
            self.execute_on_dataloader(test_loader)
        elif is_vid:
            # argument is a video
            self.execute_on_video()
        else:
            logger.info("Error: Unknown input type")

        logger.info("<<<<<<<<<<< Inference task completed <<<<<<<<<<<<<<")

    def render_single_img_pred(self, min_resolution: int = 1080) -> None:
        """Since overlaid class text is difficult to read below 1080p, we upsample predictions."""
        in_fname_stem = Path(self.input_file).stem
        output_gray_fpath = f"{in_fname_stem}_gray.jpg"
        output_demo_fpath = f"{in_fname_stem}_overlaid_classes.jpg"
        logger.info(f"Write image prediction to {output_demo_fpath}")

        rgb_img = imread_rgb(self.input_file)
        pred_label_img = self.execute_on_img(rgb_img)

        # avoid blurry images by upsampling RGB before overlaying text
        if np.amin(rgb_img.shape[:2]) < min_resolution:
            rgb_img = resize_util.resize_img_by_short_side(rgb_img, min_resolution, "rgb")
            pred_label_img = resize_util.resize_img_by_short_side(pred_label_img, min_resolution, "label")

        metadata = None
        frame_visualizer = Visualizer(rgb_img, metadata)
        overlaid_img = frame_visualizer.overlay_instances(
            label_map=pred_label_img, id_to_class_name_map=self.id_to_class_name_map
        )
        imageio.imwrite(output_demo_fpath, overlaid_img)
        imageio.imwrite(output_gray_fpath, pred_label_img)

    def create_path_lists_from_dir(self) -> None:
        """Populate a .txt file with relative paths that will be used to create a Pytorch dataloader."""
        self.args.data_root = self.input_file
        txt_output_dir = str(Path(f"{_ROOT}/temp_files").resolve())
        txt_save_fpath = dump_relpath_txt(self.input_file, txt_output_dir)
        self.args.test_list = txt_save_fpath

    def execute_on_img(self, image: np.ndarray) -> np.ndarray:
        """
        Rather than feeding in crops w/ sliding window across the full-res image, we
        downsample/upsample the image to a default inference size. This may differ
        from the best training size.

        For example, if trained on small images, we must shrink down the image in
        testing (preserving the aspect ratio), based on the parameter "base_size",
        which is the short side of the image.

        Args:
            image: Numpy array representing RGB image

        Returns:
            gray_img: prediction, representing predicted label map
        """
        h, w, _ = image.shape
        is_single_scale = len(self.scales) == 1

        if is_single_scale:
            # single scale, do addition and argmax on CPU
            image_scaled = resize_by_scaled_short_side(image, self.base_size, self.scales[0])
            prediction = torch.Tensor(self.scale_process_cuda(image_scaled, h, w))

        else:
            # multi-scale, prefer to use fast addition on the GPU
            prediction = np.zeros((h, w, self.num_eval_classes), dtype=float)
            prediction = torch.Tensor(prediction).cuda()
            for scale in self.scales:
                image_scaled = resize_by_scaled_short_side(image, self.base_size, scale)
                prediction = prediction + torch.Tensor(self.scale_process_cuda(image_scaled, h, w)).cuda()

        prediction /= len(self.scales)
        prediction = torch.argmax(prediction, axis=2)
        prediction = prediction.data.cpu().numpy()
        gray_img = np.uint8(prediction)
        return gray_img

    def execute_on_video(self, max_num_frames: int = 5000, min_resolution: int = 1080) -> None:
        """
        input_file is a path to a video file.
        Read frames from an RGB video file, and write overlaid predictions into a new video file.
        """
        in_fname_stem = Path(self.input_file).stem
        out_fname = f"{in_fname_stem}_{self.args.model_name}_universal"
        out_fname += f"_scales_{self.scales_str}_base_sz_{self.args.base_size}.mp4"

        output_video_fpath = f"{_ROOT}/temp_files/{out_fname}"
        create_leading_fpath_dirs(output_video_fpath)
        logger.info(f"Write video to {output_video_fpath}")
        writer = VideoWriter(output_video_fpath)

        reader = VideoReader(self.input_file)
        for frame_idx in range(reader.num_frames):
            logger.info(f"On image {frame_idx}/{reader.num_frames}")
            rgb_img = reader.get_frame()
            if frame_idx > max_num_frames:
                break
            pred_label_img = self.execute_on_img(rgb_img)

            # avoid blurry images by upsampling RGB before overlaying text
            if np.amin(rgb_img.shape[:2]) < min_resolution:
                rgb_img = resize_util.resize_img_by_short_side(rgb_img, min_resolution, "rgb")
                pred_label_img = resize_util.resize_img_by_short_side(pred_label_img, min_resolution, "label")

            metadata = None
            frame_visualizer = Visualizer(rgb_img, metadata)
            output_img = frame_visualizer.overlay_instances(
                label_map=pred_label_img, id_to_class_name_map=self.id_to_class_name_map
            )
            writer.add_frame(output_img)

        reader.complete()
        writer.complete()

    def execute_on_dataloader(self, test_loader: torch.utils.data.dataloader.DataLoader) -> None:
        """Run a pretrained model over each batch in a dataloader.

        Args:
             test_loader: dataloader.
        """
        if self.args.save_folder == "default":
            self.args.save_folder = f"{_ROOT}/temp_files/{self.args.model_name}_{self.args.dataset}_universal_{self.scales_str}/{self.args.base_size}"

        os.makedirs(self.args.save_folder, exist_ok=True)
        gray_folder = os.path.join(self.args.save_folder, "gray")
        self.gray_folder = gray_folder

        data_time = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        check_mkdir(self.gray_folder)

        for i, (input, _) in enumerate(test_loader):
            logger.info(f"On image {i}")
            data_time.update(time.time() - end)

            # determine path for grayscale label map
            image_path, _ = self.data_list[i]
            if self.args.img_name_unique:
                image_name = Path(image_path).stem
            else:
                image_name = get_unique_stem_from_last_k_strs(image_path)
            gray_path = os.path.join(self.gray_folder, image_name + ".png")
            if Path(gray_path).exists():
                continue

            # convert Pytorch tensor -> Numpy, then feedforward
            input = np.squeeze(input.numpy(), axis=0)
            image = np.transpose(input, (1, 2, 0))
            gray_img = self.execute_on_img(image)

            batch_time.update(time.time() - end)
            end = time.time()
            cv2.imwrite(gray_path, gray_img)

            # todo: update to time remaining.
            if ((i + 1) % self.args.print_freq == 0) or (i + 1 == len(test_loader)):
                logger.info(
                    "Test: [{}/{}] "
                    "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                        i + 1, len(test_loader), data_time=data_time, batch_time=batch_time
                    )
                )

    def scale_process_cuda(self, image: np.ndarray, raw_h: int, raw_w: int, stride_rate: float = 2 / 3) -> np.ndarray:
        """First, pad the image. If input is (384x512), then we must pad it up to shape
        to have shorter side "scaled base_size".

        Then we perform the sliding window on this scaled image, and then interpolate
        (downsample or upsample) the prediction back to the original one.

        At each pixel, we increment a counter for the number of times this pixel
        has passed through the sliding window.

        Args:
            image: Array, representing image where shortest edge is adjusted to base_size
            raw_h: integer representing native/raw image height on disk, e.g. for NYU it is 480
            raw_w: integer representing native/raw image width on disk, e.g. for NYU it is 640
            stride_rate: stride rate of sliding window operation

        Returns:
            prediction: Numpy array representing predictions with shorter side equal to self.base_size
        """
        resized_h, resized_w, _ = image.shape
        padded_image, pad_h_half, pad_w_half = pad_to_crop_sz(image, self.crop_h, self.crop_w, self.mean)
        new_h, new_w, _ = padded_image.shape
        stride_h = int(np.ceil(self.crop_h * stride_rate))
        stride_w = int(np.ceil(self.crop_w * stride_rate))
        grid_h = int(np.ceil(float(new_h - self.crop_h) / stride_h) + 1)
        grid_w = int(np.ceil(float(new_w - self.crop_w) / stride_w) + 1)

        prediction_crop = torch.zeros((self.num_eval_classes, new_h, new_w)).cuda()
        count_crop = torch.zeros((new_h, new_w)).cuda()

        # loop w/ sliding window, obtain start/end indices
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                # height indices are s_h to e_h (start h index to end h index)
                # width indices are s_w to e_w (start w index to end w index)
                s_h = index_h * stride_h
                e_h = min(s_h + self.crop_h, new_h)
                s_h = e_h - self.crop_h
                s_w = index_w * stride_w
                e_w = min(s_w + self.crop_w, new_w)
                s_w = e_w - self.crop_w
                image_crop = padded_image[s_h:e_h, s_w:e_w].copy()
                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[:, s_h:e_h, s_w:e_w] += self.net_process(image_crop)

        prediction_crop /= count_crop.unsqueeze(0)
        # disregard predictions from padded portion of image
        prediction_crop = prediction_crop[:, pad_h_half : pad_h_half + resized_h, pad_w_half : pad_w_half + resized_w]

        # CHW -> HWC
        prediction_crop = prediction_crop.permute(1, 2, 0)
        prediction_crop = prediction_crop.data.cpu().numpy()

        # upsample or shrink predictions back down to scale=1.0
        prediction = cv2.resize(prediction_crop, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
        return prediction

    def net_process(self, image: np.ndarray, flip: bool = True) -> torch.Tensor:
        """Feed input through the network.

        In addition to running a crop through the network, we can flip
        the crop horizontally, run both crops through the network, and then
        average them appropriately. Afterwards, apply softmax, then convert
        the prediction to the label taxonomy.

        Args:
            image:
            flip: boolean, whether to average with flipped patch output

        Returns:
            output: Pytorch tensor representing network predicting in evaluation taxonomy
                (not necessarily the model taxonomy)
        """
        input = torch.from_numpy(image.transpose((2, 0, 1))).float()
        normalization_utils.normalize_img(input, self.mean, self.std)
        input = input.unsqueeze(0)

        if self.use_gpu:
            input = input.cuda()
        if flip:
            # add another example to batch dimension, that is the flipped crop
            input = torch.cat([input, input.flip(3)], 0)
        with torch.no_grad():
            output = self.model(input)
        _, _, h_i, w_i = input.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode="bilinear", align_corners=True)

        prediction_conversion_req = self.model_taxonomy != self.eval_taxonomy
        if prediction_conversion_req:
            # Either (model_taxonomy='naive', eval_taxonomy='test_dataset')
            # Or (model_taxonomy='universal', eval_taxonomy='test_dataset')
            output = self.tc.transform_predictions_test(output, self.args.dataset)
        else:
            # model & eval tax match, so no conversion needed
            assert self.model_taxonomy in ["universal", "test_dataset"]
            # todo: determine when .cuda() needed here
            output = self.softmax(output)

        if flip:
            # take back out the flipped crop, correct its orientation, and average result
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        # output = output.data.cpu().numpy()
        # convert CHW to HWC order
        # output = output.transpose(1, 2, 0)
        # output = output.permute(1,2,0)

        return output


if __name__ == "__main__":
    pass
