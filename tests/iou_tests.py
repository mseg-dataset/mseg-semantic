#!/usr/bin/python3

import numpy as np
import pdb

from mseg_semantic.utils.iou import intersectionAndUnion, intersectionAndUnionGPU


def test_intersectionAndUnion_2classes():
	"""
	No way to compute union of two sets, without understanding where they intersect.
	"""
	pred = np.array(
		[
			[0,0],
			[1,0]
		])
	target = np.array(
		[
			[0,0],
			[1,1]
		])
	num_classes = 2

	# contain the number of samples in each bin.
	area_intersection, area_union, area_target = intersectionAndUnion(
		pred,
		target,
		K=num_classes,
		ignore_index=255
	)

	assert np.allclose(area_intersection, np.array([2, 1]))
	assert np.allclose(area_target, np.array([2, 2]))
	assert np.allclose(area_union, np.array([3, 2]))


def test_intersectionAndUnion_3classes():
	"""
	No way to compute union of two sets, without understanding where they intersect.
	"""
	pred = np.array(
		[
			[2,0],
			[1,0]
		])
	target = np.array(
		[
			[2,0],
			[1,1]
		])
	num_classes = 3

	# contain the number of samples in each bin.
	area_intersection, area_union, area_target = intersectionAndUnion(
		pred,
		target,
		K=num_classes,
		ignore_index=255
	)

	assert np.allclose(area_intersection, np.array([1,1,1]))
	assert np.allclose(area_target, np.array([1,2,1]))
	assert np.allclose(area_union, np.array([2,2,1]))


def test_mIoU():
	"""
	"""
	intersection = np.array([1,1,1])
	union = np.array([2,1,1])
	miou = np.mean(intersection / (union + 1e-10))


def test_test_intersectionAndUnion_ignore_label():
	"""
	Handle the ignore case. Since 255 lies outside of the histogram bins, it will be ignored.
	"""
	pred = np.array(
		[
			[1,0],
			[1,0]
		])
	target = np.array(
		[
			[255,0],
			[255,1]
		])
	num_classes = 2

	# contain the number of samples in each bin.
	area_intersection, area_union, area_target = intersectionAndUnion(pred, target, K=num_classes, ignore_index=255)

	assert np.allclose(area_intersection, np.array([1, 0]))
	assert np.allclose(area_target, np.array([1, 1]))
	assert np.allclose(area_union, np.array([2, 1]))


