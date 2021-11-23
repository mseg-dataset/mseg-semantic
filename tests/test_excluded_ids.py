#!/usr/bin/python3

"""Unit tests to ensure that exclusion of certain classes from evaluation works properly."""

import mseg.utils.names_utils as names_utils

from mseg_semantic.tool.test_universal_tax import get_excluded_class_ids


def test_get_excluded_class_ids_coco() -> None:
    """
    Ensure we can find the classes to exclude when evaluating
    a "relabeled" MSeg model on the val split of a training dataset.
    """
    dataset = "coco-panoptic-133"
    zero_class_ids = get_excluded_class_ids(dataset)
    # fmt: off
    gt_zero_class_ids = [
        5, 16, 18, 21, 22, 25, 26, 27, 29, 30, 31, 33, 44, 58, 59, 60, 61, 63, 65,
        68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 82, 83, 84, 86, 89, 91, 92, 93,
        101, 108, 112, 113, 114, 122, 123, 124, 126, 127, 128, 130, 131, 132, 133,
        134, 140, 141, 143, 145, 146, 160, 161, 162, 163, 164, 165, 167, 170, 177,
        183, 185, 189, 190,
    ]
    # fmt: on

    # None of these classes are present in COCO-Panoptic
    gt_excluded_classnames = [
        "case",
        "animal_other",
        "radiator",
        "storage_tank",
        "conveyor_belt",
        "washer_dryer",
        "fan",
        "dishwasher",
        "bathtub",
        "shower",
        "tunnel",
        "pier_wharf",
        "stage",
        "armchair",
        "swivel_chair",
        "stool",
        "seat",
        "trash_can",
        "nightstand",
        "pool_table",
        "barrel",
        "desk",
        "ottoman",
        "wardrobe",
        "crib",
        "basket",
        "chest_of_drawers",
        "bookshelf",
        "bathroom_counter",
        "kitchen_island",
        "lamp",
        "sconce",
        "chandelier",
        "whiteboard",
        "escalator",
        "fireplace",
        "stove",
        "arcade_machine",
        "runway",
        "plaything_other",
        "painting",
        "poster",
        "bulletin_board",
        "tray",
        "range_hood",
        "plate",
        "rider_other",
        "bicyclist",
        "motorcyclist",
        "streetlight",
        "road_barrier",
        "mailbox",
        "cctv_camera",
        "junction_box",
        "bike_rack",
        "billboard",
        "pole",
        "railing_banister",
        "guard_rail",
        "base",
        "sculpture",
        "column",
        "fountain",
        "awning",
        "apparel",
        "flag",
        "shower_curtain",
        "autorickshaw",
        "trailer",
        "slow_wheeled_object",
        "swimming_pool",
        "waterfall",
    ]
    u_classnames = names_utils.get_universal_class_names()
    gt_excluded_ids = [u_classnames.index(e_name) for e_name in gt_excluded_classnames]

    assert gt_zero_class_ids == gt_excluded_ids
    assert zero_class_ids == gt_zero_class_ids


# def test_get_excluded_class_ids_bdd():
# 	"""
# 	Ensure we can find the classes to exclude when evaluating
# 	a "relabeled" MSeg model on the val split of a training dataset.
# 	"""
# 	dataset = 'bdd'
# 	zero_class_ids = get_excluded_class_ids(dataset)
# 	pdb.set_trace()


if __name__ == "__main__":
    test_get_excluded_class_ids_coco()
    # test_get_excluded_class_ids_bdd()
