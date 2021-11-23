from mseg_semantic.tool.universal_demo_batched import determine_max_possible_base_size


def test_determine_max_possible_base_size():
    """ """
    native_img_height = 1200
    native_img_width = 1920
    base_size = determine_max_possible_base_size(h=native_img_height, w=native_img_width, crop_sz=473)
    assert base_size == 295
