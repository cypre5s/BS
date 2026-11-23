from dataclasses import replace

import pytest

from reacquisition.crop_mapper import CropTransform, intersection_over_union


@pytest.mark.parametrize(
    "size,box,padding,scale",
    [
        ((640, 480), (100, 80, 300, 240), 4, 1.75),
        ((640, 480), (0, 0, 130, 90), 12, 2.0),
        ((640, 480), (520, 400, 640, 480), 20, 1.3),
        ((901, 307), (10.25, 19.75, 811.3, 299.2), 5, 1.6),
        ((80, 400), (-5, 20, 83, 380), 9, 2.75),
        ((1000, 91), (700, 30, 710, 60), 0, 3.0),
    ],
)
def test_crop_coordinates_round_trip(size, box, padding, scale):
    transform = CropTransform.create(size, box, padding=padding, scale=scale)
    left, top, right, bottom = transform.crop_box
    for point in ((left, top), (right, top), (left, bottom), (right, bottom),
                  ((left + right) / 2, (top + bottom) / 2)):
        assert transform.to_global(transform.to_local(point)) == pytest.approx(point)
    assert transform.box_to_global(transform.box_to_local(box)) == pytest.approx(box)
    assert 0 <= left < right <= size[0]
    assert 0 <= top < bottom <= size[1]


def test_coordinates_use_actual_processor_size_after_rounding():
    transform = CropTransform.create((800, 600), (21, 45, 601, 212), scale=1.75)
    transform = replace(transform, resized_size=(1024, 320))
    assert transform.to_local(transform.crop_box[:2]) == (0, 0)
    assert transform.to_global((1024, 320)) == transform.crop_box[2:]


@pytest.mark.parametrize("box", [(0, 0, 0, 10), (5, 5, 4, 8), (200, 200, 300, 300)])
def test_invalid_or_disjoint_crop_rejected(box):
    with pytest.raises(ValueError):
        CropTransform.create((100, 100), box)


def test_overlap_is_in_original_coordinates():
    assert intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert intersection_over_union((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    assert intersection_over_union((2, 3, 12, 13), (2, 3, 12, 13)) == 1
