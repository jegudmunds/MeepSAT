import json

import numpy as np
import pytest

from meepsat.helpers import (
    extract_ticks,
    filter_dict,
    linear_flair_center,
    read_json,
    rot_x,
    rot_y,
    rot_z,
    rotation_matrix,
)


def test_filter_dict_keeps_only_function_parameters():
    def target(required, optional=None):
        return required, optional

    values = {"required": 1, "optional": 2, "ignored": 3}

    assert filter_dict(values, target) == {"required": 1, "optional": 2}


def test_filter_dict_rejects_non_callable():
    with pytest.raises(TypeError, match="must be callable"):
        filter_dict({"value": 1}, None)


def test_extract_ticks_uses_simulation_bounds():
    ticks = extract_ticks(data=None, num_ticks=3, sim_box=[(-1, 1), (0, 4)])

    np.testing.assert_allclose(ticks[0], [-1, 0, 1])
    np.testing.assert_allclose(ticks[1], [0, 2, 4])
    assert ticks[2] == ["-1.0", "0.0", "1.0"]
    assert ticks[3] == ["0.0", "2.0", "4.0"]


@pytest.mark.parametrize("rotation", [rot_x, rot_y, rot_z])
def test_axis_rotations_are_orthogonal(rotation):
    matrix = rotation(0.37)

    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-15)


def test_rotation_matrix_uses_zemax_order():
    tx, ty, tz = 0.1, 0.2, 0.3

    expected = rot_z(tz) @ rot_y(ty) @ rot_x(tx)
    np.testing.assert_allclose(rotation_matrix(tx, ty, tz), expected)


def test_read_json(tmp_path):
    path = tmp_path / "input.json"
    expected = {"frequency": 150e9, "components": ["lens", "stop"]}
    path.write_text(json.dumps(expected), encoding="utf-8")

    assert read_json(path) == expected


@pytest.mark.parametrize(
    ("angle_degrees", "expected"),
    [
        (60, (2.4, 2.5 + 2 * np.sin(np.radians(60)) - 0.1)),
        (120, (0.6, 2.5 + 2 * np.sin(np.radians(120)) + 0.1)),
    ],
)
def test_linear_flair_center_across_90_degrees(angle_degrees, expected):
    center = linear_flair_center(
        vertex_x=1.5,
        vertex_y=3.0,
        length=4.0,
        thickness=1.0,
        angle_degrees=angle_degrees,
        angle_axis="x",
        unit_pixel_length=0.1,
    )

    np.testing.assert_allclose(center, expected)


def test_linear_flair_center_does_not_modify_vertex_coordinates():
    vertex = [1.5, 3.0]

    linear_flair_center(
        vertex_x=vertex[0],
        vertex_y=vertex[1],
        length=4.0,
        thickness=1.0,
        angle_degrees=60,
        angle_axis="y",
        unit_pixel_length=0.1,
    )

    assert vertex == [1.5, 3.0]


@pytest.mark.parametrize("angle_axis", ["z", "invalid"])
def test_linear_flair_center_rejects_invalid_axis(angle_axis):
    with pytest.raises(ValueError, match="Invalid rotation axis"):
        linear_flair_center(0, 0, 1, 1, 45, angle_axis, 0.1)
