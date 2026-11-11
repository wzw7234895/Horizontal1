import numpy as np
import pytest
import torch

from pixel_distance.roma_probe import _sample


def test_dense_warp_sampling_uses_original_pixel_coordinates():
    height, width = 40, 60
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    normalized = torch.stack(((2 * xs + 1) / width - 1,
                              (2 * ys + 1) / height - 1), dim=-1).float()
    warp = torch.cat((normalized, normalized), dim=-1)[None]
    certainty = torch.ones((1, height, width))
    mapped, confidence = _sample(warp, certainty, np.array([24.5, 15.5]),
                                 (height, width), (height, width))
    np.testing.assert_allclose(mapped, [24.5, 15.5], atol=1e-4)
    assert confidence == pytest.approx(1)
