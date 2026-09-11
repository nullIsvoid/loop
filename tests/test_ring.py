import torch

from latent_loop import (
    RingLatentProcessor,
    RingMixConfig,
    circular_indices,
    circular_pad,
    circular_temporal_mix,
    cosine_denoise_strength,
    ring_windows,
)


def test_circular_indices_wrap_both_sides():
    idx = circular_indices(6, [0, 5], [-1, 0, 1])
    assert idx.tolist() == [[5, 0, 1], [4, 5, 0]]


def test_circular_pad_does_not_duplicate_endpoint_as_new_video_frame():
    x = torch.arange(6).reshape(1, 1, 6, 1, 1)
    padded = circular_pad(x, 1)
    assert padded.flatten().tolist() == [5, 0, 1, 2, 3, 4, 5, 0]
    assert padded.shape[2] == 8


def test_ring_windows_first_and_last_are_neighbours():
    x = torch.arange(6).reshape(1, 1, 6, 1, 1)
    windows = ring_windows(x, 1)
    assert windows.shape == (1, 1, 6, 3, 1, 1)
    assert windows[0, 0, 0, :, 0, 0].tolist() == [5, 0, 1]
    assert windows[0, 0, 5, :, 0, 0].tolist() == [4, 5, 0]


def test_mix_is_ring_symmetric():
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1, 1)
    y = circular_temporal_mix(x, radius=1, strength=1.0)
    # radius=1 uses the two neighbours equally.
    expected = torch.tensor([3.0, 1.0, 2.0, 3.0, 4.0, 2.0]).reshape(1, 1, 6, 1, 1)
    assert torch.allclose(y, expected)


def test_strength_fades_to_zero():
    strengths = [cosine_denoise_strength(i, 10, maximum=0.2, active_fraction=0.7) for i in range(10)]
    assert strengths[0] == 0.2
    assert strengths[6] == 0.0
    assert strengths[7:] == [0.0, 0.0, 0.0]
    assert all(a >= b for a, b in zip(strengths, strengths[1:]))


def test_processor_leaves_final_steps_untouched():
    x = torch.randn(1, 4, 6, 8, 8)
    processor = RingLatentProcessor(RingMixConfig(active_fraction=0.5))
    y = processor(x, step=9, total_steps=10)
    assert y.data_ptr() == x.data_ptr()


def test_negative_temporal_dim_is_supported():
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1, 1)
    windows = ring_windows(x, 1, temporal_dim=-3)
    assert windows.shape == (1, 1, 6, 3, 1, 1)
    y = circular_temporal_mix(x, radius=1, strength=1.0, temporal_dim=-3)
    assert y.shape == x.shape
    assert y.flatten().tolist() == [3.0, 1.0, 2.0, 3.0, 4.0, 2.0]
