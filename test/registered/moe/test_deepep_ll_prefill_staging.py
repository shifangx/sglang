import pytest

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    compute_deepep_ll_prefill_staging_slices,
)


@pytest.mark.parametrize(
    ("local_num_tokens", "max_num_tokens", "capacity", "expected"),
    [
        (0, 0, 64, []),
        (63, 63, 64, [slice(0, 63)]),
        (130, 130, 64, [slice(0, 64), slice(64, 128), slice(128, 130)]),
        (17, 130, 64, [slice(0, 17), slice(17, 17), slice(17, 17)]),
    ],
)
def test_compute_deepep_ll_prefill_staging_slices(
    local_num_tokens: int,
    max_num_tokens: int,
    capacity: int,
    expected: list[slice],
):
    assert (
        compute_deepep_ll_prefill_staging_slices(
            local_num_tokens=local_num_tokens,
            max_num_tokens=max_num_tokens,
            capacity=capacity,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("local_num_tokens", "max_num_tokens", "capacity"),
    [(1, 1, 0), (-1, 1, 64), (2, 1, 64)],
)
def test_compute_deepep_ll_prefill_staging_slices_rejects_invalid_inputs(
    local_num_tokens: int,
    max_num_tokens: int,
    capacity: int,
):
    with pytest.raises(ValueError):
        compute_deepep_ll_prefill_staging_slices(
            local_num_tokens=local_num_tokens,
            max_num_tokens=max_num_tokens,
            capacity=capacity,
        )
