import pytest
import torch

from sglang.jit_kernel.glm5_router_gemm import glm5_router_gemm


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16])
def test_glm5_router_gemm_accuracy(num_tokens: int):
    torch.manual_seed(42)
    hidden_states = torch.randn((num_tokens, 6144), dtype=torch.bfloat16, device="cuda")
    router_weight = torch.randn((256, 6144), dtype=torch.bfloat16, device="cuda")
    actual = glm5_router_gemm(hidden_states, router_weight)
    expected = torch.mm(hidden_states, router_weight.t(), out_dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)


def test_glm5_router_gemm_batch_invariance():
    torch.manual_seed(42)
    token = torch.randn((1, 6144), dtype=torch.bfloat16, device="cuda")
    router_weight = torch.randn((256, 6144), dtype=torch.bfloat16, device="cuda")
    expected = glm5_router_gemm(token, router_weight)

    for num_tokens, token_index in [(2, 1), (8, 5), (16, 15)]:
        hidden_states = torch.randn(
            (num_tokens, 6144), dtype=torch.bfloat16, device="cuda"
        )
        hidden_states[token_index].copy_(token[0])
        actual = glm5_router_gemm(hidden_states, router_weight)[
            token_index : token_index + 1
        ]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
