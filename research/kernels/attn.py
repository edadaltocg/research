import triton
import triton.language as tl
import torch


@triton.jit
def _attn_fwd(
    Q,  # pointer to the first element of the tensor
    K,
    V,
):
    index_batch = ...
    index_head = ...
    qkv_offset = ...
    return


class TritonMultiHeadAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        O = torch.empty_like(Q)
        grid = lambda args: (
            triton.cdiv(),
            BATCH_SIZE * NUM_HEADS,
            1,  # z in the CUDA launch grid
        )
        _attn_fwd(
            Q=Q,
            K=K,
            V=V,
        )
        ctx.causal = causal
        return
