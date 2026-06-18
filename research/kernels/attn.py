import torch
import triton


@triton.jit
def _attn_fwd(
    Q,  # pointer to the first element of the tensor
    K,
    V,
):
    return


class TritonMultiHeadAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        BATCH_SIZE, NUM_HEADS, _SEQ_LEN, _HEAD_DIM = Q.shape

        torch.empty_like(Q)
        lambda args: (
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
