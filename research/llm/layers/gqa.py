class GroupedQueryAttentionWithRoPE(GroupedQueryAttention):
    def forward(self, x: Tensor, freqs_cis: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        bsz, _, _ = x.size()

        q = self.wq(x).view(bsz, -1, self.num_q_heads, self.head_dim)
        k = self.wk(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        k = k.repeat_interleave(self.kv_repeats, dim=1)
        v = v.repeat_interleave(self.kv_repeats, dim=1)

        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0

        x = self.sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)
        x = self.out_proj(x)
        return x


class GroupedQueryAttentionWithRoPEAndCache(GroupedQueryAttentionWithRoPE):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        kv_cache: KVCache | None = None,
        dropout_p: float = 0,
        bias: bool = False,
        sdpa: Callable = F.scaled_dot_product_attention,
        device=None,
        dtype=None,
    ):
        super().__init__(
            embed_dim,
            num_q_heads,
            num_kv_heads,
            dropout_p,
            bias,
            sdpa,
            device,
            dtype,
        )

        self.kv_cache = kv_cache
        head_dim = embed_dim // num_q_heads
        self.head_dim = head_dim

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        bsz, _, _ = x.size()

        q = self.wq(x).view(bsz, -1, self.num_q_heads, self.head_dim)
        k = self.wk(x).view(bsz, -1, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, -1, self.num_kv_heads, self.head_dim)

        # rope
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # (b, s, h, d) -> (b, h, s, d)
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        # cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # gqa
        k = k.repeat_interleave(self.kv_repeats, dim=1)
        v = v.repeat_interleave(self.kv_repeats, dim=1)

        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0

        x = self.sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)
        x = self.out_proj(x)
        return x


Attention = GroupedQueryAttentionWithRoPEAndCache
