class MinimalDeepseekMLA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.q_head_dim = self.qk_nope + self.qk_rope
        self.v_head_dim = config.v_head_dim

        # === Q PATH (A→Norm→B) ===
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank,
                                  self.num_heads * self.q_head_dim,
                                  bias=False)

        # === KV PATH (A→Norm→B) ===
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + self.qk_rope,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)

        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_nope + self.v_head_dim),
            bias=False
        )

        # === Output ===
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim,
                                config.hidden_size,
                                bias=config.attention_bias)

        # === Rotary ===
        self.rotary_emb = DeepseekV3RotaryEmbedding(self.qk_rope)

        self.softmax_scale = self.q_head_dim ** -0.5

    def forward(self, hidden_states, position_ids, past_kv=None):
        bsz, seqlen, _ = hidden_states.shape

        # ----- Q -----
        q_a = self.q_a_proj(hidden_states)
        q_a = self.q_a_layernorm(q_a)
        q = self.q_b_proj(q_a)
        q = q.view(bsz, seqlen, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.qk_nope, self.qk_rope], dim=-1)

        # ----- KV -----
        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, k_pe = kv_a.split([kv_a.shape[-1] - self.qk_rope, self.qk_rope], dim=-1)
        k_pe = k_pe.view(bsz, seqlen, 1, self.qk_rope).transpose(1, 2)

        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)
        kv = kv.view(bsz, seqlen, self.num_heads, self.qk_nope + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope, self.v_head_dim], dim=-1)

        # ----- Rotary -----
        cos, sin = self.rotary_emb(v, seq_len=seqlen)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # Reconstruct full Q/K
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # ----- Attention -----
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(context)
