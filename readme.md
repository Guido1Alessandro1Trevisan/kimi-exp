
=== UNIQUE PARAMETER FAMILIES + SHAPES ===

lm_head
    shape = (163840, 7168)

model.embed_tokens
    shape = (163840, 7168)

model.layers.*.input_layernorm
    shape = (7168,)

model.layers.*.mlp.down_proj
    shape = (7168, 18432)

model.layers.*.mlp.experts.*.down_proj
    shape = (2,)
    shape = (7168, 64)
    shape = (7168, 256)

model.layers.*.mlp.experts.*.gate_proj
    shape = (2,)
    shape = (2048, 224)
    shape = (2048, 896)

model.layers.*.mlp.experts.*.up_proj
    shape = (2,)
    shape = (2048, 224)
    shape = (2048, 896)

model.layers.*.mlp.gate
    shape = (384, 7168)

model.layers.*.mlp.gate.e_score_correction_bias
    shape = (384,)

model.layers.*.mlp.gate_proj
    shape = (18432, 7168)

model.layers.*.mlp.shared_experts.down_proj
    shape = (7168, 2048)

model.layers.*.mlp.shared_experts.gate_proj
    shape = (2048, 7168)

model.layers.*.mlp.shared_experts.up_proj
    shape = (2048, 7168)

model.layers.*.mlp.up_proj
    shape = (18432, 7168)

model.layers.*.post_attention_layernorm
    shape = (7168,)

model.layers.*.self_attn.kv_a_layernorm
    shape = (512,)

model.layers.*.self_attn.kv_a_proj_with_mqa
    shape = (576, 7168)

model.layers.*.self_attn.kv_b_proj
    shape = (16384, 512)

model.layers.*.self_attn.o_proj
    shape = (7168, 8192)

model.layers.*.self_attn.q_a_layernorm
    shape = (1536,)

model.layers.*.self_attn.q_a_proj
    shape = (1536, 7168)

model.layers.*.self_attn.q_b_proj
    shape = (12288, 1536)

model.layers.*.self_attn.rotary_emb.inv_freq
    shape = (128,)

model.norm
    shape = (7168,)