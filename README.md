

hf download moonshotai/Kimi-K2-Thinking

Install model

Safetensor 0
Self-attention params: 101124096
Expert params:        0
Other params:         396376064
TOTAL params:         497500160

Safetensor 1-61
Self-attention params: 101124096
Expert params:        16911433728
Other params:         46807040
TOTAL params:         17059364864

Safetensor 62
Self-attention params: 0
Expert params:        0
Other params:         2348817408
TOTAL params:         2348817408



LAYERS:

lm_head
model.embed_tokens
model.layers.*.input_layernorm
model.layers.*.mlp.down_proj
model.layers.*.mlp.experts.*.down_proj
model.layers.*.mlp.experts.*.gate_proj
model.layers.*.mlp.experts.*.up_proj
model.layers.*.mlp.gate
model.layers.*.mlp.gate.e_score_correction_bias
model.layers.*.mlp.gate_proj
model.layers.*.mlp.shared_experts.down_proj
model.layers.*.mlp.shared_experts.gate_proj
model.layers.*.mlp.shared_experts.up_proj
model.layers.*.mlp.up_proj
model.layers.*.post_attention_layernorm
model.layers.*.self_attn.kv_a_layernorm
model.layers.*.self_attn.kv_a_proj_with_mqa
model.layers.*.self_attn.kv_b_proj
model.layers.*.self_attn.o_proj
model.layers.*.self_attn.q_a_layernorm
model.layers.*.self_attn.q_a_proj
model.layers.*.self_attn.q_b_proj
model.layers.*.self_attn.rotary_emb.inv_freq
model.norm