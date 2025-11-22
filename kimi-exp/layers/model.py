
from torch import nn
from .embed_head import VocabParallelEmbedding, ParallelLMHead
from .attention import MLA
from .moe import KimiMoE, MLP
from .norm import RMSNorm


class Config:
    def __init__(self):
        self.vocab_size = 163840
        self.hidden_size = 7168

        # Attention
        self.num_attention_heads = 64 # MQA is disabled

        self.q_lora_rank = 1536
        self.kv_lora_rank = 512

        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = 64
        self.v_head_dim = 128

        # Rope and RMSnorm
        self.max_position_embeddings = 262144
        self.rope_theta = 50000.0
        self.rms_norm_eps = 1e-5
        self.rope_scaling_factor = 64.0

        # Moe
        self.intermediate_size = 18432 # MLP layer
        self.n_routed_experts = 384         
        self.n_shared_experts = 1           
        self.num_experts_per_tok = 8        
        self.moe_intermediate_size = 2048     
        self.moe_layer_freq = 1   
        self.routed_scaling_factor = 2.827         

        # Depth
        self.num_hidden_layers = 61

        # Token ids
        self.bos_token_id = 163584
        self.eos_token_id = 163586
        self.pad_token_id = 163839

        # More Training Params
        self.batch_size = 1
        self.seq_len = 256


config = Config()

class TransformerBlock(nn.Module):

    def __init__(self, layer_idx):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.self_attn = MLA(config)
        if layer_idx == 0:
            self.mlp = MLP(config.hidden_size, config.intermediate_size)
        else:
            self.mlp = KimiMoE(config)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual

        return x


class KimiV2Thinking(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            TransformerBlock(layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size
        )

    def forward(self, input_ids):

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits