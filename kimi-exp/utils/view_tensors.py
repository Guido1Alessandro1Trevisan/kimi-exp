from safetensors import safe_open
import os
from tabulate import tabulate

from safetensors import safe_open

def compute_compressed_params(f, base):
    # use get_tensor instead of f[...] !!
    wshape = f.get_tensor(base + "weight_shape")
    M, N = wshape.tolist()
    return M * N


def analyze_attention_experts(full_path):
    attn_params = 0
    expert_params = 0
    other_params = 0
    rows = []

    with safe_open(full_path, framework="pt") as f:
        keys = list(f.keys())

        for k in keys:

            # -----------------------------
            # Case 1 — INT4 compressed weights
            # -----------------------------
            if k.endswith("weight_packed"):
                base = k[:-len("weight_packed")]

                params = compute_compressed_params(f, base)

                if ".self_attn." in base:
                    group = "self_attn"
                    attn_params += params
                elif ".mlp.experts." in base:
                    group = "experts"
                    expert_params += params
                else:
                    group = "other"
                    other_params += params

                rows.append([base, group, params])
                continue

            # -----------------------------
            # Case 2 — FP16/BF16/FP32 raw weights
            # -----------------------------
            if k.endswith("weight") and not (
                "weight_scale" in k or "weight_shape" in k
            ):
                t = f.get_tensor(k)
                params = t.numel()

                if ".self_attn." in k:
                    group = "self_attn"
                    attn_params += params
                elif ".mlp.experts." in k:
                    group = "experts"
                    expert_params += params
                else:
                    group = "other"
                    other_params += params

                rows.append([k, group, params])
                continue

    return rows, attn_params, expert_params, other_params



# ====================
# RUN IT
# ====================
path = "/workspace/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"
file = "model-00062-of-000062.safetensors"
full_path = os.path.join(path, file)

rows, attn, experts, other = analyze_attention_experts(full_path)

print(tabulate(rows, headers=["Tensor", "Group", "Params"], tablefmt="github"))

print("\nSUMMARY")
print("Self-attention params:", attn)
print("Expert params:       ", experts)
print("Other params:        ", other)
print("TOTAL params:        ", attn + experts + other)
