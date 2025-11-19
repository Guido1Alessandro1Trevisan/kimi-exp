import re
from safetensors import safe_open
import os
from glob import glob


def canonical_name(key: str) -> str:
    """
    Normalize tensor names while preserving real architectural structure.
    """

    # Remove low-level suffixes
    key = re.sub(r"\.(weight|weight_packed|weight_scale|weight_shape|bias)$", "", key)

    # Collapse expert index → experts.*
    key = re.sub(r"experts\.\d+", "experts.*", key)

    # Collapse layer index → layers.*
    key = re.sub(r"layers\.\d+", "layers.*", key)

    return key


def extract_unique_param_families(folder):
    unique = set()

    # Load ALL shards (1–62)
    files = sorted(glob(os.path.join(folder, "*.safetensors")))

    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                name = canonical_name(k)
                unique.add(name)

    return sorted(unique)


# ====================
# RUN IT
# ====================

folder = "/workspace/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"

unique_names = extract_unique_param_families(folder)

print("\n=== UNIQUE PARAMETER FAMILIES ===")
for u in unique_names:
    print(u)

print("\nTOTAL UNIQUE PARAM GROUPS:", len(unique_names))
