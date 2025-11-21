import re
from safetensors import safe_open
import os
from glob import glob
from collections import defaultdict


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


def extract_unique_param_families_with_shapes(folder):
    families = defaultdict(set)

    # Load ALL shards
    files = sorted(glob(os.path.join(folder, "*.safetensors")))

    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                canon = canonical_name(k)
                shape = tuple(f.get_tensor(k).shape)
                families[canon].add(shape)

    return families


# ====================
# RUN IT
# ====================

folder = "/workspace/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"

families = extract_unique_param_families_with_shapes(folder)

print("\n=== UNIQUE PARAMETER FAMILIES + SHAPES ===")

for name in sorted(families.keys()):
    print(f"\n{name}")
    for shape in sorted(families[name]):
        print(f"    shape = {shape}")

print("\nTOTAL UNIQUE PARAM GROUPS:", len(families))
