from safetensors import safe_open
from glob import glob
from collections import defaultdict
import os, re


def canonical_name(key: str) -> str:
    # your function unchanged
    key = re.sub(r"\.(weight|weight_packed|weight_scale|weight_shape|bias)$", "", key)
    key = re.sub(r"experts\.\d+", "experts.*", key)
    key = re.sub(r"layers\.\d+", "layers.*", key)
    return key


def extract_unique_param_families_with_shapes(folder):
    families = defaultdict(set)

    files = sorted(glob(os.path.join(folder, "*.safetensors")))

    for fp in files:
        # be explicit: keep everything on CPU
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                canon = canonical_name(k)

                # ⭐ This reads shape from the header, no full tensor load
                s = f.get_slice(k)
                shape = tuple(s.get_shape())

                families[canon].add(shape)

    return families
