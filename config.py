from __future__ import annotations

from pathlib import Path

from yaml import safe_load

_DEFAULT_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load and return the YAML config, with computed defaults filled in."""
    path = Path(path) if path else _DEFAULT_PATH
    with open(path) as f:
        cfg = safe_load(f)

    env = cfg.setdefault("environment", {})
    env.setdefault("grid_num", 10)
    env.setdefault("frame_rate", 30)
    env.setdefault("frame_count_divisor", 15)

    ro = env.setdefault("render_options", {})
    ro.setdefault("unit_size", 50)
    ro.setdefault("render_unit_size", 45)
    ro.setdefault("block_stroke", False)
    ro.setdefault("unit_stroke_weight", 0.25)

    return cfg
