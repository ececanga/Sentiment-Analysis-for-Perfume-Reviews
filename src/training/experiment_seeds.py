"""Keep training randomness independent of data, bootstrap and permutation seeds."""
from typing import Any, Mapping


def data_seed(config: Mapping[str, Any]) -> int:
    return int(config.get("seeds", {}).get("data_seed", 42))


def model_seed(config: Mapping[str, Any]) -> int:
    seeds = config.get("seeds", {})
    # Weighted historical protocols used model.random_seed; benchmarks used data_seed.
    return int(seeds.get("model_seed", seeds.get("data_seed", config.get("model", {}).get("random_seed", 42))))
