from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from lvlm_video_repetition_benchmark.runner import run_benchmark


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="benchmark",
)
def main(cfg: DictConfig) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run_benchmark(cfg)