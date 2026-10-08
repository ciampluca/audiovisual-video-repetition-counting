from pathlib import Path

import hydra
from omegaconf import DictConfig

from lvlm_video_repetition_benchmark.training import train_qwen3_vl_lora


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="train_lora",
)
def main(cfg: DictConfig) -> None:
    train_qwen3_vl_lora(cfg)


if __name__ == "__main__":
    main()