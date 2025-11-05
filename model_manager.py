from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from common.config import load_config
from projects.video_diffusion_sr.infer import VideoDiffusionInfer


@dataclass(frozen=True)
class ModelSpec:
    weight: str
    config: str


class RunnerManager:
    def __init__(
        self,
        *,
        device: torch.device,
        checkpoint_dir: Path,
        specs: Dict[str, ModelSpec],
        default_variant: str,
    ) -> None:
        self._device = device
        self._checkpoint_dir = checkpoint_dir
        self._specs = specs

        self._runners: Dict[str, VideoDiffusionInfer] = {}
        self._configs: Dict[str, DictConfig] = {}
        self._devices: Dict[str, str] = {}
        self._active: Optional[str] = None

        self.use(default_variant)

    def use(self, variant: str) -> Tuple[VideoDiffusionInfer, DictConfig]:
        if variant not in self._specs:
            raise ValueError(f"Unknown model variant '{variant}'. Choose from {list(self._specs.keys())}.")

        if variant == self._active and self._devices.get(variant) == "cuda":
            return self._runners[variant], self._configs[variant]

        if self._active and self._active != variant:
            self._move_to_cpu(self._active)
            torch.cuda.empty_cache()

        runner = self._ensure_on_cuda(variant)
        self._active = variant
        return runner, self._configs[variant]

    def _ensure_on_cuda(self, variant: str) -> VideoDiffusionInfer:
        runner = self._runners.get(variant)
        if runner is None:
            runner = self._build_runner(variant)
        if self._devices.get(variant) == "cuda":
            return runner

        runner.dit.to(self._device)
        runner.vae.to(self._device)
        runner.device = "cuda"
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
        self._devices[variant] = "cuda"
        return runner

    def _move_to_cpu(self, variant: str) -> None:
        runner = self._runners.get(variant)
        if runner is None or self._devices.get(variant) == "cpu":
            return
        runner.dit.to("cpu")
        runner.vae.to("cpu")
        runner.device = "cpu"
        self._devices[variant] = "cpu"

    def _build_runner(self, variant: str) -> VideoDiffusionInfer:
        spec = self._specs[variant]
        weight_path = self._resolve_weight(spec.weight)
        config = load_config(spec.config)
        OmegaConf.set_readonly(config, False)

        dit_model = config.dit.model
        dit_model.norm = "rms"
        dit_model.vid_out_norm = "rms"
        if hasattr(dit_model, "txt_in_norm"):
            dit_model.txt_in_norm = "layer"
        if hasattr(dit_model, "qk_norm"):
            dit_model.qk_norm = "rms"

        runner = VideoDiffusionInfer(config)
        runner.configure_dit_model(device="cuda", checkpoint=str(weight_path))
        runner.configure_vae_model()
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

        self._runners[variant] = runner
        self._configs[variant] = config
        self._devices[variant] = "cuda"
        return runner

    def _resolve_weight(self, filename: str) -> Path:
        path = self._checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected weight {path} missing. Did the CDN download succeed?")
        return path
