import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file

from rvc.configs.config import Config
from rvc.synthesizer.audio import load_audio
from rvc.synthesizer.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc.vc.pipeline import Pipeline
from rvc.vc.utils import load_hubert


class VC:
    def __init__(self):
        self.tgt_sr: int | None = None
        self.net_g = None
        self.pipeline: Pipeline | None = None
        self.if_f0: int | None = None
        self.version: str | None = None
        self.hubert_model: Any = None

        self.config = Config()

    def get_vc(
        self, rvc_path: str, rvc_cfg_path: str, hubert_path: str, hubert_cfg_path: str
    ):
        if not os.path.exists(hubert_path):
            raise FileNotFoundError("hubert_path not found.")

        rvc_state = load_file(rvc_path)
        with open(rvc_cfg_path, "r") as f:
            self.rvc_model_config = json.load(f)
        self.tgt_sr = self.rvc_model_config[-1]
        self.rvc_model_config = list(self.rvc_model_config)
        self.rvc_model_config[-3] = rvc_state["emb_g.weight"].shape[0]
        self.if_f0 = 1
        self.version = "v1"

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }
        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.rvc_model_config, is_half=self.config.is_half)

        self.net_g.load_state_dict(rvc_state, strict=False)
        self.net_g.eval().to(self.config.device)
        self.net_g = self.net_g.half() if self.config.is_half else self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)

        self.hubert_model = load_hubert(self.config, hubert_path, hubert_cfg_path)

    def vc_inference(
        self,
        sid: int,
        input_audio_path: str,
        f0_up_key: int = 0,
        f0_method: str = "pm",
        f0_file: str | None = None,
        index_file: str | None = None,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ):
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError("input_audio_path not found.")

        if isinstance(f0_file, str):
            f0_file = Path(f0_file)
        elif not isinstance(f0_file, Path) and f0_file is not None:
            raise RuntimeError(
                f"pathlib.Path, str, or None expected for f0_file. Got {type(f0_file)}"
            )

        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        audio_opt = self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            sid,
            audio,
            f0_up_key,
            f0_method,
            index_file,
            index_rate,
            self.if_f0,
            filter_radius,
            self.tgt_sr,
            resample_sr,
            rms_mix_rate,
            self.version,
            protect,
            f0_file,
        )

        tgt_sr = resample_sr if self.tgt_sr != resample_sr >= 16000 else self.tgt_sr

        return tgt_sr, audio_opt
