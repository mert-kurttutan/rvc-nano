import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from dotenv import load_dotenv


def _get_model_path() -> str:
    model_path = os.getenv("RVC_TEST_MODEL") or os.getenv("RVC_MODEL_PATH")
    if not model_path:
        pytest.skip("Set RVC_TEST_MODEL (or RVC_MODEL_PATH) to a .pth model file.")
    return str(model_path)


def _get_input_path(tmp_path: Path) -> Path:
    input_path = os.getenv("RVC_TEST_INPUT")
    if input_path:
        return Path(input_path)

    # Generate a short mono sine wave for a minimal smoke test.
    sr = 16000
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    wav_path = tmp_path / "input.wav"
    sf.write(wav_path, audio, sr)
    return wav_path


def test_vc_inference_shapes(tmp_path: Path) -> None:
    load_dotenv()

    from rvc.vc.modules import VC

    # model_path = _get_model_path()
    input_path = _get_input_path(tmp_path)
    print(f"input_path: {input_path}")

    # Provide hubert_path if assets exist; otherwise skip with a clear message.
    hubert_path = "assets/hubert_base.pt"

    # Provide rmvpe_root for f0 extraction.
    rmvpe_root = Path("assets/rmvpe")
    if rmvpe_root.exists():
        os.environ.setdefault("rmvpe_root", str(rmvpe_root))
    else:
        pytest.skip("rmvpe_root not found under assets/. Set rmvpe_root env.")

    vc = VC()
    rvc_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/rvc_model.safetensors"
    hubert_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/hubert_model.safetensors"

    hubert_cfg_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/rvc/configs/hubert_cfg.json"
    rvc_cfg_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/rvc/configs/rvc_model_config.json"
    vc.get_vc(rvc_path, rvc_cfg_path, hubert_path, hubert_cfg_path)
    tgt_sr, audio_opt = vc.vc_inference(0, str(input_path))

    assert tgt_sr > 0
    assert audio_opt is not None
    assert audio_opt.ndim in (1, 2)
    assert audio_opt.shape[0] > 0

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.wav"
    sf.write(output_path, np.asarray(audio_opt), tgt_sr)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_vc_inference_shapes(Path(tempfile.mkdtemp()))
