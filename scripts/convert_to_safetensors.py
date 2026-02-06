import torch
from safetensors.torch import save_file

rvc_model_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/weights/rvc-model.pt"

hubert_model_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/hubert_base.pt"
import json


def convert_to_safetensors(
    model_path: str,
    output_path: str,
    key: str,
    json_keys: list[tuple[str, str]],
):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"keys: {state_dict.keys()}")
    if key not in state_dict:
        raise KeyError(f"Key '{key}' not found in the model state dict.")
    model_state = state_dict[key]
    save_file(model_state, output_path)

    for json_key, json_path in json_keys:
        if json_key in state_dict:
            # save the json value to a .json file
            with open(json_path, "w") as f:
                json.dump(state_dict[json_key], f, indent=4)

    print(f"Converted '{key}' from {model_path} to {output_path}.")


if __name__ == "__main__":
    convert_to_safetensors(
        rvc_model_path,
        "rvc_model.safetensors",
        "weight",
        [("config", "rvc_model_config.json")],
    )
    convert_to_safetensors(
        hubert_model_path,
        "hubert_model.safetensors",
        "model",
        [("cfg", "hubert_cfg.json")],
    )
