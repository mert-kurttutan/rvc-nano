# /// script
# dependencies = [
#   "fairseq @ git+https://github.com/Tps-F/fairseq.git@main",
#   "safetensors",
# ]
# ///
import os

import torch
from safetensors.torch import save_file


def remove_unused_parameters(state_dict: dict) -> dict:
    updated_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("enc_q"):
            continue
        if key == "emb_g.bias":
            continue
        updated_state_dict[key] = value

    return updated_state_dict


def update_weight_normalization(state_dict: dict) -> dict:
    """Updates every weight-normalized key from old to new naming convention
    weight_g -> weight.original0
    weight_v -> weight.original1
    see the migration guide here: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
    """

    updated_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith("weight_g"):
            new_key = key[: -len("weight_g")] + "parametrizations.weight.original0"
            updated_state_dict[new_key] = value
        elif key.endswith("weight_v"):
            new_key = key[: -len("weight_v")] + "parametrizations.weight.original1"
            updated_state_dict[new_key] = value
        else:
            updated_state_dict[key] = value
    return updated_state_dict


def convert_to_safetensors(
    model_path: str,
):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    print("-" * 32)
    print(f"Loaded model from '{model_path}'.")
    if "model" in state_dict.keys():
        model_state = state_dict["model"]
    elif "weight" in state_dict.keys():
        model_state = state_dict["weight"]
    else:
        assert model_path.endswith("rmvpe.pt"), "Expected the key to be either 'model' or 'weight'."
        model_state = state_dict
    model_state_updated = update_weight_normalization(model_state)
    model_state_updated = remove_unused_parameters(model_state_updated)
    # get the file extension by using right most dot
    output_path = model_path.rsplit(".", 1)[0] + ".safetensors"
    save_file(model_state_updated, output_path)


def convert_all_files(folder: str):
    """
    Scans all the pt/pth files inside folder (including subfolders) and converts them to safetensors format using the
    convert_to_safetensors function.
    """

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".pt") or file.endswith(".pth"):
                model_path = os.path.join(root, file)
                convert_to_safetensors(model_path)


if __name__ == "__main__":
    convert_all_files("/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets")
