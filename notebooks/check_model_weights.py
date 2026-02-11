import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from torch.nn.utils.parametrizations import weight_norm
    from torch import nn


    return nn, torch, weight_norm


@app.cell
def _():
    f0g = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/pretrained/f0G48k.pth"
    g = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/pretrained/G48k.pth"
    return (g,)


@app.cell
def _(g, torch):
    hubert = torch.load(g, weights_only=False, map_location=torch.device("cpu"))
    return (hubert,)


@app.cell
def _(hubert):
    hubert.keys()
    return


@app.cell
def _(hubert):
    hubert["model"]
    return


@app.cell
def _(hubert):
    for i in sorted(hubert["model"].keys()):
        print(i)
        # print(i)
    return


@app.cell
def _(hubert):
    hubert["model"]
    return


@app.cell
def _():
    # rvc = torch.load("/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/weights/rvc-model.pt", weights_only=False)
    return


@app.cell
def _(nn, weight_norm):
    m = weight_norm(nn.Linear(20, 40), name='weight')
    return (m,)


@app.cell
def _(m):
    m.state_dict()
    return


@app.cell
def _(m):
    m
    return


@app.cell
def _():
    from rvc.vc.pipeline import Pipeline

    rvc_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/rvc_model.safetensors"
    hubert_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/hubert_model.safetensors"
    hubert_cfg_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/rvc/configs/hubert_cfg.json"
    rvc_cfg_path = "/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/rvc/configs/rvc_model_config.json"


    pipe = Pipeline(
        rvc_path=rvc_path,
        rvc_cfg_path=rvc_cfg_path,
        hubert_path=hubert_path,
        hubert_cfg_path=hubert_cfg_path,
    )
    return (pipe,)


@app.cell
def _(pipe):
    for idx in pipe.synthesizer.state_dict().keys():
        print(idx)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
