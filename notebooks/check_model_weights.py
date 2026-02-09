import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _(torch):
    hubert = torch.load("/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/pretrained/G32k.pth", weights_only=False)
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
def _(torch):
    rvc = torch.load("/home/mert/Desktop/projects/RVC/Retrieval-based-Voice-Conversion/assets/weights/rvc-model.pt", weights_only=False)
    return (rvc,)


@app.cell
def _(rvc):
    rvc
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
