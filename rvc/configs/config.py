import json
import os
from multiprocessing import cpu_count

import torch

version_config_list: list = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__)))
    for file in files
    if file.endswith(".json")
]


class Config:
    def __init__(self):
        self.device: str = "cuda:0"
        self.is_half: bool = True
        self.n_cpu: int = cpu_count()
        self.gpu_name: str | None = None
        self.json_config = self.load_config_json()
        self.gpu_mem: int | None = None
        self.instead: str | None = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        configs: dict = {}
        for config_file in version_config_list:
            with open(config_file) as handle:
                configs[config_file] = json.load(handle)
        return configs

    def params_config(self) -> tuple:
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        elif self.is_half:
            # 6G PU_RAM conf
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G GPU_RAM conf
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41
        return x_pad, x_query, x_center, x_max

    def use_cuda(self) -> None:
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        if (
            ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
            or "P40" in self.gpu_name.upper()
            or "P10" in self.gpu_name.upper()
            or "1060" in self.gpu_name
            or "1070" in self.gpu_name
            or "1080" in self.gpu_name
        ):
            self.is_half = False
            self.use_fp32_config()
        self.gpu_mem = int(torch.cuda.get_device_properties(i_device).total_memory / 1024 / 1024 / 1024 + 0.4)

    def use_cpu(self) -> None:
        self.device = self.instead = "cpu"
        self.is_half = False
        self.use_fp32_config()
        self.params_config()

    def use_fp32_config(self) -> None:
        for config_file, data in self.json_config.items():
            data["train"]["fp16_run"] = False
            with open(config_file, "w") as json_file:
                json.dump(data, json_file, indent=4)

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            self.use_cuda()
        return self.params_config()
