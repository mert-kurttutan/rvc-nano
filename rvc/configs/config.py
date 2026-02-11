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
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.json_config = self.load_config_json()
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
        # 5G GPU_RAM conf
        x_pad = 1
        x_query = 6
        x_center = 38
        x_max = 41
        return x_pad, x_query, x_center, x_max

    def use_cpu(self) -> None:
        self.device = self.instead = "cpu"
        self.use_fp32_config()
        self.params_config()

    def use_fp32_config(self) -> None:
        for config_file, data in self.json_config.items():
            data["train"]["fp16_run"] = False
            with open(config_file, "w") as json_file:
                json.dump(data, json_file, indent=4)

    def device_config(self) -> tuple:
        return self.params_config()
