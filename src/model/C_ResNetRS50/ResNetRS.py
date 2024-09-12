from .blocks import ResNetRS
import torch.nn as nn


class ResNet(object):
    config_set = {
        "resnetrs50": {"layers": [3, 4, 6, 3]}, # 3 4 6 3
        "resnetrs101": {"layers": [3, 4, 23, 3]},
        "resnetrs152": {"layers": [3, 8, 36, 3]},
        "resnetrs200": {"layers": [3, 24, 36, 3]},
        "resnetrs270": {"layers": [4, 29, 53, 4]},
        "resnetrs350": {"layers": [4, 36, 72, 4]},
        "resnetrs420": {"layers": [4, 44, 87, 4]},
    }
    for config in config_set.values():
        config.update({"se_ratio": 0.25, "num_classes": 7})

    @classmethod
    def build_model(self, name: str) -> nn.Module:
        return ResNetRS(self.config_set[name])