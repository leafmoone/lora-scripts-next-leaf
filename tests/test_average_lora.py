import importlib.util
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AVERAGE_LORA_PATH = PROJECT_ROOT / "tools" / "average_lora.py"


spec = importlib.util.spec_from_file_location("average_lora", AVERAGE_LORA_PATH)
average_lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(average_lora)


def test_find_lora_pairs_supports_lora_a_b_keys():
    state = {
        "block.lora_A.weight": torch.zeros(4, 8),
        "block.lora_B.weight": torch.zeros(16, 4),
        "block.alpha": torch.tensor(4),
    }

    assert average_lora.find_lora_pairs(state) == [
        ("block", "block.lora_A.weight", "block.lora_B.weight")
    ]


def test_find_lora_pairs_supports_lora_down_up_keys():
    state = {
        "diffusion_model.block.lora_down.weight": torch.zeros(4, 8),
        "diffusion_model.block.lora_up.weight": torch.zeros(16, 4),
        "diffusion_model.block.alpha": torch.tensor(4),
    }

    assert average_lora.find_lora_pairs(state) == [
        (
            "diffusion_model.block",
            "diffusion_model.block.lora_down.weight",
            "diffusion_model.block.lora_up.weight",
        )
    ]
