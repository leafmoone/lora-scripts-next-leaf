import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _method_args(path: Path, class_name: str, method_name: str) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return [arg.arg for arg in item.args.args]
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def test_stable_sdxl_cache_text_encoder_signature_matches_vendor_base():
    method_name = "cache_text_encoder_outputs_if_needed"
    stable_args = _method_args(
        ROOT / "scripts" / "stable" / "sdxl_train_network.py",
        "SdxlNetworkTrainer",
        method_name,
    )
    base_args = _method_args(
        ROOT / "vendor" / "sd-scripts" / "train_network.py",
        "NetworkTrainer",
        method_name,
    )

    assert stable_args == base_args
