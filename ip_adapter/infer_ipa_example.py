"""
Minimal IP-Adapter inference example for Anima.

This shows how to make an already-loaded Anima DiT image-promptable with a
trained IP-Adapter sidecar, using independent character (CCIP) and style
(LSNet) references.

There are two ways to "infer":

  1) DURING TRAINING (already wired): set in the WebUI / toml
        sample_every_n_steps = 100
        sample_prompts       = ./sample_prompts.txt
        sample_reference_image = /path/to/ref.png
     The trainer's `sample_images` override encodes the reference and injects
     IP tokens automatically. Samples land in `<output_dir>/sample/`.

  2) STANDALONE (this file): wrap your DiT with `AnimaIPAdapter`, call
     `set_reference(...)`, then run your normal Anima generation. The IP tokens
     ride along inside every cross-attention via the concat fusion.

The actual DiT/VAE/text-encoder loading and the denoise loop are Anima-specific
and reused from your existing generation code — only the 3 marked lines below
are IP-Adapter-specific.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "vendor" / "sd-scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ip_adapter.anima_ip_adapter import AnimaIPAdapter  # noqa: E402


def main():
    device = "cuda"
    dtype = torch.bfloat16

    # ── 1. Load your Anima DiT however you normally do ───────────────
    #    (e.g. via library.anima_models, or your existing inference code).
    #    `dit` must be the same module class the trainer wraps (has "Block"
    #    submodules with `.cross_attn`).
    dit = load_my_anima_dit().to(device, dtype).eval()  # <-- your loader

    # ── 2. Attach the trained IP-Adapter (IP-Adapter-specific) ───────
    adapter = AnimaIPAdapter.from_pretrained(
        dit,
        sidecar_path="output/ipa/last.ipadapter.safetensors",
        clip_model="openai/clip-vit-large-patch14",
        ccip_ckpt="/root/lanyun-tmp/workspace/ccip/ccip-caformer_b36-24.ckpt",
        # lsnet_ckpt="/root/lanyun-tmp/workspace/lsnet/best_checkpoint.pth",
        device=device,
        dtype=dtype,
    )

    # ── 3. Set references (IP-Adapter-specific) ──────────────────────
    #    Character identity from one image, art style from another.
    adapter.set_reference(
        ccip_image="char_ref.png",  ccip_scale=1.0,   # who
        lsnet_image="style_ref.png", lsnet_scale=0.8,  # how it looks
        clip_image="char_ref.png",  clip_scale=0.5,    # general content
    )

    # ── 4. Run your normal Anima generation ──────────────────────────
    #    The DiT is now IP-aware; nothing else changes.
    image = generate_with_anima(dit, prompt="1girl, masterpiece")  # <-- your loop
    image.save("ipa_out.png")

    # Optional: drop IP tokens for a text-only generation on the same DiT.
    adapter.clear()


# ---- placeholders: replace with your existing Anima generation code ----
def load_my_anima_dit():
    raise NotImplementedError(
        "Plug in your Anima DiT loader here (the same model the trainer uses)."
    )


def generate_with_anima(dit, prompt: str):
    raise NotImplementedError(
        "Plug in your Anima sampling loop here. The IP tokens are already "
        "injected into the DiT, so no change to the loop is needed."
    )


if __name__ == "__main__":
    main()
