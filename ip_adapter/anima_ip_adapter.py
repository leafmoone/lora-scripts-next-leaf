"""
Anima IP-Adapter Inference Wrapper (multi-stream, concat fusion)

Loads a trained IP-Adapter sidecar (``*.ipadapter.safetensors``) and makes an
Anima DiT image-promptable at inference time. It mirrors the training pipeline
exactly:

  * CLIP (224, CLIP-norm)   → clip_proj  (768→1024) ┐
  * CCIP (384, CLIP-norm)   → ccip_proj  (768→1024) ├─ image_proj → IP tokens
  * LSNet (448, ImageNet)   → lsnet_proj (768→1024) ┘
  * IP tokens are concatenated onto the text context inside every DiT Block's
    cross-attention (see ``AnimaIPCrossAttention``).

Each stream is independent, so you can pass a *different* reference image and a
*different* scale for the character (CCIP) and style (LSNet) streams — which is
exactly how you drive "通用角色参考 + 画风参考" at once.

Typical usage
-------------
    from ip_adapter.anima_ip_adapter import AnimaIPAdapter

    adapter = AnimaIPAdapter.from_pretrained(
        dit,                                   # your loaded Anima DiT (nn.Module)
        "output/ipa/last.ipadapter.safetensors",
        clip_model="openai/clip-vit-large-patch14",
        ccip_ckpt="/path/ccip-caformer_b36-24.ckpt",   # only if aux has ccip
        lsnet_ckpt="/path/best_checkpoint.pth",         # only if aux has lsnet
        device="cuda", dtype=torch.bfloat16,
    )

    # one reference for everything, or per-stream references:
    adapter.set_reference(
        ccip_image="char_ref.png", ccip_scale=1.0,     # identity
        lsnet_image="style_ref.png", lsnet_scale=0.8,  # style
        clip_image="char_ref.png", clip_scale=0.6,     # content
    )

    # ... run your normal Anima generation loop here; tokens inject themselves ...

    adapter.clear()   # remove IP tokens (e.g. before a text-only generation)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .anima_ip_attention import AnimaIPCrossAttention
from .anima_ip_converter import AnimaIPAConverter
from .anima_ip_image_proj import ImageProjModel, MultiStreamProj

# CLIP image normalization (OpenAI CLIP mean/std), applied to [0,1] pixels.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

ImageLike = Union[str, "torch.Tensor", "object"]  # path | tensor | PIL.Image


def _clip_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _make_projection(in_dim: int = 768, out_dim: int = 1024) -> nn.Module:
    """Trainable 768→1024 projection (fp32) — must match the trainer exactly."""
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)).float()


def _to_image01(img: ImageLike) -> torch.Tensor:
    """Coerce a path / PIL.Image / tensor into a (1, 3, H, W) float tensor in [0,1]."""
    if isinstance(img, str):
        from PIL import Image

        img = Image.open(img).convert("RGB")
    if isinstance(img, torch.Tensor):
        t = img.float()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.max() > 1.5:  # assume 0..255
            t = t / 255.0
        return t.clamp(0, 1)
    # PIL.Image
    import numpy as np

    arr = np.array(img.convert("RGB"), dtype="float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.clamp(0, 1)


class AnimaIPAdapter:
    """Apply trained IP-Adapter weights to an Anima DiT at inference time."""

    def __init__(
        self,
        dit: nn.Module,
        *,
        aux_encoders: tuple[str, ...],
        num_ip_tokens: int,
        clip_image_encoder: nn.Module,
        clip_proj: nn.Module,
        image_proj: Union[ImageProjModel, MultiStreamProj],
        ccip_encoder: Optional[nn.Module] = None,
        ccip_proj: Optional[nn.Module] = None,
        lsnet_encoder: Optional[nn.Module] = None,
        lsnet_proj: Optional[nn.Module] = None,
        image_proj_resampler: Optional[MultiStreamProj] = None,
        ipa_mode: str = "simple",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dit = dit
        self.aux_encoders = aux_encoders
        self.num_ip_tokens = num_ip_tokens
        self.device = torch.device(device)
        self.dtype = dtype

        self.clip_image_encoder = clip_image_encoder
        self.clip_proj = clip_proj
        self.ccip_encoder = ccip_encoder
        self.ccip_proj = ccip_proj
        self.lsnet_encoder = lsnet_encoder
        self.lsnet_proj = lsnet_proj
        self.image_proj = image_proj
        self.image_proj_resampler = image_proj_resampler
        self.ipa_mode = ipa_mode

        # Inject concat-fusion cross-attention into every DiT Block.
        self.ip_adapters: Dict[str, AnimaIPCrossAttention] = AnimaIPAConverter.create(dit)

        self._move_to_device()
        self._eval_all()

    # ── construction ───────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        dit: nn.Module,
        sidecar_path: str,
        *,
        clip_model: str = "openai/clip-vit-large-patch14",
        ccip_ckpt: str = "",
        lsnet_ckpt: str = "",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "AnimaIPAdapter":
        """Reconstruct encoders + projections from a sidecar and load its weights."""
        from safetensors import safe_open

        if not os.path.isfile(sidecar_path):
            raise FileNotFoundError(f"IP-Adapter sidecar not found: {sidecar_path}")

        with safe_open(sidecar_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            state_dict = {k: f.get_tensor(k) for k in f.keys()}

        aux_str = metadata.get("ipa_aux_encoders", "")
        aux_encoders = tuple(a for a in aux_str.split(",") if a in ("ccip", "lsnet"))
        num_streams = 1 + len(aux_encoders)
        num_ip_tokens = int(metadata.get("ipa_num_ip_tokens", "4"))
        nt_clip = int(metadata.get("ipa_num_ip_tokens_clip", str(num_ip_tokens)))
        nt_ccip = int(metadata.get("ipa_num_ip_tokens_ccip", str(num_ip_tokens)))
        nt_lsnet = int(metadata.get("ipa_num_ip_tokens_lsnet", str(num_ip_tokens)))
        cad = int(metadata.get("ipa_cross_attention_dim", "1024"))
        ipa_mode = metadata.get("ipa_mode", "simple")
        need_patches = ipa_mode != "simple"

        # Per-stream token list: [CLIP, CCIP?, LSNet?]
        tokens_per_stream = [nt_clip]
        if "ccip" in aux_encoders:
            tokens_per_stream.append(nt_ccip)
        if "lsnet" in aux_encoders:
            tokens_per_stream.append(nt_lsnet)

        # CLIP (always present)
        clip_image_encoder = cls._load_clip(clip_model, device, dtype)
        clip_proj = _make_projection()

        ccip_encoder = ccip_proj = lsnet_encoder = lsnet_proj = None
        if "ccip" in aux_encoders:
            from .ccip_encoder import load_ccip_encoder, DEFAULT_CKPT as _DEF_CCIP

            ccip_encoder = load_ccip_encoder(
                ckpt_path=ccip_ckpt or _DEF_CCIP, device="cpu", dtype=dtype
            )
            ccip_proj = _make_projection()
        if "lsnet" in aux_encoders:
            if not lsnet_ckpt:
                raise ValueError("lsnet_ckpt is required: this sidecar uses an LSNet stream")
            from .lsnet_encoder import load_lsnet_encoder

            lsnet_encoder = load_lsnet_encoder(ckpt_path=lsnet_ckpt, device="cpu", dtype=dtype)
            lsnet_proj = _make_projection()

        if num_streams > 1:
            image_proj = MultiStreamProj(
                num_streams=num_streams,
                cross_attention_dim=cad,
                embed_dim=cad,
                tokens_per_stream=tokens_per_stream,
            )
        else:
            image_proj = ImageProjModel(
                cross_attention_dim=cad,
                clip_embeddings_dim=cad,
                clip_extra_context_tokens=tokens_per_stream[0],
            )

        image_proj_resampler = None
        if ipa_mode in ("resampler", "double"):
            from .anima_ip_image_proj import Resampler
            nq_clip = int(metadata.get("ipa_num_queries_clip", metadata.get("ipa_num_queries", str(num_ip_tokens))))
            nq_ccip = int(metadata.get("ipa_num_queries_ccip", str(nq_clip)))
            nq_lsnet = int(metadata.get("ipa_num_queries_lsnet", str(nq_clip)))
            queries_per_stream = [nq_clip]
            if "ccip" in aux_encoders:
                queries_per_stream.append(nq_ccip)
            if "lsnet" in aux_encoders:
                queries_per_stream.append(nq_lsnet)
            modules = [
                Resampler(dim=1024, depth=4, dim_head=64, heads=16,
                          num_queries=nq, output_dim=1024)
                for nq in queries_per_stream
            ]
            if ipa_mode == "resampler":
                image_proj = MultiStreamProj.from_modules(modules)
            else:
                # double mode: keep image_proj as global, resampler for fine
                image_proj_resampler = MultiStreamProj.from_modules(modules)

        # Load trained weights into each module by prefix.
        for prefix, mod in (
            ("clip_proj", clip_proj),
            ("ccip_proj", ccip_proj),
            ("lsnet_proj", lsnet_proj),
            ("image_proj", image_proj),
            ("image_proj_resampler", image_proj_resampler),
        ):
            if mod is None:
                continue
            sub = {k[len(prefix) + 1:]: v for k, v in state_dict.items()
                   if k.startswith(prefix + ".")}
            if not sub:
                raise RuntimeError(
                    f"Sidecar is missing weights for '{prefix}' — it does not match "
                    f"the requested encoder configuration (aux={aux_encoders})."
                )
            mod.load_state_dict(sub)

        return cls(
            dit,
            aux_encoders=aux_encoders,
            num_ip_tokens=num_ip_tokens,
            ipa_mode=ipa_mode,
            clip_image_encoder=clip_image_encoder,
            clip_proj=clip_proj,
            image_proj=image_proj,
            image_proj_resampler=image_proj_resampler,
            ccip_encoder=ccip_encoder,
            ccip_proj=ccip_proj,
            lsnet_encoder=lsnet_encoder,
            lsnet_proj=lsnet_proj,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _load_clip(model_id: str, device, dtype) -> nn.Module:
        from transformers import CLIPVisionModelWithProjection

        model = CLIPVisionModelWithProjection.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _move_to_device(self):
        for m in (self.clip_image_encoder, self.ccip_encoder, self.lsnet_encoder):
            if m is not None:
                m.to(self.device)
        # Projections stay in fp32 for numerical parity with training.
        for m in (self.clip_proj, self.ccip_proj, self.lsnet_proj, self.image_proj,
                  self.image_proj_resampler):
            if m is not None:
                m.to(self.device).float()

    def _eval_all(self):
        for m in (self.clip_image_encoder, self.ccip_encoder, self.lsnet_encoder,
                  self.clip_proj, self.ccip_proj, self.lsnet_proj, self.image_proj):
            if m is not None:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    # ── encoding ────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        clip_image: Optional[ImageLike] = None,
        ccip_image: Optional[ImageLike] = None,
        lsnet_image: Optional[ImageLike] = None,
        *,
        clip_images: Optional[list[ImageLike]] = None,
        ccip_images: Optional[list[ImageLike]] = None,
        lsnet_images: Optional[list[ImageLike]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Encode per-stream reference images into IP tokens.

        Each stream accepts a single image or a list.  When a list is given,
        each image is encoded independently and the resulting **1024-dim
        projected features are averaged** before projection to IP tokens.
        This fuses multiple viewpoints into one set of IP tokens.

        Any stream left as ``None`` is skipped.
        Returns ``{"clip": tokens, "ccip": tokens|None, "lsnet": tokens|None}``.
        """

        def _resolve(single, multi) -> list[torch.Tensor]:
            sources: list[ImageLike] = []
            if single is not None:
                sources.append(single)
            if multi:
                sources.extend(multi)
            return [_to_image01(s).to(self.device, self.dtype) for s in sources]

        clip_embeds = ccip_embeds = lsnet_embeds = None
        clip_patches = ccip_patches = lsnet_patches = None
        need_patches = self.ipa_mode != "simple"

        # ── CLIP ─────────────────────────────────────────────
        clip_sources = _resolve(clip_image, clip_images)
        if clip_sources:
            feats, patch_feats = [], []
            for x in clip_sources:
                x = interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                x = _clip_normalize(x)
                out = self.clip_image_encoder(x, output_hidden_states=need_patches)
                feats.append(self.clip_proj(out.image_embeds.float()))
                if need_patches:
                    patch_feats.append(out.hidden_states[-1][:, 1:, :].float())  # (1, 256, 1024)
            clip_embeds = torch.stack(feats).mean(dim=0)
            if need_patches:
                clip_patches = torch.cat(patch_feats).mean(dim=0, keepdim=True)

        # ── CCIP ─────────────────────────────────────────────
        ccip_sources = _resolve(ccip_image, ccip_images)
        if ccip_sources and self.ccip_encoder is not None:
            feats, patch_feats = [], []
            for x in ccip_sources:
                x = interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)
                result = self.ccip_encoder(x, return_patches=need_patches)
                if need_patches:
                    f, p = result
                    patch_feats.append(p.float())  # (1, 144, 768)
                else:
                    f = result
                feats.append(self.ccip_proj(f.float()))
            ccip_embeds = torch.stack(feats).mean(dim=0)
            if need_patches:
                ccip_patches = torch.cat(patch_feats).mean(dim=0, keepdim=True)

        # ── LSNet ────────────────────────────────────────────
        lsnet_sources = _resolve(lsnet_image, lsnet_images)
        if lsnet_sources and self.lsnet_encoder is not None:
            feats, patch_feats = [], []
            for x in lsnet_sources:
                x = interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
                result = self.lsnet_encoder(x, return_patches=need_patches)
                if need_patches:
                    f, p = result
                    patch_feats.append(p.float())  # (1, 196, 768)
                else:
                    f = result
                feats.append(self.lsnet_proj(f.float()))
            lsnet_embeds = torch.stack(feats).mean(dim=0)
            if need_patches:
                lsnet_patches = torch.cat(patch_feats).mean(dim=0, keepdim=True)

        tokens = {"clip": None, "ccip": None, "lsnet": None, "clip_fine": None}
        if isinstance(self.image_proj, MultiStreamProj):
            # MultiStreamProj projects positionally: index 0 = clip, then aux order.
            stream_order = ["clip", *self.aux_encoders]
            embeds_by_name = {"clip": clip_embeds, "ccip": ccip_embeds, "lsnet": lsnet_embeds}
            for i, name in enumerate(stream_order):
                emb = embeds_by_name.get(name)
                if emb is not None:
                    tokens[name] = self.image_proj.projs[i](emb).to(self.dtype)
        else:
            if clip_embeds is not None:
                tokens["clip"] = self.image_proj(clip_embeds).to(self.dtype)

        # ── Fine IP tokens (resampler / double mode) ─────────
        if self.image_proj_resampler is not None:
            # Encode patch features per stream, project to 1024, then resample
            patch_feats = {"clip": clip_patches, "ccip": ccip_patches, "lsnet": lsnet_patches}
            stream_order = ["clip", *self.aux_encoders]
            for i, name in enumerate(stream_order):
                p = patch_feats.get(name)
                if p is not None and p.numel() > 0:
                    B, L, D = p.shape
                    if D != 1024:
                        # CCIP/LSNet patches are 768-dim → project via encoder proj
                        proj = {"ccip": self.ccip_proj, "lsnet": self.lsnet_proj}[name]
                        p = proj(p.reshape(-1, D)).reshape(B, L, -1)
                    tokens[f"{name}_fine"] = self.image_proj_resampler.projs[i](p).to(self.dtype)

        return tokens

    # ── injection ───────────────────────────────────────────────

    def set_reference(
        self,
        image: Optional[ImageLike] = None,
        *,
        clip_image: Optional[ImageLike] = None,
        ccip_image: Optional[ImageLike] = None,
        lsnet_image: Optional[ImageLike] = None,
        scale: float = 1.0,
        clip_scale: Optional[float] = None,
        ccip_scale: Optional[float] = None,
        lsnet_scale: Optional[float] = None,
    ) -> None:
        """Encode reference image(s) and stash IP tokens on every Block.

        ``image`` is the fallback used for any stream without an explicit image.
        Per-stream scales are baked into the tokens (so the attention layer's own
        ``ip_scale`` is left at 1.0). Pass a stream's image as ``None`` *and* no
        fallback to disable that stream.
        """
        clip_src = clip_image if clip_image is not None else image
        ccip_src = ccip_image if ccip_image is not None else image
        lsnet_src = lsnet_image if lsnet_image is not None else image

        tokens = self.encode(
            clip_image=clip_src,
            ccip_image=ccip_src if self.ccip_encoder is not None else None,
            lsnet_image=lsnet_src if self.lsnet_encoder is not None else None,
        )

        def _scaled(name, default):
            t = tokens.get(name)
            if t is None:
                return None
            s = {"clip": clip_scale, "ccip": ccip_scale, "lsnet": lsnet_scale}[name]
            s = scale if s is None else s
            return t * s if s != 1.0 else t

        ip = _scaled("clip", scale)
        ccip = _scaled("ccip", scale)
        lsnet = _scaled("lsnet", scale)
        ip_fine = _scaled("clip_fine", scale) if self.image_proj_resampler is not None else None
        self._stash(ip, ccip, lsnet, ip_fine)

    def _stash(self, ip, ccip, lsnet, ip_fine=None):
        for attn in self.ip_adapters.values():
            attn.ip_scale = 1.0  # per-stream scale already baked into the tokens
            attn._ip_tokens = ip
            attn._ip_tokens_fine = ip_fine
            attn._ip_tokens_ccip = ccip
            attn._ip_tokens_lsnet = lsnet

    def clear(self) -> None:
        """Remove IP tokens so subsequent generations are text-only."""
        self._stash(None, None, None)
