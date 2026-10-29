"""Anima Reference Adapter (v2) for Forge Neo, as a UNet patch.

The adapter reads a reference image through asymmetric attention: reference tokens attend only to
themselves and skip cross-attention; target tokens attend to themselves and the reference. The
reference stream therefore does not depend on the step, the prompt or the target, so its per-block
keys and values are computed once per image and reused by every step and CFG branch. Its RoPE (h, w)
coordinates sit one target height below the target grid, so it is read as appearance, not copied.
The adapter LoRA applies only to calls that carry the reference.

Reference guidance runs as a post-CFG function, as in the ComfyUI node:
    guided = cfg_out + (ref_guidance - 1) * (branch with reference - same branch without it)
followed by APG or CFG-rescale correction.
"""
from __future__ import annotations

import json
import logging
import math
import os

import huggingface_hub
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from backend.sampling.sampling_function import calc_cond_uncond_batch
from backend.utils import pad_to_patch_size

HF_REPO = "ElectronGirl/anima-ref-adapter"
HF_FILES = ("anima_ref_adapter_v2_r64.safetensors",)  # downloaded on first use
NO_REF_KEY = "anima_ref_off"  # transformer_options: run this call without the reference
HOOKS_ATTR = "_anima_ref_adapter_hooks"  # on each hooked module: {adapter key: hook handle}
_adapters: dict[tuple, "Adapter"] = {}
_active = {"key": None}  # key of the adapter whose LoRA the hooks apply right now


def lora_names(lora_dir: str) -> list[str]:
    """Released adapter files first (even before download), then the LoRAs on disk."""
    local = sorted(f for f in os.listdir(lora_dir) if f.endswith(".safetensors")) if os.path.isdir(lora_dir) else []
    return list(HF_FILES) + [name for name in local if name not in HF_FILES]


def lora_path(lora_dir: str, name: str) -> str:
    path = os.path.join(lora_dir, name)
    if not os.path.isfile(path):
        if name not in HF_FILES:
            raise FileNotFoundError(f"LoRA not found: {path}")
        logging.info(f"[Anima Ref Adapter] downloading {name} from huggingface.co/{HF_REPO}")
        path = huggingface_hub.hf_hub_download(HF_REPO, name, local_dir=lora_dir)
    return path


class Adapter:
    """LoRA weights applied through forward hooks, only while this adapter's key is active."""

    def __init__(self, path: str, strength: float):
        self.key = (os.path.abspath(path), os.path.getmtime(path), float(strength))
        with safe_open(path, "pt") as handle:
            metadata = json.loads((handle.metadata() or {}).get("anima_ref_adapter", "{}"))
        if metadata.get("arch") != "asym":
            raise ValueError(f"{os.path.basename(path)} is not a v2 (asymmetric) Anima reference adapter; "
                             "the pilot adapter is supported only in ComfyUI")
        self.offset_h = float(metadata["slots"]["character"]["offset_h"])
        self.ref_lora = metadata.get("ref_lora", True)
        tensors = load_file(path)
        self.weights = {}
        for key, down in tensors.items():
            if key.endswith(".lora_down.weight"):
                up = tensors[key.replace("lora_down", "lora_up")]
                alpha = float(tensors.get(key.replace("lora_down.weight", "alpha"), torch.tensor(down.shape[0])))
                self.weights[key[len("diffusion_model."):-len(".lora_down.weight")]] = [down, up, strength * alpha / down.shape[0]]

    @property
    def nbytes(self) -> int:
        return sum(down.numel() * 2 + up.numel() * 2 for down, up, _ in self.weights.values())  # 16-bit on the GPU

    def offload(self) -> None:
        """Back to the CPU after sampling: Forge's memory manager does not track these tensors."""
        for entry in self.weights.values():
            entry[0], entry[1] = entry[0].cpu(), entry[1].cpu()

    @classmethod
    def get(cls, path: str, strength: float) -> "Adapter":
        key = (os.path.abspath(path), os.path.getmtime(path), float(strength))
        if key not in _adapters:
            _adapters[key] = cls(path, strength)
        return _adapters[key]

    def install(self, diffusion_model) -> None:
        for name, entry in self.weights.items():
            module = diffusion_model.get_submodule(name)
            installed = getattr(module, HOOKS_ATTR, None)
            if installed is None:
                installed = {}
                setattr(module, HOOKS_ATTR, installed)
            if self.key in installed:
                continue

            def hook(module, inputs, output, entry=entry, key=self.key):
                if _active["key"] != key:
                    return output
                x = inputs[0]
                if entry[0].device != x.device or entry[0].dtype != x.dtype:
                    entry[0] = entry[0].to(device=x.device, dtype=x.dtype)
                    entry[1] = entry[1].to(device=x.device, dtype=x.dtype)
                return output + F.linear(F.linear(x, entry[0]), entry[1]) * entry[2]

            installed[self.key] = module.register_forward_hook(hook)


def _rope(pe, t_index: int, height: int, width: int, h0: int, w0: int, device) -> torch.Tensor:
    """VideoRopePosition3DEmb for one frame at an explicit (t, h, w) offset: [L, D/2, 2, 2]."""
    hh, ww = torch.meshgrid(torch.arange(height, device=device) + h0, torch.arange(width, device=device) + w0, indexing="ij")
    t, h, w = torch.full_like(hh, t_index).flatten(), hh.flatten(), ww.flatten()

    def rotations(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)

    h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(device))
    t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(device))
    emb = torch.cat([rotations(t, t_freqs), rotations(h, h_freqs), rotations(w, w_freqs)], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _frame_embed(dm, latent, t_index: int, h0: int, w0: int, sigma):
    """Patch embedding, RoPE and timestep embedding of one frame [B, C, 1, H, W]."""
    x_emb, _, _ = dm.prepare_embedded_sequence(latent)
    _, _, hp, wp, _ = x_emb.shape
    rope = _rope(dm.pos_embedder, t_index, hp, wp, h0, w0, latent.device)
    # the timestep embedding sets the compute dtype, so take it before the residual stream goes to fp32
    emb, adaln = dm.t_embedder[1](dm.t_embedder[0](sigma.reshape(-1, 1).float()).to(x_emb.dtype))
    if x_emb.dtype == torch.float16:
        x_emb = x_emb.float()  # residual stream in fp32, as Anima.forward does for fp16
    return x_emb, rope.unsqueeze(1).unsqueeze(0), dm.t_embedding_norm(emb), adaln


def _mods(block, emb, adaln):
    """(shift, scale, gate) [B, T, 1, 1, D] for self-attention, cross-attention and MLP."""
    return [tuple(t[:, :, None, None, :] for t in (m(emb) + adaln).chunk(3, dim=-1))
            for m in (block.adaln_modulation_self_attn, block.adaln_modulation_cross_attn, block.adaln_modulation_mlp)]


def _norm(x, layer_norm, scale, shift):
    return torch.addcmul(shift, layer_norm(x), 1 + scale)


def _asym_block(block, xt, xr, emb_t, emb_r, adaln_t, adaln_r, context, rope_t, rope_r, ref_kv=None, options=None):
    """Block.forward with asymmetric attention. Returns (xt, xr, (k_r, v_r))."""
    attn = block.self_attn
    k_r = v_r = None
    if xr is not None:
        sa, _, mlp = _mods(block, emb_r, adaln_r)
        residual, compute = xr.dtype, emb_r.dtype
        B, T, H, W, _ = xr.shape
        n = _norm(xr, block.layer_norm_self_attn, sa[1], sa[0]).to(compute).flatten(1, 3)
        q_r, k_r, v_r = attn.compute_qkv(n, None, rope_emb=rope_r)
        xr = torch.addcmul(xr, sa[2].to(residual), attn.compute_attention(q_r, k_r, v_r).reshape(B, T, H, W, -1).to(residual))
        n = _norm(xr, block.layer_norm_mlp, mlp[1], mlp[0]).to(compute)
        xr = torch.addcmul(xr, mlp[2].to(residual), block.mlp(n).to(residual))
    elif ref_kv is not None:
        k_r, v_r = ref_kv
    if xt is not None:
        options = options or {}
        sa, ca, mlp = _mods(block, emb_t, adaln_t)
        residual, compute = xt.dtype, emb_t.dtype
        B, T, H, W, _ = xt.shape
        n = _norm(xt, block.layer_norm_self_attn, sa[1], sa[0]).to(compute).flatten(1, 3)
        q_t, k_t, v_t = attn.compute_qkv(n, None, rope_emb=rope_t)
        k_t, v_t = torch.cat([k_t, k_r.to(k_t)], dim=1), torch.cat([v_t, v_r.to(v_t)], dim=1)
        out = attn.compute_attention(q_t, k_t, v_t, transformer_options=options).reshape(B, T, H, W, -1)
        xt = torch.addcmul(xt, sa[2].to(residual), out.to(residual))
        n = _norm(xt, block.layer_norm_cross_attn, ca[1], ca[0]).to(compute).flatten(1, 3)
        out = block.cross_attn(n, context, rope_emb=None, transformer_options=options).reshape(B, T, H, W, -1)
        xt = torch.addcmul(xt, ca[2].to(residual), out.to(residual))
        n = _norm(xt, block.layer_norm_mlp, mlp[1], mlp[0]).to(compute)
        xt = torch.addcmul(xt, mlp[2].to(residual), block.mlp(n).to(residual))
    return xt, xr, (k_r, v_r)


def encode_ref(dm, ref, offset: tuple[int, int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-block reference keys/values: the reference stream run once."""
    xr, rope_r, emb_r, adaln_r = _frame_embed(dm, ref, 1, *offset, torch.zeros(ref.shape[0], device=ref.device))
    cache = []
    for block in dm.blocks:
        _, xr, kv = _asym_block(block, None, xr, None, emb_r, None, adaln_r, None, None, rope_r)
        cache.append(kv)
    return cache


def asym_forward(dm, x, timesteps, context, ref_kv, options):
    """Anima.forward for the target frame alone, attending to cached reference keys/values."""
    orig_shape = list(x.shape)
    x = pad_to_patch_size(x, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    xt, rope_t, emb_t, adaln_t = _frame_embed(dm, x[:, :, :1], 0, 0, 0, timesteps.reshape(x.shape[0], -1)[:, 0])
    batch = xt.shape[0]
    for block, (k, v) in zip(dm.blocks, ref_kv):
        kv = (k.expand(batch, *k.shape[1:]), v.expand(batch, *v.shape[1:])) if k.shape[0] != batch else (k, v)
        xt, _, _ = _asym_block(block, xt, None, emb_t, None, adaln_t, None, context, rope_t, None, kv, options)
    out = dm.unpatchify(dm.final_layer(xt.to(context.dtype), emb_t, adaln_lora_B_T_3D=adaln_t))[:, :, :1]
    return out[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]


def _correct(guided, pos, correction: str):
    """APG (drop the guidance component parallel to the conditional prediction) or CFG rescale (phi 0.5)."""
    dims = tuple(range(1, pos.ndim))
    if correction == "apg":
        update = (guided - pos).double()
        unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
        parallel = (update * unit).sum(dim=dims, keepdim=True) * unit
        return pos + (update - parallel).to(pos.dtype)
    if correction == "rescale":
        ratio = pos.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True)
        return 0.5 * guided * ratio + 0.5 * guided
    return guided


def reference_image(image: Image.Image, second: Image.Image | None, megapixels: float) -> torch.Tensor:
    """The reference as [1, H, W, 3] in 0..1: a second image is placed to the right at the same height
    (as ComfyUI's Image Stitch), then the whole is resized to `megapixels` with its aspect ratio."""
    image = image.convert("RGB")
    if second is not None:
        second = second.convert("RGB")
        second = second.resize((round(second.width * image.height / second.height), image.height), Image.LANCZOS)
        canvas = Image.new("RGB", (image.width + second.width, image.height))
        canvas.paste(image, (0, 0))
        canvas.paste(second, (image.width, 0))
        image = canvas
    scale = (megapixels * 1024 * 1024 / (image.width * image.height)) ** 0.5
    size = tuple(max(16, round(d * scale / 16) * 16) for d in image.size)
    return torch.from_numpy(np.asarray(image.resize(size, Image.LANCZOS), dtype=np.float32) / 255.0).unsqueeze(0)


def patch(unet, adapter: Adapter, ref: torch.Tensor, ref_guidance: float, correction: str, ref_end: float, cache: dict):
    """A clone of `unet` that reads the reference latent `ref` [1, C, 1, H, W] (already process_in'ed).
    `cache` holds the reference keys/values; the caller empties it after sampling."""
    m = unet.clone()
    tokens = (ref.shape[-2] // 2) * (ref.shape[-1] // 2)
    dm = unet.model.diffusion_model
    kv_bytes = 2 * len(dm.blocks) * tokens * dm.blocks[0].x_dim * 2  # k and v per block, 16-bit
    m.add_extra_preserved_memory_during_sampling(kv_bytes + adapter.nbytes)

    def ref_kv(model, x):
        hp = -(-x.shape[-2] // model.diffusion_model.patch_spatial)
        key = (id(model.diffusion_model), round(adapter.offset_h * hp), x.dtype, x.device)
        if key not in cache:
            cache.clear()
            r = pad_to_patch_size(ref.to(device=x.device, dtype=x.dtype), (1, 2, 2))
            previous = _active["key"]
            _active["key"] = adapter.key if adapter.ref_lora else None
            try:
                cache[key] = encode_ref(model.diffusion_model, r, (key[1], 0))
            finally:
                _active["key"] = previous
        return cache[key]

    def model_wrapper(apply_model, args):
        c = args["c"]
        options = c.get("transformer_options", {})
        if options.get(NO_REF_KEY):
            return apply_model(args["input"], args["timestep"], **c)
        model = apply_model.__self__
        x, sigma = args["input"], args["timestep"]
        dtype = model.computation_dtype
        xc = model.predictor.calculate_input(sigma, x).to(dtype)
        t = model.predictor.timestep(sigma).float()
        context = c["c_crossattn"].to(dtype)
        adapter.install(model.diffusion_model)
        kv = ref_kv(model, xc)
        previous = _active["key"]
        _active["key"] = adapter.key
        try:
            out = asym_forward(model.diffusion_model, xc, t, context, kv, options).float()
        finally:
            _active["key"] = previous
        return model.predictor.calculate_denoised(sigma, out, x)

    def post_cfg(args):
        guided, pos = args["denoised"], args["cond_denoised"]
        model = args["model"]
        sigma = float(args["sigma"].flatten()[0])
        if ref_guidance != 1.0 and (ref_end >= 1.0 or sigma >= float(model.predictor.percent_to_sigma(ref_end)) - 1e-5):
            # reference effect = (a branch with the reference) - (the same branch without it): the uncond
            # branch, or the cond branch when cfg is 1 and Forge skipped the uncond
            if math.isclose(args["cond_scale"], 1.0):
                cond, ref_pred = args["cond"], pos
            else:
                cond, ref_pred = args["uncond"], args["uncond_denoised"]
            options = args["model_options"].copy()
            options["transformer_options"] = {**options.get("transformer_options", {}), NO_REF_KEY: True}
            base, _ = calc_cond_uncond_batch(model, cond, None, args["input"], args["sigma"], options)
            guided = guided + (ref_guidance - 1.0) * (ref_pred - base)
        return _correct(guided, pos, correction)

    m.set_model_unet_function_wrapper(model_wrapper)
    m.set_model_sampler_post_cfg_function(post_cfg)
    return m
