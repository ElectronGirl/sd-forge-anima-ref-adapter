<div align="center">

<img src="https://raw.githubusercontent.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/main/docs/showcase/banner.jpg" width="100%" alt="Anima Reference Adapter v2">

# Anima Reference Adapter for Forge Neo

**Character and style from a reference image. Pose, place and framing from your prompt.**

[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-anima--ref--adapter-ffcc4d)](https://huggingface.co/ElectronGirl/anima-ref-adapter)
[![ComfyUI](https://img.shields.io/badge/%F0%9F%A7%A9%20ComfyUI-nodes-1f6feb)](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter)
[![Training code](https://img.shields.io/badge/%F0%9F%94%AC%20training-code-8a63d2)](https://github.com/ElectronGirl/anima-ref-adapter-research)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-support%20training-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/electrongirl)

**English** · [한국어](README.ko.md) · [日本語](README.ja.md)

</div>

<br>

An extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
that runs the [Anima Reference Adapter v2](https://huggingface.co/ElectronGirl/anima-ref-adapter). Give
[Anima](https://huggingface.co/circlestone-labs/Anima) one reference image and a prompt that describes only the
scene: the character (hair, eyes, outfit, accessories) and much of the drawing style come from the image,
the pose, background and composition from the prompt. A second reference gives two characters in one image.

Generated in Forge Neo with this extension (RTX 3060 Ti):

<img src="docs/examples.jpg" width="100%" alt="Forge Neo results">

The examples, measurements and method are in the
[ComfyUI version's README](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter); this extension runs
the same adapter with the same guidance.

> Development of the adapter is paused for lack of funding. To support further training:
> **[Ko-fi](https://ko-fi.com/electrongirl)**.

## Install

1. In Forge Neo, open **Extensions → Install from URL**, paste
   `https://github.com/ElectronGirl/sd-forge-anima-ref-adapter`, and press **Install**.
   Or clone it into `extensions/`:
   ```bash
   cd sd-webui-forge-neo/extensions
   git clone https://github.com/ElectronGirl/sd-forge-anima-ref-adapter.git
   ```
2. Restart Forge Neo.
3. Load Anima base v1.0 as the checkpoint, with `qwen_3_06b_base` and `qwen_image_vae` selected under
   **VAE / Text Encoder**, as for Anima without the extension.

The adapter weights (`anima_ref_adapter_v2_r64.safetensors`, 183 MB) download into your LoRA folder the
first time you generate with the extension on.

## Use

1. Open the **Anima Reference Adapter** panel in txt2img and tick it on.
2. Drop a reference image into **Reference**.
3. Write a prompt with quality tags, `1girl` or `1boy`, `solo` and the scene only:
   `masterpiece, best quality, score_7, safe, 1girl, solo, cowboy shot, sitting, cafe, holding cup, smile, looking at viewer, indoors, window, day`
   No character name, artist tags or outfit tags.
4. Generate. Anima's usual settings work: ER SDE or Euler, 30 steps, CFG 4.5.

**Two characters:** add the second character to **Second character (right, optional)**; it is placed to the
right of the first. Use `2girls` (or `1boy, 1girl`) instead of `solo`, say who is on the left and who is
on the right with a few words each, and use a landscape size such as 1216x832.

| setting | default | |
|---|---|---|
| Reference guidance | 4.5 | how hard to push toward the reference; 1 = no extra model call (faster, weaker) |
| Correction | `apg` | keeps colors in check at high guidance; or `rescale`, `none` |
| Reference guidance end | 1.0 | stop the extra guidance at this point of the schedule; ~0.6 is faster with nearly the same image |
| Reference size (MP) | 1.0 | the reference is resized to this area; lower it to save VRAM |

The settings are saved in the image's infotext (`anima_ref_*`).

**Turbo LoRA:** add a turbo LoRA for Anima base in the prompt (for example `<lora:anima-turbo-lora-v0.2:1>`)
and use 10 steps, CFG 1, reference guidance 1.

## Memory

The cached reference takes about 1 GB of VRAM at 1 MP, which the extension reserves from Forge's memory
manager. On an 8 GB GPU shared with other programs, start Forge with `--reserve-vram 2` if generation becomes
very slow (Forge warns about low free memory in the console), or lower the reference size.

## API

The extension is an always-on script, so it also works through `/sdapi/v1/txt2img`:

```json
"alwayson_scripts": {
  "anima reference adapter": {
    "args": [true, "<reference as base64>", "", "anima_ref_adapter_v2_r64.safetensors", 4.5, "apg", 1.0, 1.0]
  }
}
```

The arguments are: enabled, reference (base64), second reference (base64 or `""`), adapter file, reference
guidance, correction, reference guidance end, reference size in MP.

## Notes

- Tested with Forge Neo `neo-2.31` (PyTorch 2.14, CUDA 13.0) on an RTX 3060 Ti 8 GB, in txt2img. img2img
  and hires fix are untested.
- Only the v2 adapter is supported. The pilot adapter needs the ComfyUI nodes.
- The checkpoint must be Anima; with other models the panel does nothing.

## License

The extension code is **Apache-2.0** ([LICENSE](LICENSE)). The adapter weights are a derivative of the
CircleStone Model and are distributed on Hugging Face under the **CircleStone Non-Commercial License**.
This project is not affiliated with or endorsed by CircleStone Labs or the Forge Neo authors.
