import base64
import io
import logging

import gradio as gr
from PIL import Image

from backend import memory_management
from lib_anima_ref import adapter as ara
from modules import scripts, shared
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion


def _lora_dir() -> str:
    return shared.cmd_opts.lora_dir


def _image(value):
    """The UI passes a PIL image; the API (alwayson_scripts) passes a base64 string."""
    if isinstance(value, str):
        return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[-1]))) if value else None
    return value


class AnimaRefAdapterForForge(scripts.Script):
    sorting_priority = 15

    def __init__(self):
        self.cache = {}  # reference keys/values of the current sampling
        self.adapter = None

    def title(self):
        return "Anima Reference Adapter"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown("Character and style from the reference, pose and scene from the prompt. Write quality tags, "
                        "`1girl` / `1boy`, `solo` and scene tags only. For two characters, add a second reference "
                        "(placed on the right) and say in the prompt who is on the left and who is on the right.")
            with gr.Row():
                image = gr.Image(label="Reference", type="pil", height=320)
                image2 = gr.Image(label="Second character (right, optional)", type="pil", height=320)
            with gr.Row():
                lora = gr.Dropdown(label="Adapter", choices=ara.lora_names(_lora_dir()), value=ara.HF_FILES[0],
                                   info=f"the released file downloads from huggingface.co/{ara.HF_REPO} on first use")
                correction = gr.Radio(label="Correction", choices=["apg", "rescale", "none"], value="apg")
            with gr.Row():
                ref_guidance = gr.Slider(label="Reference guidance", minimum=0.0, maximum=10.0, step=0.1, value=4.5,
                                         info="1 = no extra model call (faster, weaker); higher = closer to the reference")
                ref_end = gr.Slider(label="Reference guidance end", minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                                    info="~0.6 is faster with nearly the same image")
                megapixels = gr.Slider(label="Reference size (MP)", minimum=0.25, maximum=2.0, step=0.05, value=1.0)

        self.infotext_fields = [
            PasteField(ref_guidance, "anima_ref_guidance"),
            PasteField(correction, "anima_ref_correction"),
            PasteField(ref_end, "anima_ref_end"),
        ]
        return [enable, image, image2, lora, ref_guidance, correction, ref_end, megapixels]

    def process_before_every_sampling(self, p, enable, image, image2, lora, ref_guidance, correction, ref_end, megapixels, **kwargs):
        image, image2 = _image(image), _image(image2)
        if not enable or image is None:
            return
        if type(p.sd_model).__name__ != "Anima":
            logging.warning("[Anima Ref Adapter] the loaded checkpoint is not Anima; skipped")
            return
        adapter = self.adapter = ara.Adapter.get(ara.lora_path(_lora_dir(), lora), 1.0)
        vae = p.sd_model.forge_objects.vae
        ref = vae.first_stage_model.process_in(vae.encode(ara.reference_image(image, image2, megapixels)))
        self.cache.clear()
        p.sd_model.forge_objects.unet = ara.patch(p.sd_model.forge_objects.unet, adapter, ref, ref_guidance, correction,
                                                  ref_end, self.cache)
        p.extra_generation_params.update({"anima_ref_adapter": lora, "anima_ref_guidance": ref_guidance,
                                          "anima_ref_correction": correction, "anima_ref_end": ref_end,
                                          "anima_ref_two": image2 is not None})

    def post_sample(self, p, ps, *args):
        if self.adapter is None:
            return
        self.cache.clear()  # ~1 GB of reference keys/values at 1 MP
        self.adapter.offload()
        self.adapter = None
        memory_management.soft_empty_cache()
