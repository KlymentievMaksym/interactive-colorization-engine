import torch
import kornia
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from colorization_engine.models.util_models import BaseColorizer
from colorization_engine.factory.registry import register_model

@register_model("controlnet_recolor")
class ControlNetRecolorWrapper(BaseColorizer):
    def __init__(self, inference_steps: int = 20, prompt: list[str] | str | None = None, negative_prompt: list[str] | None = None, device: str = "cuda"):
        super().__init__()

        self.inference_steps = inference_steps

        self.prompt = prompt if prompt is not None else ["high quality, photorealistic, vibrant colors"]
        if isinstance(self.prompt, str):
            self.prompt = list(self.prompt)
        if len(self.prompt) != 1:
            raise ValueError(f"Wrong promt list length, received {len(self.prompt)}, expected 1")
        if not isinstance(self.prompt, list):
            raise ValueError(f"Wrong promt type, received {type(self.prompt)}, expected str list[str] | str | None")

        self.negative_prompt = negative_prompt if negative_prompt is not None else ["monochrome, grayscale, bad quality, blurry"]
        if isinstance(self.negative_prompt, str):
            self.negative_prompt = list(self.negative_prompt)
        if len(self.negative_prompt) != 1:
            raise ValueError(f"Wrong promt list length, received {len(self.negative_prompt)}, expected 1")
        if not isinstance(self.negative_prompt, list):
            raise ValueError(f"Wrong negative promt type, received {type(self.negative_prompt)}, expected str list[str] | str | None")

        self.device = device

        # controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_recolor", torch_dtype=torch.float16, use_safetensors=True)
        controlnet = ControlNetModel.from_single_file("https://huggingface.co/lllyasviel/sd_control_collection/blob/main/ioclab_sd15_recolor.safetensors", torch_dtype=torch.float16)

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True).to(self.device)
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload() 
        
    def forward(self, l_norm: torch.Tensor, hints: torch.Tensor | None = None) -> torch.Tensor:
        """
        l_norm: [B, 1, H, W] in [-1, 1]
        hints: [B, 3, H, W] in [-1, 1] (ab channels + mask) або None
        """
        B, _, H, W = l_norm.shape
        l_unnorm = (l_norm + 1.0) * 50.0
        if hints is not None:
            ab_hints_unnorm = hints[:, :2, :, :] * 110.0
            mask = hints[:, 2:, :, :]

            ab_condition = torch.zeros((B, 2, H, W), device=self.device, dtype=l_norm.dtype)
            ab_condition = ab_condition * (1 - mask) + ab_hints_unnorm * mask

            lab_condition = torch.cat([l_unnorm, ab_condition], dim=1)
        else:
            ab_condition = torch.zeros((B, 2, H, W), device=self.device, dtype=l_norm.dtype)
            lab_condition = torch.cat([l_unnorm, ab_condition], dim=1)

        rgb_condition = kornia.color.lab_to_rgb(lab_condition)

        prompt = self.prompt * B
        negative_prompt = self.negative_prompt * B

        rgb_out = self.pipe(
            prompt,
            image=rgb_condition.half(), # Diffusers очікує float16
            num_inference_steps=self.inference_steps,
            negative_prompt=negative_prompt,
            output_type="pt" 
        ).images

        rgb_out_float32 = rgb_out.to(dtype=torch.float32, device=self.device)
    
        lab_out = kornia.color.rgb_to_lab(rgb_out_float32)

        ab_out_unnorm = lab_out[:, 1:, :, :]
        ab_out_norm = ab_out_unnorm / 110.0

        return ab_out_norm