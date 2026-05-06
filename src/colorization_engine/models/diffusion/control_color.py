import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import kornia

# ==============================================================================
# ZERO-MODIFICATION NAMESPACE INJECTION
# ==============================================================================
# Обчислюємо абсолютний шлях до кореня сабмодуля відносно поточного файлу.
# Це гарантує інваріантність коду до точки запуску (CWD).
_CURRENT_DIR = Path(__file__).resolve().parent
_SUBMODULE_ROOT = _CURRENT_DIR.parent / "util_models" / "ControlColor"

# Додаємо корінь репозиторію.
if str(_SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_ROOT))

# Якщо автори використовують жахливу структуру і імпортують внутрішні папки як глобальні
# (наприклад, taming або ldm), може знадобитися додати їх явно:
_TAMING_ROOT = _SUBMODULE_ROOT / "taming"
if _TAMING_ROOT.exists() and str(_TAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TAMING_ROOT))

# --- ПАТЧ ДЛЯ PYTORCH LIGHTNING 2.0+ ---
# Перехоплюємо застарілий імпорт LDM та перенаправляємо його на актуальний API
import pytorch_lightning
try:
    import pytorch_lightning.utilities.distributed
except ImportError:
    import pytorch_lightning.utilities.rank_zero
    # Підміняємо модуль у кеші інтерпретатора
    sys.modules['pytorch_lightning.utilities.distributed'] = sys.modules['pytorch_lightning.utilities.rank_zero']
# ---------------------------------------

# # Тепер імпорти стороннього коду відпрацюють без помилок ModuleNotFoundError
# from cldm.model import create_model, load_state_dict
# from cldm.ddim_haced_sag_step import DDIMSampler
# from ldm.models.autoencoder_train import AutoencoderKL
# ==============================================================================

from colorization_engine.models.util_models.ControlColor.cldm.model import create_model, load_state_dict
from colorization_engine.models.util_models.ControlColor.cldm.ddim_haced_sag_step import DDIMSampler
from colorization_engine.models.util_models.ControlColor.ldm.models.autoencoder_train import AutoencoderKL

from colorization_engine.models.util_models import BaseColorizer
from colorization_engine.factory.registry import register_model


@register_model("control_color")
class ControlColorWrapper(BaseColorizer):
    def __init__(
            self, 
            config_path: str = './models/cldm_v15_inpainting_infer1.yaml', 
            ckpt_path: str = './pretrained_models/main_model.ckpt',
            vae_path: Optional[str] = './pretrained_models/content-guided_deformable_vae.ckpt',
            base_resolution: int = 512,
            inference_steps: int = 20
        ):
        super().__init__()
        self.base_res = base_resolution
        self.inference_steps = inference_steps
        
        # 1. Завантаження моделі (ControlNet + SD 1.5)
        self.model = create_model(config_path)
        self.model.load_state_dict(load_state_dict(ckpt_path, location='cpu'), strict=False)
        self.sampler = DDIMSampler(self.model)

        # 2. Завантаження Deformable VAE (для усунення Color Bleeding)
        self.use_deformable_vae = vae_path is not None
        if self.use_deformable_vae:
            self.vae_model = self._build_vae(vae_path)
        else:
            self.vae_model = None

        # 3. Сувора VRAM оптимізація (відключення графів градієнтів)
        self.model.eval()
        self.model.requires_grad_(False)
        if self.use_deformable_vae:
            self.vae_model.eval()
            self.vae_model.requires_grad_(False)

        # 4. Фіксовані промпти для колоризації (замість повільного BLIP)
        self.pos_prompt = "high quality, detailed, photorealistic colorized image, best quality, real"
        self.neg_prompt = "black and white, bad anatomy, low quality, artifacts, watermark"

    def _build_vae(self, ckpt_path: str) -> AutoencoderKL:
        """Ініціалізація кастомного VAE авторів."""
        init_config = {
            "embed_dim": 4, "monitor": "val/rec_loss",
            "ddconfig": {"double_z": True, "z_channels": 4, "resolution": 256, 
                         "in_channels": 3, "out_ch": 3, "ch": 128, 
                         "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, 
                         "attn_resolutions": [], "dropout": 0.0},
            "lossconfig": {"target": "ldm.modules.losses.LPIPSWithDiscriminator",
                           "params": {"disc_start": 501, "kl_weight": 0, 
                                      "disc_weight": 0.025, "disc_factor": 1.0}}
        }
        vae = AutoencoderKL(**init_config)
        vae.load_state_dict(load_state_dict(ckpt_path, location='cpu'))
        return vae

    def _l_to_rgb(self, L_norm: torch.Tensor) -> torch.Tensor:
        """Перехід від L [-1, 1] до нейтрального sRGB [0, 1] (Grayscale)"""
        L = (L_norm + 1.0) * 50.0
        y = (L + 16.0) / 116.0
        mask = y > 0.2068966
        Y = torch.where(mask, y**3, (y - 16.0/116.0) / 7.787)
        mask_gamma = Y > 0.0031308
        RGB = torch.where(mask_gamma, 1.055 * (Y.clamp(min=1e-6)**(1/2.4)) - 0.055, 12.92 * Y)
        return RGB.clamp(0, 1).repeat(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, l_norm: torch.Tensor, hints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Головний цикл інференсу.
        l_norm: [B, 1, H, W] діапазон [-1, 1]
        hints: [B, 3, H, W] канали [a, b, mask], діапазон a,b: [-1, 1], mask: {0, 1}
        """
        B, _, H_orig, W_orig = l_norm.shape
        device = l_norm.device

        # 1. Просторове вирівнювання під архітектуру SD (512x512)
        l_resized = F.interpolate(l_norm, size=(self.base_res, self.base_res), mode='bilinear', align_corners=False)
        
        # 2. Формування базового Grayscale зображення (ControlNet condition)
        control_rgb = self._l_to_rgb(l_resized) # [B, 3, 512, 512] в [0, 1]
        
        # 3. Підготовка масок та підказок
        if hints is not None:
            hints_resized = F.interpolate(hints, size=(self.base_res, self.base_res), mode='nearest')
            hint_ab = hints_resized[:, 0:2, :, :]
            hint_mask = hints_resized[:, 2:3, :, :].clamp(0, 1) # 1 - є підказка
            
            # --- ВПРОВАДЖЕННЯ ПІДКАЗОК (LAB -> RGB) ---
            # Відновлюємо масштаб каналів для простору CIELAB
            L_kornia = (l_resized + 1.0) * 50.0
            ab_kornia = hint_ab * 110.0
            lab_hints = torch.cat([L_kornia, ab_kornia], dim=1)
            
            # Конвертуємо підказки в RGB [0, 1] для подачі у VAE
            rgb_hints = kornia.color.lab_to_rgb(lab_hints).clamp(0.0, 1.0)
            # ------------------------------------------
            
            # Маска для SD: 1 = генерувати (невідомо), 0 = зберегти (відома підказка)
            sd_mask = 1.0 - hint_mask 
            
            # Зображення з підказками для VAE (невідомі зони = 0)
            masked_image = rgb_hints * hint_mask
        else:
            sd_mask = torch.ones((B, 1, self.base_res, self.base_res), device=device)
            masked_image = control_rgb.clone() * 0.0 # Немає підказок

        # 4. Латентне кодування
        # SD VAE очікує вхід у діапазоні [-1, 1]
        masked_image_vae = (masked_image * 2.0) - 1.0 
        masked_image_latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(masked_image_vae)).detach()
        
        # Даунсемплінг маски для латентного простору
        mask_latent = F.interpolate(sd_mask, size=(self.base_res // 8, self.base_res // 8), mode='nearest')

        # 5. Текстове кондиціонування (Cross-Attention)
        cond_text = self.model.get_learned_conditioning([self.pos_prompt] * B)
        uncond_text = self.model.get_learned_conditioning([self.neg_prompt] * B)
        
        cond = {"c_concat": [control_rgb], "c_crossattn": [cond_text]}
        un_cond = {"c_concat": [control_rgb], "c_crossattn": [uncond_text]}

        # 6. Марковський процес семплювання (генерація)
        shape = (4, self.base_res // 8, self.base_res // 8)
        # Seed для багатоваріантності
        noise = torch.randn((B, *shape), device=device)
        # # --- ЗАБЕЗПЕЧЕННЯ БАГАТОВАРІАНТНОСТІ ---
        # import os
        # # Беремо 8 байт (64 біти) істинної системної ентропії для уникнення глобального seed
        # local_seed = int.from_bytes(os.urandom(8), "big") & 0xFFFFFFFFFFFFFFFF
        # # Створюємо ізольований генератор на GPU
        # gen = torch.Generator(device=device).manual_seed(local_seed)
        # noise = torch.randn((B, *shape), generator=gen, device=device)
        # # ---------------------------------------
        self.model.control_scales = [1.0] * 13 

        samples, _ = self.sampler.sample(
            model=self.model,
            S=self.inference_steps,
            batch_size=B,
            shape=shape,
            conditioning=cond,
            mask=mask_latent,
            masked_image_latents=masked_image_latents,
            unconditional_guidance_scale=7.0,
            unconditional_conditioning=un_cond,
            sag_scale=0.05,
            SAG_influence_step=600,
            noise=noise,
            eta=0.0,
            verbose=False
        )

        # 7. Декодування результату (Latent -> RGB)
        if self.use_deformable_vae:
            samples_before_vae = self.model.decode_first_stage_before_vae(samples)
            
            # --- ВИПРАВЛЕННЯ АРХІТЕКТУРНОГО БАГУ АВТОРІВ ---
            # Витягуємо перше зображення з батчу і перетворюємо [C, H, W] -> [H, W, C]
            # Це імітує поведінку numpy-масиву, під який писався Deformable VAE
            gray_input_for_vae = control_rgb[0].permute(1, 2, 0)
            
            x_samples_rgb_list = []
            for i in range(B):
                gray_input_for_vae = control_rgb[i].permute(1, 2, 0) # [H, W, C]
                gray_content_z = self.vae_model.get_gray_content_z(gray_input_for_vae)
                
                # Decode just this single item (keep batch dim of 1)
                single_sample = samples_before_vae[i:i+1] 
                decoded_rgb = self.vae_model.decode(single_sample, gray_content_z)
                x_samples_rgb_list.append(decoded_rgb)
                
            x_samples_rgb = torch.cat(x_samples_rgb_list, dim=0)
            # gray_content_z = self.vae_model.get_gray_content_z(gray_input_for_vae)
            # x_samples_rgb = self.vae_model.decode(samples_before_vae, gray_content_z)
            # -----------------------------------------------
        else:
            x_samples_rgb = self.model.decode_first_stage(samples)

        # 8. Відновлення розмірності та перехід в LAB
        x_samples_rgb = (x_samples_rgb + 1.0) * 0.5 # [-1, 1] -> [0, 1]
        x_samples_rgb = F.interpolate(x_samples_rgb, size=(H_orig, W_orig), mode='bicubic', align_corners=False)
        
        # Спрощена екстракція (в дипломній використовуйте kornia.color.rgb_to_lab(x_samples_rgb))
        # Припускаємо наявність методу _rgb_to_ab у вашому базовому класі
        ab_pred = self._rgb_to_ab(x_samples_rgb) # Має повертати [B, 2, H, W] в [-1, 1]

        return ab_pred

    def _rgb_to_ab(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Конвертує RGB тензор у діапазоні [0, 1] у нормалізовані канали AB [-1, 1].
        Операція виконується повністю на GPU зі збереженням графів Autograd.
        
        Args:
            rgb_tensor (torch.Tensor): Тензор зображення [B, 3, H, W] у діапазоні [0.0, 1.0].
            
        Returns:
            torch.Tensor: Тензор каналів a та b [B, 2, H, W] у діапазоні [-1.0, 1.0].
        """
        # 1. Трансформація RGB -> LAB (Kornia очікує RGB у діапазоні [0, 1])
        # Результат: L \in [0, 100], a, b \approx [-110, 110] (залежить від ілюмінанту D65)
        lab_tensor = kornia.color.rgb_to_lab(rgb_tensor)
        
        # 2. Екстракція хроматичних каналів a та b (індекси 1 та 2 за віссю каналів)
        ab_unnorm = lab_tensor[:, 1:3, :, :]
        
        # 3. Симетрична нормалізація у діапазон [-1, 1]
        # Використовуємо константу 110.0 для інваріантності з вашим kornia_lab_to_rgb
        ab_norm = ab_unnorm / 110.0
        
        # 4. Запобігання артефактам переповнення
        # Теоретичні межі a,b можуть незначно виходити за межі \pm 110 через FP похибки
        # або екстремальні вхідні значення RGB. Жорстке обмеження стабілізує подальші операції.
        ab_norm = torch.clamp(ab_norm, min=-1.0, max=1.0)
        
        return ab_norm