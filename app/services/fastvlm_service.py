from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import get_settings
from app.core.device import resolve_device


class FastVLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self.model_dir = Path(self.settings.fastvlm_local_dir)

    def load(self) -> None:
        self._device = resolve_device(self.settings.inference_device)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.settings.fastvlm_model_id,
            local_dir=str(self.model_dir),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True)
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self._device == "cuda" else torch.float32,
        }
        if self._device == "cuda":
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            **model_kwargs,
        )
        if self._device != "cuda":
            self._model = self._model.to("cpu")
        self._model.eval()

    def describe_image(self, image_path: Path) -> str:
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("FastVLM model not initialized")

        image = Image.open(image_path).convert("RGB")
        prompt = "Describe this image in one concise English sentence."
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        rendered = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        image_token = "<image>"
        if image_token not in rendered:
            raise RuntimeError("FastVLM chat template does not contain <image> placeholder.")
        pre, post = rendered.split(image_token, 1)

        pre_ids = self._tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self._tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        image_token_id = torch.tensor([[-200]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, image_token_id, post_ids], dim=1)

        model_device = next(self._model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = torch.ones_like(input_ids, device=model_device)

        pixel_values = self._model.get_vision_tower().image_processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(model_device, dtype=self._model.dtype)

        with torch.no_grad():
            outputs = self._model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                max_new_tokens=64,
            )

        generated = outputs[0][input_ids.shape[1]:]
        text = self._tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return text or "Failed to generate an image description."
