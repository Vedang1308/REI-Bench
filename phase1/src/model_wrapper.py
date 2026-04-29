"""
Model loading and inference wrapper.

Supports:
  - NVIDIA A100 GPU via HuggingFace Transformers
  - Intel Gaudi HPU via optimum-habana
  - CPU fallback
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .device_utils import detect_device

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Unified wrapper for LLM inference across CUDA, HPU, and CPU.

    Usage:
        wrapper = ModelWrapper("meta-llama/Llama-3.2-1B-Instruct", hf_token="...")
        output = wrapper.generate("What is 2+2?", max_new_tokens=64)
    """

    def __init__(self, model_name: str, hf_token: str = None):
        """
        Load the model and tokenizer on the detected device.

        Args:
            model_name: HuggingFace model identifier.
            hf_token: HuggingFace access token for gated models.
        """
        self.model_name = model_name
        self.device_type = detect_device()
        self.hf_token = hf_token

        logger.info(f"Loading model: {model_name} on {self.device_type}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model based on device type
        if self.device_type == "hpu":
            self._load_hpu(model_name, hf_token)
        elif self.device_type == "cuda":
            self._load_cuda(model_name, hf_token)
        else:
            self._load_cpu(model_name, hf_token)

        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device_type}")

    def _load_cuda(self, model_name: str, hf_token: str):
        """Load model for NVIDIA GPU with bfloat16."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.device = torch.device("cuda:0")

    def _load_hpu(self, model_name: str, hf_token: str):
        """Load model for Intel Gaudi HPU."""
        try:
            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )
            adapt_transformers_to_gaudi()
        except ImportError:
            logger.warning(
                "optimum-habana not found, attempting standard load on HPU"
            )

        import habana_frameworks.torch.core as htcore  # noqa: F401

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.device = torch.device("hpu")
        self.model = self.model.to(self.device)

    def _load_cpu(self, model_name: str, hf_token: str):
        """Load model on CPU with float32."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.device = torch.device("cpu")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            do_sample: Whether to use sampling.

        Returns:
            Generated text (excluding the input prompt).
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if do_sample and temperature > 0:
                gen_kwargs["temperature"] = temperature

            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated part (exclude input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        # Sync HPU if needed
        if self.device_type == "hpu":
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

        return generated_text.strip()

    def generate_with_chat_template(
        self,
        messages: list[dict],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate using the model's chat template.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated assistant response.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    def constrained_generate(
        self,
        prompt: str,
        valid_options: list[str],
        max_new_tokens: int = 32,
    ) -> str:
        """
        Generate text constrained to select from valid options.

        This implements the SayCan-style constrained decoding where the model
        must pick from a predefined skill set. We generate freely and then
        find the closest match among valid options.

        Args:
            prompt: Input prompt.
            valid_options: List of valid output strings.
            max_new_tokens: Maximum tokens for generation.

        Returns:
            The best matching option from valid_options.
        """
        generated = self.generate(prompt, max_new_tokens=max_new_tokens)
        generated_lower = generated.lower().strip()

        # Try exact match first
        for option in valid_options:
            if option.lower() in generated_lower:
                return option

        # Try partial matching - find the option with most word overlap
        best_option = valid_options[0]
        best_score = 0
        gen_words = set(generated_lower.split())

        for option in valid_options:
            option_words = set(option.lower().split())
            overlap = len(gen_words & option_words)
            if overlap > best_score:
                best_score = overlap
                best_option = option

        return best_option

    @property
    def short_name(self) -> str:
        """Short display name for the model."""
        return self.model_name.split("/")[-1]
