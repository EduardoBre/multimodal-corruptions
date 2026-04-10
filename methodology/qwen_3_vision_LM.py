import torch
import numpy as np
import random
import time
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

class Qwen3VLInstance:

    def __init__(self, seed: int, max_new_tokens: int = 1024, dvc: str = "gpu"):
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if dvc.lower() == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Loading model on {self.device}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            dtype=torch.bfloat16,
            device_map=None if self.device.type == "cpu" else "cuda",
            attn_implementation="flash_attention_2"
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    def run_inference(self, image, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        start_time = time.time()
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
        )
        runtime = time.time() - start_time

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Total generated tokens (includes <think> block if present)
        raw_token_count = len(generated_ids_trimmed[0])

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Visible output tokens only (excludes thinking tokens)
        token_count = len(self.processor.tokenizer.encode(
            output_text[0], add_special_tokens=False
        ))

        return output_text[0], token_count, raw_token_count, runtime

    def run_batch_inference(self, images, prompts):
        self.processor.tokenizer.padding_side = "left"

        messages = []
        for image, prompt in zip(images, prompts):
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
            }])

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(self.device)

        start_time = time.time()
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
        )
        runtime = time.time() - start_time

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        raw_token_counts = [len(ids) for ids in generated_ids_trimmed]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        token_counts = [
            len(self.processor.tokenizer.encode(t, add_special_tokens=False))
            for t in output_texts
        ]

        return output_texts, token_counts, raw_token_counts, runtime
