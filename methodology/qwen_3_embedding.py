import torch
import numpy as np
import random
import time
from transformers import AutoModel, AutoTokenizer

# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

class Qwen3EmbeddingInstance:

    def __init__(self, seed: int, max_length: int = 8192, dvc: str = "gpu"):
        self.max_length = max_length
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if dvc.lower() == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Loading model on {self.device}...")

        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-4B",
            dtype=torch.bfloat16,
            device_map=None if self.device.type == "cpu" else "cuda",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-4B",
            trust_remote_code=True
        )

        # Decoder-only model: left padding makes last-token pooling straightforward
        self.tokenizer.padding_side = "left"

    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Extracts the hidden state of the last non-padding token per sequence."""
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]

        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

    def run_inference(self, text: str, instruction: str = None):
        """Generates an L2-normalised embedding (dim 2560) for a single text input."""
        full_text = f"{instruction}{text}" if instruction else text

        inputs = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        token_count = inputs.input_ids.shape[1]

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._last_token_pool(outputs.last_hidden_state, inputs.attention_mask)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        runtime = time.time() - start_time

        embedding_vector = embeddings[0].float().cpu().numpy()

        return embedding_vector, token_count, runtime

    def run_batch_inference(self, texts, instructions=None):
        """Generates L2-normalised embeddings for a batch of text inputs."""
        if instructions is None:
            full_texts = texts
        elif isinstance(instructions, str):
            full_texts = [f"{instructions}{text}" for text in texts]
        elif isinstance(instructions, list):
            if len(instructions) != len(texts):
                raise ValueError("Instructions list length must match texts list length.")
            full_texts = [f"{inst}{text}" for inst, text in zip(instructions, texts)]
        else:
            full_texts = texts

        inputs = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        token_counts = [len(ids) for ids in inputs.input_ids]

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._last_token_pool(outputs.last_hidden_state, inputs.attention_mask)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        runtime = time.time() - start_time

        embedding_vectors = [emb.float().cpu().numpy() for emb in embeddings]

        return embedding_vectors, token_counts, runtime
