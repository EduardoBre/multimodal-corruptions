import os
import json
import re
import numpy as np
import random
import torch
from tqdm import tqdm

from qwen_3_embedding import Qwen3EmbeddingInstance

# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

SEED = 42669
OUTPUT_DIR = "unimodal/text_corruptions"


def extract_object_labels(prompt):
    """
    Extract individual object labels from between objects "..." in a prompt.
    Falls back to [prompt] if the pattern is not found.
    """
    if not prompt:
        return [prompt] if prompt else []
    match = re.search(r'objects "(.*?)"', prompt)
    if match:
        content = match.group(1)
        return [label.strip() for label in content.split(',')]
    return [prompt]


def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0.0:
        return 0.0
    return float(dot / denom)


class CosineSimilarityPatcher:
    def __init__(self, output_dir, seed):
        self.output_dir = output_dir
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.embedder = None
        self._emb_cache = {}

    def _ensure_embedder(self):
        """Load the embedding model on first use."""
        if self.embedder is None:
            self.embedder = Qwen3EmbeddingInstance(seed=self.seed)

    def _embed_single(self, text):
        """Embed a single label string, cached to avoid re-embedding."""
        if text in self._emb_cache:
            return self._emb_cache[text]

        vec, _, _ = self.embedder.run_inference(text)
        self._emb_cache[text] = vec
        return vec

    def collect_json_files(self):
        json_files = []
        for root, _, files in os.walk(self.output_dir):
            for fname in files:
                if fname.startswith("scale_") and fname.endswith("_TEXT.json"):
                    json_files.append(os.path.join(root, fname))
        json_files.sort()
        return json_files

    def patch_file(self, json_path):
        """
        For every corruption entry missing 'cosine_similarity', compute
        per-label mean cosine similarity and write the file back.
        Returns True if the file was modified.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        original_prompt = data.get("original_prompt", "")
        corruptions = data.get("corruptions", {})

        modified = False

        for c_name, c_data in corruptions.items():
            if "cosine_similarity" in c_data:
                continue

            orig_labels = extract_object_labels(original_prompt)
            corr_labels = extract_object_labels(c_data.get("prompt", ""))

            pair_sims = []
            for orig_lbl, corr_lbl in zip(orig_labels, corr_labels):
                orig_emb = self._embed_single(orig_lbl)
                corr_emb = self._embed_single(corr_lbl)
                pair_sims.append(cosine_similarity(orig_emb, corr_emb))

            cos_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

            c_data["cosine_similarity"] = float(f"{cos_sim:.6f}")
            modified = True

        if modified:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)

        return modified

    def run(self):
        json_files = self.collect_json_files()
        print(f"Found {len(json_files)} JSON files in {self.output_dir}")

        files_to_patch = []
        for json_path in json_files:
            with open(json_path, 'r') as f:
                data = json.load(f)

            corruptions = data.get("corruptions", {})
            for c_data in corruptions.values():
                if "cosine_similarity" not in c_data:
                    files_to_patch.append(json_path)
                    break

        if not files_to_patch:
            print("All files already have cosine_similarity. Nothing to patch.")
            return

        print(f"{len(files_to_patch)} files need patching.")

        self._ensure_embedder()

        patched_count = 0
        for json_path in tqdm(files_to_patch, desc="Patching cosine_similarity"):
            was_modified = self.patch_file(json_path)
            if was_modified:
                patched_count += 1

        print(f"\nDone. Patched {patched_count} files.")
        print(f"Embedding cache: {len(self._emb_cache)} unique labels embedded.")
        print(f"Results in: {os.path.abspath(self.output_dir)}")


if __name__ == "__main__":
    patcher = CosineSimilarityPatcher(
        output_dir=OUTPUT_DIR,
        seed=SEED,
    )
    patcher.run()
