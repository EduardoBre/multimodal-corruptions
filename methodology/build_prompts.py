"""
# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

Build ready-to-send LLM prompts by combining a prompt template
with chunked keys from label_mapping.json.

Usage:
    python build_prompts.py

Outputs:
    prompts/homophone/chunk_01.txt ... chunk_NN.txt
    prompts/synonym/chunk_01.txt   ... chunk_NN.txt
"""

import json
import os
import math

LABEL_MAPPING_PATH = "label_mapping.json"
OUTPUT_DIR = "prompts"
CHUNK_SIZE = 20

PROMPTS = {
    "homophone_prompt.txt": "homophone",
    "synonym_prompt.txt":   "synonym",
}


def chunk_dict(d, size):
    """Split a dict into a list of smaller dicts of at most `size` keys."""
    items = list(d.items())
    return [dict(items[i:i + size]) for i in range(0, len(items), size)]


def main():
    with open(LABEL_MAPPING_PATH, 'r') as f:
        label_mapping = json.load(f)

    n_keys = len(label_mapping)
    n_chunks = math.ceil(n_keys / CHUNK_SIZE)
    print(f"Loaded {n_keys} labels -> {n_chunks} chunks of {CHUNK_SIZE} keys each\n")

    chunks = chunk_dict(label_mapping, CHUNK_SIZE)

    for template_name, output_subdir in PROMPTS.items():
        with open(template_name, 'r') as f:
            template = f.read()

        out_dir = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(out_dir, exist_ok=True)

        for idx, chunk in enumerate(chunks, start=1):
            chunk_json = json.dumps(chunk, indent=4)
            final_prompt = f"{template}\n{chunk_json}"

            filename = f"chunk_{idx:02d}.txt"
            filepath = os.path.join(out_dir, filename)

            with open(filepath, 'w') as f:
                f.write(final_prompt)

        keys_per_chunk = [len(c) for c in chunks]
        chunk_summary = ", ".join(str(k) for k in keys_per_chunk)
        print(f"  {output_subdir}/")
        print(f"    {n_chunks} files written  (keys per chunk: {chunk_summary})")

    print(f"\nDone. All prompts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
