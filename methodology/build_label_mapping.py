import os
import json
import re

# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

RESULTS_BASE_DIR = "results/selection"
OUTPUT_FILE = "label_mapping.json"

CATEGORIES = ["single/solo", "single/multi", "multi"]


def extract_base_label(key):
    """Strip trailing '_N' suffixes used for duplicate instances."""
    return re.sub(r'_\d+$', '', key)


def collect_labels():
    labels = set()

    for category in CATEGORIES:
        category_dir = os.path.join(RESULTS_BASE_DIR, category)
        if not os.path.isdir(category_dir):
            print(f"Warning: category directory not found: {category_dir}")
            continue

        for folder_name in sorted(os.listdir(category_dir)):
            json_path = os.path.join(category_dir, folder_name, "original.json")
            if not os.path.isfile(json_path):
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            gt = data.get("ground_truth", {})
            for key in gt:
                labels.add(extract_base_label(key))

    return labels


def main():
    labels = collect_labels()
    print(f"Found {len(labels)} unique class labels.")

    mapping = {label: ["empty"] for label in sorted(labels)}

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(mapping, f, indent=4)

    print(f"Saved mapping to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
