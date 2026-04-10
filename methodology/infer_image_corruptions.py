import os
import json
import numpy as np
import random
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from image_perturbator import ImagePerturbator
from qwen_3_vision_LM import Qwen3VLInstance


# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

SEED = 42669
BATCH_SIZE = 10
SELECTION_DIR = "results/selection"
OUTPUT_DIR = "unimodal/image_corruptions"
QWEN_SCALE_FACTOR = 1000
MAX_RESOLUTION = 1024

SCALE_MAP = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1}

SELECTED_CORRUPTIONS = [
    "jpeg_filter",
    "pixelate",
    "defocus_blur",
    "motion_blur",
    "gaussian_noise",
    "fog_filter",
    "snow_filter",
    "contrast",
    "elastic",
    "cutout",
    "false_color",
    "grayscale",
]


def resize_image_smart(img, max_side=1080):
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def ensure_rgb(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def scale_pred_box(pred_box, target_w, target_h):
    return [
        pred_box[0] * target_w / QWEN_SCALE_FACTOR,
        pred_box[1] * target_h / QWEN_SCALE_FACTOR,
        pred_box[2] * target_w / QWEN_SCALE_FACTOR,
        pred_box[3] * target_h / QWEN_SCALE_FACTOR,
    ]


def extract_json_array(pred_str):
    text = pred_str.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def calculate_iou_score(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / float(denom)


def compute_mean_iou(gt_dict, pred_list, ref_w, ref_h):
    if not gt_dict:
        return 0.0
    gt_items = []
    for key, box in gt_dict.items():
        label = key.split('_')[0]
        box_coord = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
        gt_items.append((label, box_coord))

    ious = []
    for gt_label, gt_box in gt_items:
        best_iou_for_this_obj = 0.0
        relevant_preds = [
            p for p in pred_list
            if isinstance(p, dict) and p.get('label') == gt_label and 'bbox_2d' in p
        ]
        for p in relevant_preds:
            scaled_box = scale_pred_box(p['bbox_2d'], ref_w, ref_h)
            curr_iou = calculate_iou_score(gt_box, scaled_box)
            if curr_iou > best_iou_for_this_obj:
                best_iou_for_this_obj = curr_iou
        ious.append(best_iou_for_this_obj)

    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def get_scaled_gt_boxes(gt_dict, orig_w, orig_h, new_w, new_h):
    if not gt_dict:
        return []
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    box_list = []
    for key, val in gt_dict.items():
        b = [
            int(val['xmin'] * scale_x),
            int(val['ymin'] * scale_y),
            int(val['xmax'] * scale_x),
            int(val['ymax'] * scale_y)
        ]
        box_list.append(b)
    return box_list


def normalised_frobenius(clean_np, corrupt_np):
    """
    Compute the normalised Frobenius distance between a clean and
    corrupted image (both HWC uint8 numpy arrays of the same shape).
    """
    clean = clean_np.astype(np.float64) / 255.0
    corrupt = corrupt_np.astype(np.float64) / 255.0
    diff = clean - corrupt
    c, h, w = clean.shape[2], clean.shape[0], clean.shape[1]
    return float(np.linalg.norm(diff) / np.sqrt(c * h * w))


def visualize_result(image_array, output_path, gt_dict, pred_list, orig_size=None):
    try:
        if isinstance(image_array, np.ndarray):
            img = Image.fromarray(image_array.astype(np.uint8))
        else:
            img = image_array.convert("RGB")
        draw = ImageDraw.Draw(img)
        vis_w, vis_h = img.size
        scale_x, scale_y = 1.0, 1.0
        if orig_size:
            orig_w, orig_h = orig_size
            scale_x = vis_w / orig_w
            scale_y = vis_h / orig_h

        for label, box in gt_dict.items():
            shape = [
                box['xmin'] * scale_x,
                box['ymin'] * scale_y,
                box['xmax'] * scale_x,
                box['ymax'] * scale_y
            ]
            draw.rectangle(shape, outline="lime", width=3)

        for p in pred_list:
            if 'bbox_2d' in p and isinstance(p['bbox_2d'], list):
                scaled_box = scale_pred_box(p['bbox_2d'], vis_w, vis_h)
                draw.rectangle(scaled_box, outline="red", width=3)
        img.save(output_path)
    except Exception as e:
        print(f"Error saving visualization to {output_path}: {e}")


class PerturbationEvaluator:
    def __init__(self, selection_dir, output_dir, seed):
        self.selection_dir = selection_dir
        self.output_dir = output_dir
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.perturbator = ImagePerturbator()
        self.qwen = Qwen3VLInstance(seed=seed, max_new_tokens=800)

    def get_directories(self):
        """
        Traverse the selection folder structure:
          selection_dir/single/solo/1..250/
          selection_dir/single/multi/1..250/
          selection_dir/multi/1..500/
        Returns list of (source_folder, relative_path) tuples.
        """
        dirs_to_process = []

        categories = [
            os.path.join("single", "solo"),
            os.path.join("single", "multi"),
            "multi",
        ]

        for cat_rel in categories:
            cat_abs = os.path.join(self.selection_dir, cat_rel)
            if not os.path.exists(cat_abs):
                continue

            numbered_dirs = []
            for folder_name in os.listdir(cat_abs):
                folder_path = os.path.join(cat_abs, folder_name)
                if os.path.isdir(folder_path) and folder_name.isdigit():
                    rel_path = os.path.join(cat_rel, folder_name)
                    numbered_dirs.append((folder_path, rel_path))

            numbered_dirs.sort(key=lambda x: int(os.path.basename(x[0])))
            dirs_to_process.extend(numbered_dirs)

        return dirs_to_process

    def load_folder_data(self, src_folder, rel_path, scale_idx):
        """
        Loads data for a single selection folder.
        Returns "SKIPPED" if the output JSON for this scale already exists.
        """
        input_json_path = os.path.join(src_folder, "original.json")
        input_img_path = os.path.join(src_folder, "data_point.JPEG")

        out_folder = os.path.join(self.output_dir, rel_path)
        output_json_path = os.path.join(out_folder, f"scale_{scale_idx}_IMG.json")

        if os.path.exists(output_json_path):
            return "SKIPPED", None

        if not os.path.exists(input_json_path) or not os.path.exists(input_img_path):
            return "MISSING", None

        try:
            with open(input_json_path, 'r') as f:
                base_data = json.load(f)

            clean_image_raw = Image.open(input_img_path).convert("RGB")
            orig_width, orig_height = clean_image_raw.size
            clean_image = resize_image_smart(clean_image_raw, MAX_RESOLUTION)
            current_width, current_height = clean_image.size
            clean_img_array = np.array(clean_image)

            data_item = {
                "src_folder": src_folder,
                "rel_path": rel_path,
                "prompt": base_data.get("prompt"),
                "gt_bboxes": base_data.get("ground_truth", {}),
                "filename": base_data.get("image"),
                "orig_dims": (orig_width, orig_height),
                "curr_dims": (current_width, current_height),
                "clean_img_array": clean_img_array,
                "results_map": {},
            }
            return "LOADED", data_item

        except Exception as e:
            print(f"Error loading {src_folder}: {e}")
            return "ERROR", None

    def process_batch_for_scale(self, batch_data_items, scale_idx):
        """
        Processes a batch of images across all 12 corruptions at a single scale.
        For each corruption: apply perturbation, run batch inference,
        compute IoU + Frobenius distance. Saves one JSON per image.
        """
        if not batch_data_items:
            return

        scale_float = SCALE_MAP[scale_idx]

        for c_name in SELECTED_CORRUPTIONS:
            inference_imgs = []
            inference_prompts = []
            valid_indices = []
            corrupted_arrays = []

            for idx, item in enumerate(batch_data_items):
                try:
                    input_array = ensure_rgb(item["clean_img_array"]).copy()

                    if c_name == "cutout":
                        scaled_gt_list = get_scaled_gt_boxes(
                            item["gt_bboxes"],
                            item["orig_dims"][0], item["orig_dims"][1],
                            item["curr_dims"][0], item["curr_dims"][1],
                        )
                        perturbed_array = self.perturbator.apply_perturbation(
                            input_array, c_name, scale=scale_float, bboxes=scaled_gt_list
                        )
                    else:
                        perturbed_array = self.perturbator.apply_perturbation(
                            input_array, c_name, scale=scale_float
                        )

                    perturbed_pil = Image.fromarray(perturbed_array.astype(np.uint8))

                    inference_imgs.append(perturbed_pil)
                    inference_prompts.append(item["prompt"])
                    corrupted_arrays.append(perturbed_array)
                    valid_indices.append(idx)

                except Exception as e:
                    print(f"Error applying {c_name} (scale {scale_idx}) to {item['rel_path']}: {e}")
                    continue

            if not inference_imgs:
                continue

            if len(inference_imgs) == 1:
                text, tc, raw_tc, rt = self.qwen.run_inference(inference_imgs[0], inference_prompts[0])
                responses = [text]
                token_counts = [tc]
                raw_token_counts = [raw_tc]
                batch_runtime = rt
            else:
                responses, token_counts, raw_token_counts, batch_runtime = self.qwen.run_batch_inference(
                    inference_imgs, inference_prompts
                )

            avg_runtime_per_img = batch_runtime / len(inference_imgs)

            for i, response_str in enumerate(responses):
                original_idx = valid_indices[i]
                item = batch_data_items[original_idx]
                corr_array = corrupted_arrays[i]

                parsed_preds = extract_json_array(response_str)
                iou_score = compute_mean_iou(
                    item["gt_bboxes"], parsed_preds,
                    item["orig_dims"][0], item["orig_dims"][1],
                )

                frob_dist = normalised_frobenius(
                    ensure_rgb(item["clean_img_array"]),
                    ensure_rgb(corr_array),
                )

                item["results_map"][c_name] = {
                    "result": response_str,
                    "IoU": float(f"{iou_score:.5f}"),
                    "token_count": token_counts[i],
                    "raw_token_count": raw_token_counts[i],
                    "runtime_seconds": float(f"{avg_runtime_per_img:.4f}"),
                    "frobenius_distance": float(f"{frob_dist:.6f}"),
                }

                out_folder = os.path.join(self.output_dir, item["rel_path"])
                os.makedirs(out_folder, exist_ok=True)
                viz_filename = (
                    f"{os.path.splitext(item['filename'])[0]}"
                    f"_{c_name}_{scale_idx}_IMG.jpg"
                )
                viz_path = os.path.join(out_folder, viz_filename)
                visualize_result(
                    corr_array, viz_path,
                    item["gt_bboxes"], parsed_preds,
                    orig_size=item["orig_dims"],
                )

        for item in batch_data_items:
            if not item["results_map"]:
                continue

            out_folder = os.path.join(self.output_dir, item["rel_path"])
            os.makedirs(out_folder, exist_ok=True)
            output_json_path = os.path.join(out_folder, f"scale_{scale_idx}_IMG.json")

            final_output = {
                "prompt": item["prompt"],
                "filename": item["filename"],
                "corruption_scale": scale_idx,
                "corruption_scale_float": SCALE_MAP[scale_idx],
                "original_dims": list(item["orig_dims"]),
                "inference_dims": list(item["curr_dims"]),
                "corruptions": item["results_map"],
                "ground_truth_bboxes": item["gt_bboxes"],
                "seed": self.seed,
            }

            with open(output_json_path, 'w') as f:
                json.dump(final_output, f, indent=4)

    def run(self):
        all_dirs = self.get_directories()
        print(f"Found {len(all_dirs)} data-point directories in selection.")

        for scale_idx in sorted(SCALE_MAP.keys()):
            scale_float = SCALE_MAP[scale_idx]
            print(f"\nScale {scale_idx} (float: {scale_float})")

            unprocessed = []
            skipped = 0
            for src_folder, rel_path in all_dirs:
                out_json = os.path.join(self.output_dir, rel_path, f"scale_{scale_idx}_IMG.json")
                if os.path.exists(out_json):
                    skipped += 1
                else:
                    unprocessed.append((src_folder, rel_path))

            print(f"  Skipping {skipped} already processed. {len(unprocessed)} remaining.")

            if not unprocessed:
                continue

            batches = [
                unprocessed[i:i + BATCH_SIZE]
                for i in range(0, len(unprocessed), BATCH_SIZE)
            ]

            pbar = tqdm(batches, desc=f"Scale {scale_idx}")
            processed = 0

            for batch_folders in pbar:
                batch_data_items = []
                for src_folder, rel_path in batch_folders:
                    status, item = self.load_folder_data(src_folder, rel_path, scale_idx)
                    if status == "LOADED":
                        batch_data_items.append(item)

                if batch_data_items:
                    self.process_batch_for_scale(batch_data_items, scale_idx)
                    processed += len(batch_data_items)

                pbar.set_postfix(processed=processed)

        print(f"\nDone. Results saved in: {os.path.abspath(self.output_dir)}")


if __name__ == "__main__":
    evaluator = PerturbationEvaluator(
        selection_dir=SELECTION_DIR,
        output_dir=OUTPUT_DIR,
        seed=SEED,
    )
    evaluator.run()
