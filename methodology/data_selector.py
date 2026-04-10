import os
import random
import json
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil
import numpy as np
import torch
from qwen_3_vision_LM import Qwen3VLInstance

# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

IOU_THRESHOLD = 0.5
QWEN_SCALE = 1000
GROUP_SIZE = 500
BATCH_SIZE = 25
RESULTS_BASE_DIR = "results/selection"
SEED = 42669
MAX_RESOLUTION = 1024


def load_synset_to_label(mat_file_path):
    import scipy.io
    meta = scipy.io.loadmat(mat_file_path)
    synsets = meta['synsets']
    synset_to_label = {}
    for entry in synsets[0]:
        synset = entry[1][0]
        label = entry[2][0]
        synset_to_label[synset] = label
    return synset_to_label


def parse_annotation(xml_file, synset_to_label):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ground_truth = {}
    unique_labels = set()

    objects = root.findall('object')
    instance_count = len(objects)

    for obj in objects:
        synset = obj.find('name').text
        label = synset_to_label.get(synset, synset)
        unique_labels.add(label)

        bndbox = obj.find('bndbox')
        box = {
            "xmin": int(bndbox.find('xmin').text),
            "ymin": int(bndbox.find('ymin').text),
            "xmax": int(bndbox.find('xmax').text),
            "ymax": int(bndbox.find('ymax').text),
        }

        key = label
        if key in ground_truth:
            suffix = 1
            while f"{label}_{suffix}" in ground_truth:
                suffix += 1
            key = f"{label}_{suffix}"
        ground_truth[key] = box

    return ground_truth, unique_labels, instance_count


def calculate_iou(boxA, boxB):
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


def scale_pred_box(pred_box, img_w, img_h):
    return [
        pred_box[0] * img_w / QWEN_SCALE,
        pred_box[1] * img_h / QWEN_SCALE,
        pred_box[2] * img_w / QWEN_SCALE,
        pred_box[3] * img_h / QWEN_SCALE,
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


def draw_bounding_boxes(image_path, output_path, gt_dict, pred_list):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size

        for label, box in gt_dict.items():
            shape = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
            draw.rectangle(shape, outline="lime", width=3)

        for p in pred_list:
            if 'bbox_2d' in p and isinstance(p['bbox_2d'], list):
                scaled_box = scale_pred_box(p['bbox_2d'], width, height)
                draw.rectangle(scaled_box, outline="red", width=3)

        img.save(output_path)
    except Exception as e:
        print(f"Error drawing boxes for {image_path}: {e}")


class DataSelector:
    def __init__(self, dataset_path, annotations_path, mat_file_path, seed):
        self.dataset_path = dataset_path
        self.annotations_path = annotations_path
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print("Loading Synset mappings...")
        self.synset_to_label = load_synset_to_label(mat_file_path)

        print("Initializing Qwen Model...")
        self.qwen = Qwen3VLInstance(seed=seed, max_new_tokens=800)

    def scan_and_sort_candidates(self):
        print("Scanning dataset annotations...")
        xml_files = sorted([f for f in os.listdir(self.annotations_path) if f.endswith('.xml')])

        single_class_multi_instance = []
        single_class_solo_instance = []
        multi_class_candidates = []

        for xml_file in tqdm(xml_files, desc="Parsing XMLs"):
            xml_path = os.path.join(self.annotations_path, xml_file)
            gt_data, unique_labels, instance_count = parse_annotation(xml_path, self.synset_to_label)

            image_file = os.path.splitext(xml_file)[0] + ".JPEG"
            candidate = {"xml_file": xml_file, "image_file": image_file, "gt": gt_data}

            if len(unique_labels) == 1:
                if instance_count >= 2:
                    single_class_multi_instance.append(candidate)
                else:
                    single_class_solo_instance.append(candidate)
            elif len(unique_labels) >= 2:
                multi_class_candidates.append(candidate)

        print(f"Found {len(single_class_multi_instance)} Single-Class (Multi-Instance) candidates.")
        print(f"Found {len(single_class_solo_instance)} Single-Class (Solo-Instance) candidates.")
        print(f"Found {len(multi_class_candidates)} Multi-Class candidates.")

        random.shuffle(single_class_multi_instance)
        random.shuffle(single_class_solo_instance)
        random.shuffle(multi_class_candidates)

        return single_class_solo_instance, single_class_multi_instance, multi_class_candidates

    def get_existing_progress(self, category):
        """
        Scans the results folder to find:
        1. Which index we should start saving at (next_index).
        2. Which images have already been processed (completed_filenames).
        """
        category_dir = os.path.join(RESULTS_BASE_DIR, category)
        if not os.path.exists(category_dir):
            return 1, set()

        completed_filenames = set()
        existing_indices = []

        for folder_name in os.listdir(category_dir):
            if not folder_name.isdigit():
                continue

            folder_path = os.path.join(category_dir, folder_name)
            result_file = os.path.join(folder_path, "original.json")

            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        if "image" in data:
                            completed_filenames.add(data["image"])
                    existing_indices.append(int(folder_name))
                except Exception:
                    pass

        next_index = max(existing_indices) + 1 if existing_indices else 1
        return next_index, completed_filenames

    def resize_image_smart(self, img, max_side=1024):
        """Resizes image so the longest side does not exceed max_side. Preserves aspect ratio."""
        w, h = img.size
        if max(w, h) <= max_side:
            return img

        scale = max_side / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def evaluate_batch(self, candidates):
        """
        Batch evaluation using run_batch_inference.
        Downsizes images for inference if larger than MAX_RESOLUTION.
        Falls back to single inference for batches of size 1.
        """
        valid_indices = []
        inference_images = []
        prompts = []
        metadata = []

        for i, cand in enumerate(candidates):
            image_path = os.path.join(self.dataset_path, cand['image_file'])
            if not os.path.exists(image_path):
                continue

            try:
                pil_image = Image.open(image_path).convert("RGB")
                orig_w, orig_h = pil_image.size

                inference_image = pil_image.copy()
                if max(orig_w, orig_h) > MAX_RESOLUTION:
                    inference_image = self.resize_image_smart(inference_image, MAX_RESOLUTION)
                inf_w, inf_h = inference_image.size

                object_names = set(k.split('_')[0] for k in cand['gt'].keys())
                objects_str = ', '.join(sorted(object_names))
                prompt = f'Identify the objects "{objects_str}" in the image and return their bounding boxes in JSON format:'

                gt_items = []
                for key, box in cand['gt'].items():
                    label = key.split('_')[0]
                    box_coord = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
                    gt_items.append((label, box_coord))

                if not gt_items:
                    continue

                valid_indices.append(i)
                inference_images.append(inference_image)
                prompts.append(prompt)
                metadata.append((orig_w, orig_h, inf_w, inf_h, gt_items))

            except Exception as e:
                print(f"Error loading {image_path}: {e}")

        if not valid_indices:
            return [(False, None)] * len(candidates)

        try:
            if len(inference_images) == 1:
                pred_strs_list, token_counts, raw_token_counts, runtime = [None], [None], [None], None
                text, tc, raw_tc, rt = self.qwen.run_inference(inference_images[0], prompts[0])
                pred_strs_list[0] = text
                token_counts[0] = tc
                raw_token_counts[0] = raw_tc
                runtime = rt
            else:
                pred_strs_list, token_counts, raw_token_counts, runtime = self.qwen.run_batch_inference(
                    inference_images, prompts
                )
        except Exception as e:
            print(f"Batch inference error: {e}")
            return [(False, None)] * len(candidates)

        batch_results_map = {}
        for batch_pos, cand_idx in enumerate(valid_indices):
            pred_str = pred_strs_list[batch_pos]
            token_count = token_counts[batch_pos]
            raw_token_count = raw_token_counts[batch_pos]
            orig_w, orig_h, inf_w, inf_h, gt_items = metadata[batch_pos]
            cand = candidates[cand_idx]

            predictions = extract_json_array(pred_str)

            ious = []
            for gt_label, gt_box in gt_items:
                best_iou_for_this_obj = 0.0
                relevant_preds = [p for p in predictions if p.get('label') == gt_label and 'bbox_2d' in p]

                for p in relevant_preds:
                    scaled_box = scale_pred_box(p['bbox_2d'], orig_w, orig_h)
                    curr_iou = calculate_iou(gt_box, scaled_box)
                    if curr_iou > best_iou_for_this_obj:
                        best_iou_for_this_obj = curr_iou
                ious.append(best_iou_for_this_obj)

            mean_iou = sum(ious) / len(ious) if ious else 0.0

            if mean_iou > IOU_THRESHOLD:
                result_entry = {
                    "image": cand['image_file'],
                    "prompt": prompts[batch_pos],
                    "prediction": pred_str,
                    "token_count": token_count,
                    "raw_token_count": raw_token_count,
                    "runtime": runtime,
                    "original_dims": [orig_w, orig_h],
                    "inference_dims": [inf_w, inf_h],
                    "parsed_preds": predictions,
                    "IoU": f"{mean_iou:.5f}",
                    "seed": str(self.seed),
                    "ground_truth": cand['gt']
                }
                batch_results_map[cand_idx] = (True, result_entry)
            else:
                batch_results_map[cand_idx] = (False, None)

        results = []
        for i in range(len(candidates)):
            results.append(batch_results_map.get(i, (False, None)))

        return results

    def save_selection(self, data, category, index):
        dir_path = os.path.join(RESULTS_BASE_DIR, category, str(index))
        os.makedirs(dir_path, exist_ok=True)

        output_data = data.copy()
        parsed_preds = output_data.pop('parsed_preds', [])

        json_path = os.path.join(dir_path, "original.json")
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        image_src_path = os.path.join(self.dataset_path, data['image'])
        image_viz_path = os.path.join(dir_path, "visualization.jpg")
        draw_bounding_boxes(image_src_path, image_viz_path, data['ground_truth'], parsed_preds)

        image_orig_dest_path = os.path.join(dir_path, "data_point.JPEG")
        try:
            shutil.copy2(image_src_path, image_orig_dest_path)
        except Exception as e:
            print(f"Error copying original image: {e}")

    def process_group(self, candidates, group_name, target_size):
        print(f"\n--- Processing Group: {group_name} ---")

        next_save_index, completed_filenames = self.get_existing_progress(group_name)

        current_count = next_save_index - 1
        needed = target_size - current_count

        print(f"Status: {current_count}/{target_size} already completed. Need {needed} more.")

        if needed <= 0:
            print("Group already complete.")
            return

        candidates_to_process = [c for c in candidates if c['image_file'] not in completed_filenames]

        pbar = tqdm(total=needed, initial=0, unit="img")
        candidate_idx = 0

        while pbar.n < needed and candidate_idx < len(candidates_to_process):

            batch_end = min(candidate_idx + BATCH_SIZE, len(candidates_to_process))
            current_batch_candidates = candidates_to_process[candidate_idx: batch_end]

            if not current_batch_candidates:
                break

            batch_results = self.evaluate_batch(current_batch_candidates)

            for success, data in batch_results:
                if success:
                    self.save_selection(data, group_name, next_save_index)
                    next_save_index += 1
                    pbar.update(1)

                    if pbar.n >= needed:
                        break

            candidate_idx += BATCH_SIZE

        pbar.close()
        if pbar.n < needed:
            print(f"Warning: Exhausted candidates for {group_name}. Found {current_count + pbar.n}/{target_size}.")

    def run_selection(self):
        solo_candidates, multi_inst_candidates, multi_class_candidates = self.scan_and_sort_candidates()

        # 50/50 split for single-class candidates
        self.process_group(solo_candidates, "single/solo", target_size=GROUP_SIZE // 2)
        self.process_group(multi_inst_candidates, "single/multi", target_size=GROUP_SIZE // 2)

        self.process_group(multi_class_candidates, "multi", target_size=GROUP_SIZE // 2)

        print(f"\nProcessing Complete. Results saved in: {os.path.abspath(RESULTS_BASE_DIR)}")


DATASET_PATH = r"dataset/2017/ILSVRC/Data/DET/val"
ANNOTATIONS_PATH = r"dataset/2017/ILSVRC/Annotations/DET/val"
MAT_FILE_PATH = r"dataset/2017/ILSVRC/devkit/data/meta_det.mat"

if __name__ == "__main__":
    selector = DataSelector(
        dataset_path=DATASET_PATH,
        annotations_path=ANNOTATIONS_PATH,
        mat_file_path=MAT_FILE_PATH,
        seed=SEED
    )

    selector.run_selection()
