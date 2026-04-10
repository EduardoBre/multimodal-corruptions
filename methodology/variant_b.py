"""
# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

main_evolution.py  (Variant B, combined perturbations, BATCHED)

Multi-Objective Evolutionary Algorithm (NSGA-II via pymoo) for finding
optimal adversarial perturbations against the Qwen-3-VL model.

Each individual represents a COMBINED multi-corruption attack across both
modalities.  Every image corruption and every text corruption is applied
sequentially, each with its own continuous scale in [0, 1].  The sum of
all image scales must be smaler_equal BUDGET_MAX and the sum of all text scales
must be smaler_equal BUDGET_MAX.

VLM and embedding inference are batched (controlled by BATCH_SIZE) for
GPU throughput.  Early stopping operates *between* batches: once a
perfect individual is found, the current batch finishes but all
subsequent batches in the generation are skipped, and the next
generation is not started.

The algorithm simultaneously optimises three conflicting objectives:
    F1: Minimise IoU           (maximise detection damage)
    F2: Minimise Image Distance (maximise visual stealth)
    F3: Minimise (1 - TextSim)  (maximise semantic stealth)

The result is a Pareto front of non-dominated solutions.

Post-optimisation, two "best" solutions are selected and saved:
    1. Best L2 , closest to the ideal point [0, 0, 0] (Euclidean knee)
    2. Best SWAD*, highest Stealth-Weighted Adversarial Degradation

For each best solution, a ground-truth overlay image is also produced
for qualitative analysis.
"""

import os
import json
import random
import re
import time
import difflib

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from image_perturbator_seq import ImagePerturbator
from text_perturbator import TextPerturbator
from qwen_3_vision_LM import Qwen3VLInstance
from qwen_3_embedding import Qwen3EmbeddingInstance

SEED = 42669
MAX_RESOLUTION = 1024
QWEN_SCALE_FACTOR = 1000
RESULTS_DIR = "results/selection"
OUTPUT_BASE_DIR = "multimodal/variant_b"

PARETO_FILENAME = "pareto_front.json"

BEST_L2_RESULT_FILENAME = "best_l2_result.json"
BEST_L2_IMAGE_FILENAME = "best_l2_adversarial.png"
BEST_L2_GT_FILENAME = "best_l2_adversarial_gt.jpeg"
BEST_L2_GT_ONLY_FILENAME = "best_l2_adversarial_gt_only.jpeg"

BEST_SWAD_RESULT_FILENAME = "best_swad_result.json"
BEST_SWAD_IMAGE_FILENAME = "best_swad_adversarial.png"
BEST_SWAD_GT_FILENAME = "best_swad_adversarial_gt.jpeg"
BEST_SWAD_GT_ONLY_FILENAME = "best_swad_adversarial_gt_only.jpeg"

# Sequential application order matters, see inline comments
IMAGE_ATTACKS = [
    "gaussian_noise",   # Noise on clean pixels, gets smoothed later
    "defocus_blur",     # Absorbs noise, degrades spatial features
    "motion_blur",      # Compounds blur, more natural on soft input
    "fog_filter",       # Global overlay, physically plausible after blur
    "pixelate",         # Locks in all prior damage at reduced resolution
    "jpeg_filter",      # ALWAYS LAST, simulates real compression
]

TEXT_ATTACKS = [
    "homophone",        # Dictionary lookup, needs clean tokens
    "synonym",          # Dictionary lookup, needs clean tokens
    "ata_saliency",     # Saliency-targeted typos, needs word boundaries
    "fragmentation",    # Splits words, after all word-level ops
    "character_noise",  # ALWAYS LAST, character-level, breaks everything upstream
]

N_IMG = len(IMAGE_ATTACKS)
N_TXT = len(TEXT_ATTACKS)
N_VAR = N_IMG + N_TXT

BUDGET_MAX = 1

POP_SIZE = 30
NUM_GENERATIONS = 15
BATCH_SIZE = 15
N_OBJ = 3

EARLY_STOP_IOU_MAX = 0.35
EARLY_STOP_IMG_DIST_MAX = 0.1
EARLY_STOP_TXT_SIM_MIN = 0.70

_DUMMY_F = np.array([1.0, 1.0, 1.0])

_GT_COLOUR = (0, 200, 0)
_PRED_COLOUR = (220, 40, 40)
_GT_FILL = (0, 200, 0, 90)
_PRED_FILL = (220, 40, 40, 70)


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


def resize_image_smart(img, max_side=1080):
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def extract_json_array(pred_str):
    text = pred_str.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def extract_target_objects(prompt):
    if not prompt:
        return []
    match = re.search(r'objects "(.*?)"', prompt)
    if match:
        return [_normalize_label(x) for x in match.group(1).split(",")]
    return []


def _normalize_label(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\s\u200b\u200d]+", "", text).lower()


def _are_labels_compatible(pred_label, gt_label):
    p = _normalize_label(pred_label)
    g = _normalize_label(gt_label)
    if p == g:
        return True
    if sorted(list(p)) == sorted(list(g)):
        return True
    if difflib.SequenceMatcher(None, p, g).ratio() >= 0.75:
        return True
    return False


def _scale_pred_box(pred_box, target_w, target_h):
    return [
        pred_box[0] * target_w / QWEN_SCALE_FACTOR,
        pred_box[1] * target_h / QWEN_SCALE_FACTOR,
        pred_box[2] * target_w / QWEN_SCALE_FACTOR,
        pred_box[3] * target_h / QWEN_SCALE_FACTOR,
    ]


def _calculate_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / float(denom) if denom > 0 else 0.0


def compute_mean_iou(gt_dict, pred_list, ref_w, ref_h, valid_prompt_labels=None):
    if not gt_dict:
        return 0.0
    gt_items = []
    for key, box in gt_dict.items():
        label = key.split("_")[0]
        gt_items.append((label, [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]))

    ious = []
    for gt_label, gt_box in gt_items:
        best_iou = 0.0
        for p in pred_list:
            if not isinstance(p, dict) or "bbox_2d" not in p:
                continue
            p_label = p.get("label", "")
            is_match = _are_labels_compatible(p_label, gt_label)
            if not is_match and valid_prompt_labels:
                for pl in valid_prompt_labels:
                    if _are_labels_compatible(p_label, pl):
                        is_match = True
                        break
            if is_match:
                cur = _calculate_iou(gt_box, _scale_pred_box(p["bbox_2d"], ref_w, ref_h))
                if cur > best_iou:
                    best_iou = cur
        ious.append(best_iou)
    return sum(ious) / len(ious) if ious else 0.0


def _is_perfect(iou, img_dist, txt_sim):
    return (
        iou <= EARLY_STOP_IOU_MAX
        and img_dist < EARLY_STOP_IMG_DIST_MAX
        and txt_sim > EARLY_STOP_TXT_SIM_MIN
    )


def decode_genome(x):
    """Convert a raw decision vector into a human-readable dict."""
    img_scales = x[:N_IMG]
    txt_scales = x[N_IMG:]
    return {
        "image_corruptions": {
            name: float(f"{img_scales[i]:.6f}")
            for i, name in enumerate(IMAGE_ATTACKS)
        },
        "text_corruptions": {
            name: float(f"{txt_scales[i]:.6f}")
            for i, name in enumerate(TEXT_ATTACKS)
        },
        "img_budget_used": float(f"{sum(img_scales):.6f}"),
        "txt_budget_used": float(f"{sum(txt_scales):.6f}"),
        "budget_max": BUDGET_MAX,
    }


def _active_corruptions_str(x):
    """One-line summary of corruptions with scale > 0."""
    parts = []
    for i, name in enumerate(IMAGE_ATTACKS):
        if x[i] > 0:
            parts.append(f"{name}={x[i]:.3f}")
    img_str = "+".join(parts) if parts else "none"

    parts = []
    for i, name in enumerate(TEXT_ATTACKS):
        if x[N_IMG + i] > 0:
            parts.append(f"{name}={x[N_IMG + i]:.3f}")
    txt_str = "+".join(parts) if parts else "none"

    return f"Img[{img_str}] Txt[{txt_str}]"


def compute_swad_metrics(iou_0, iou_m, d_I, d_T):
    """
    Compute the Stealth-Weighted Adversarial Degradation index.

    Parameters
    ----------
    iou_0 : float   Baseline mean IoU (clean image, clean prompt).
    iou_m : float   Mean IoU under attack.
    d_I   : float   Image perceptual distance \in [0, 1].
    d_T   : float   Text semantic distance \in [0, 1].

    Returns
    -------
    swad        : float   SWAD score \in [0, 1].
    delta_m_plus: float   Clamped relative degradation \in [0, 1].
    phi         : float   Stealth product \in [0, 1].
    """
    if iou_0 <= 0:
        delta_m_plus = 0.0
    else:
        delta_m = (iou_0 - iou_m) / iou_0
        delta_m_plus = max(0.0, delta_m)

    phi = (1.0 - d_I) * (1.0 - d_T)
    swad = delta_m_plus * phi
    return float(swad), float(delta_m_plus), float(phi)


def _get_font(size=14):
    """Load a TrueType font with robust fallback across OS font paths.
    I ran this on linux and it was bugging b4 """
    import glob as _glob

    _CANDIDATES = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText-Bold.otf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    _GLOB_PATTERNS = [
        "/usr/share/fonts/**/*Bold*.ttf",
        "/usr/share/fonts/**/*bold*.ttf",
    ]

    for fp in _CANDIDATES:
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue

    for pattern in _GLOB_PATTERNS:
        hits = _glob.glob(pattern, recursive=True)
        if hits:
            try:
                return ImageFont.truetype(hits[0], size)
            except Exception:
                continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_gt_overlay(adv_image_pil, gt_bboxes, vlm_parsed, orig_w, orig_h,
                    draw_predictions=True):
    """
    Draw ground-truth and (optionally) VLM-predicted bounding boxes on a
    copy of the adversarial image for qualitative analysis.

    Parameters
    ----------
    adv_image_pil   : PIL.Image   The adversarial image.
    gt_bboxes       : dict        Ground-truth bounding boxes from original.json.
    vlm_parsed      : list[dict]  Parsed VLM predictions (each with 'label', 'bbox_2d').
    orig_w, orig_h  : int         Original image dimensions for scaling predictions.
    draw_predictions: bool        If False, only ground-truth boxes are drawn.

    Returns
    -------
    PIL.Image with overlaid bounding boxes.
    """
    img = adv_image_pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_main = ImageDraw.Draw(img)

    img_w, img_h = img.size
    short_side = min(img_w, img_h)

    gt_font_size = max(16, short_side // 25)
    pred_font_size = max(14, short_side // 30)
    font = _get_font(size=gt_font_size)
    small_font = _get_font(size=pred_font_size)
    line_width = max(3, short_side // 150)

    pad_x, pad_y = 6, 4
    _outline_w = max(1, gt_font_size // 12)

    for key, box in gt_bboxes.items():
        label = key.split("_")[0]
        x1, y1 = box["xmin"], box["ymin"]
        x2, y2 = box["xmax"], box["ymax"]

        draw_overlay.rectangle([x1, y1, x2, y2], fill=_GT_FILL)
        draw_main.rectangle(
            [x1, y1, x2, y2], outline=_GT_COLOUR, width=line_width + 1
        )

        tag = f"GT: {label}"
        bbox = font.getbbox(tag)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - th - 2 * pad_y)
        draw_main.rectangle(
            [x1, label_y, x1 + tw + 2 * pad_x, label_y + th + 2 * pad_y],
            fill=_GT_COLOUR,
        )
        tx, ty = x1 + pad_x, label_y + pad_y
        draw_main.text((tx, ty), tag, fill="white", font=font,
                       stroke_width=_outline_w, stroke_fill="black")

    if draw_predictions and vlm_parsed:
        _pred_outline_w = max(1, pred_font_size // 12)
        for p in vlm_parsed:
            if not isinstance(p, dict) or "bbox_2d" not in p:
                continue
            p_label = p.get("label", "?")
            scaled = _scale_pred_box(p["bbox_2d"], orig_w, orig_h)
            x1, y1, x2, y2 = [int(round(v)) for v in scaled]

            draw_overlay.rectangle([x1, y1, x2, y2], fill=_PRED_FILL)
            draw_main.rectangle(
                [x1, y1, x2, y2], outline=_PRED_COLOUR, width=line_width
            )

            tag = f"Pred: {p_label}"
            bbox = small_font.getbbox(tag)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            label_y = min(img_h - th - 2 * pad_y, y2 + 2)
            draw_main.rectangle(
                [x1, label_y, x1 + tw + 2 * pad_x, label_y + th + 2 * pad_y],
                fill=_PRED_COLOUR,
            )
            tx, ty = x1 + pad_x, label_y + pad_y
            draw_main.text(
                (tx, ty), tag, fill="white", font=small_font,
                stroke_width=_pred_outline_w, stroke_fill="black",
            )

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


class FitnessEvaluator:
    """Initialises all heavy models once and exposes evaluate methods."""

    def __init__(self, seed=SEED):
        print("Initialising FitnessEvaluator: loading models ...")
        self.image_perturbator = ImagePerturbator()
        self.text_perturbator = TextPerturbator()
        self.qwen_vl = Qwen3VLInstance(seed=seed, max_new_tokens=800)
        self.qwen_emb = Qwen3EmbeddingInstance(seed=seed)
        print("All models loaded.\n")

    @staticmethod
    def _normalised_frobenius(clean_np, corrupt_np):
        clean = clean_np.astype(np.float64) / 255.0
        corrupt = corrupt_np.astype(np.float64) / 255.0
        diff = clean - corrupt
        c, h, w = clean.shape[2], clean.shape[0], clean.shape[1]
        return np.linalg.norm(diff) / np.sqrt(c * h * w)

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        dot = np.dot(vec_a, vec_b)
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return dot / denom if denom > 0 else 0.0

    @staticmethod
    def _extract_object_list(prompt):
        match = re.search(r'objects "(.*?)"', prompt)
        if match:
            return [item.strip() for item in match.group(1).split(",")]
        return [prompt]

    def evaluate_single(self, x, sample_data):
        """
        Evaluate one genome vector. Used for cache-miss re-evaluation
        during Pareto-front saving.
        """
        img_scales = x[:N_IMG]
        txt_scales = x[N_IMG:]

        clean_pil = sample_data["clean_image_pil"]
        clean_np = ensure_rgb(np.array(clean_pil))
        current_np = clean_np.copy()

        applied_img = {}
        for i, attack_name in enumerate(IMAGE_ATTACKS):
            scale = float(img_scales[i])
            current_np = self.image_perturbator.apply_perturbation(
                current_np, attack_name, scale=scale,
            )
            current_np = ensure_rgb(current_np)
            applied_img[attack_name] = scale

        corrupt_pil = Image.fromarray(current_np.astype(np.uint8))

        original_prompt = sample_data["original_prompt"]
        current_prompt = original_prompt
        applied_txt = {}
        for i, attack_name in enumerate(TEXT_ATTACKS):
            scale = float(txt_scales[i])
            current_prompt = self.text_perturbator.process_prompt(
                current_prompt, attack_name, scale=scale,
            )
            applied_txt[attack_name] = scale

        response_text, token_count, raw_token_count, runtime = self.qwen_vl.run_inference(
            corrupt_pil, current_prompt,
        )

        parsed_preds = extract_json_array(response_text)
        orig_w, orig_h = sample_data["orig_dims"]
        valid_prompt_labels = extract_target_objects(current_prompt)
        iou = compute_mean_iou(
            sample_data["gt_bboxes"], parsed_preds, orig_w, orig_h,
            valid_prompt_labels=valid_prompt_labels,
        )

        img_dist = self._normalised_frobenius(clean_np, current_np)

        objs_orig = self._extract_object_list(original_prompt)
        objs_corr = self._extract_object_list(current_prompt)
        pair_sims = []
        for t_orig, t_corr in zip(objs_orig, objs_corr):
            emb_o, _, _ = self.qwen_emb.run_inference(t_orig)
            emb_c, _, _ = self.qwen_emb.run_inference(t_corr)
            pair_sims.append(self._cosine_similarity(emb_o, emb_c))
        txt_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

        return {
            "iou": float(f"{iou:.5f}"),
            "img_dist": float(f"{img_dist:.5f}"),
            "txt_sim": float(f"{txt_sim:.5f}"),
            "vlm_response": response_text,
            "vlm_parsed": parsed_preds,
            "corrupt_prompt": current_prompt,
            "token_count": token_count,
            "raw_token_count": raw_token_count,
            "runtime_seconds": float(f"{runtime:.4f}"),
            "applied_img_corruptions": applied_img,
            "applied_txt_corruptions": applied_txt,
        }

    def evaluate_batch(self, xs, sample_data):
        """
        Evaluate a batch of genome vectors on a single data sample.

        Parameters
        ----------
        xs : list[ndarray]
            Each element is a genome vector of shape (N_VAR,).
        sample_data : dict
            The sample to evaluate against.

        Returns
        -------
        metrics_list : list[dict]
            One metrics dict per individual, same format as evaluate_single.
        """
        batch_size = len(xs)
        clean_pil = sample_data["clean_image_pil"]
        clean_np = ensure_rgb(np.array(clean_pil))
        original_prompt = sample_data["original_prompt"]
        orig_w, orig_h = sample_data["orig_dims"]

        corrupt_pils = []
        corrupt_nps = []
        corrupt_prompts = []
        all_applied_img = []
        all_applied_txt = []

        for x in xs:
            img_scales = x[:N_IMG]
            txt_scales = x[N_IMG:]

            current_np = clean_np.copy()
            applied_img = {}
            for i, name in enumerate(IMAGE_ATTACKS):
                scale = float(img_scales[i])
                current_np = self.image_perturbator.apply_perturbation(
                    current_np, name, scale=scale,
                )
                current_np = ensure_rgb(current_np)
                applied_img[name] = scale

            corrupt_nps.append(current_np)
            corrupt_pils.append(Image.fromarray(current_np.astype(np.uint8)))
            all_applied_img.append(applied_img)

            current_prompt = original_prompt
            applied_txt = {}
            for i, name in enumerate(TEXT_ATTACKS):
                scale = float(txt_scales[i])
                current_prompt = self.text_perturbator.process_prompt(
                    current_prompt, name, scale=scale,
                )
                applied_txt[name] = scale

            corrupt_prompts.append(current_prompt)
            all_applied_txt.append(applied_txt)

        vlm_responses, vlm_token_counts, vlm_raw_token_counts, vlm_runtime = (
            self.qwen_vl.run_batch_inference(corrupt_pils, corrupt_prompts)
        )

        # Per-label embedding similarity (iterative; caches across individuals)
        objs_orig = self._extract_object_list(original_prompt)

        orig_emb_cache = {}
        for label in objs_orig:
            if label not in orig_emb_cache:
                orig_emb_cache[label], _, _ = self.qwen_emb.run_inference(label)

        corr_obj_lists = [self._extract_object_list(cp) for cp in corrupt_prompts]
        corr_emb_cache = {}

        metrics_list = []
        for idx in range(batch_size):
            parsed_preds = extract_json_array(vlm_responses[idx])
            valid_prompt_labels = extract_target_objects(corrupt_prompts[idx])
            iou = compute_mean_iou(
                sample_data["gt_bboxes"], parsed_preds, orig_w, orig_h,
                valid_prompt_labels=valid_prompt_labels,
            )
            img_dist = self._normalised_frobenius(clean_np, corrupt_nps[idx])

            objs_corr = corr_obj_lists[idx]
            pair_sims = []
            for t_orig, t_corr in zip(objs_orig, objs_corr):
                if t_orig == t_corr:
                    pair_sims.append(1.0)
                else:
                    if t_corr not in corr_emb_cache:
                        corr_emb_cache[t_corr], _, _ = self.qwen_emb.run_inference(t_corr)
                    pair_sims.append(
                        self._cosine_similarity(orig_emb_cache[t_orig], corr_emb_cache[t_corr])
                    )
            txt_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

            metrics_list.append({
                "iou": float(f"{iou:.5f}"),
                "img_dist": float(f"{img_dist:.5f}"),
                "txt_sim": float(f"{txt_sim:.5f}"),
                "vlm_response": vlm_responses[idx],
                "vlm_parsed": parsed_preds,
                "corrupt_prompt": corrupt_prompts[idx],
                "token_count": vlm_token_counts[idx],
                "raw_token_count": vlm_raw_token_counts[idx],
                "runtime_seconds": float(f"{vlm_runtime / batch_size:.4f}"),
                "applied_img_corruptions": all_applied_img[idx],
                "applied_txt_corruptions": all_applied_txt[idx],
            })

        return metrics_list


class BudgetRepair(Repair):
    """
    Clips all genes to [0, 1] and proportionally rescales each modality's
    scales if their sum exceeds BUDGET_MAX.
    """

    def _do(self, problem, X, **kwargs):
        np.clip(X, 0.0, 1.0, out=X)

        img_block = X[:, :N_IMG]
        img_sums = img_block.sum(axis=1, keepdims=True)
        over = (img_sums > BUDGET_MAX).flatten()
        if over.any():
            img_block[over] = img_block[over] / img_sums[over] * BUDGET_MAX

        txt_block = X[:, N_IMG:]
        txt_sums = txt_block.sum(axis=1, keepdims=True)
        over = (txt_sums > BUDGET_MAX).flatten()
        if over.any():
            txt_block[over] = txt_block[over] / txt_sums[over] * BUDGET_MAX

        return X


class BudgetAwareSampling(Sampling):
    """
    Initial population sampler that produces diverse budget-usage levels.

    Picks a budget fraction t ~ Uniform(0, BUDGET_MAX) per individual per
    modality, then distributes via Dirichlet(1,...,1). This gives coverage
    from near-zero to full budget, unlike FloatRandomSampling which would
    always saturate the budget after BudgetRepair.
    """

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, N_VAR))
        for i in range(n_samples):
            t_img = np.random.uniform(0.0, BUDGET_MAX)
            if N_IMG > 0:
                fracs_img = np.random.dirichlet(np.ones(N_IMG))
                X[i, :N_IMG] = fracs_img * t_img

            t_txt = np.random.uniform(0.0, BUDGET_MAX)
            if N_TXT > 0:
                fracs_txt = np.random.dirichlet(np.ones(N_TXT))
                X[i, N_IMG:] = fracs_txt * t_txt

        return X


class EarlyStopCallback(Callback):
    def __init__(self, problem_ref):
        super().__init__()
        self.problem_ref = problem_ref
        self.trigger_gen = None

    @property
    def found_perfect(self):
        return self.problem_ref.early_stop_triggered

    def notify(self, algorithm):
        if self.trigger_gen is not None:
            return
        if self.problem_ref.early_stop_triggered:
            self.trigger_gen = algorithm.n_gen
            algorithm.termination.force_termination = True


class AdversarialProblem(Problem):
    """
    Continuous multi-objective problem for NSGA-II.

    Decision variables (N_VAR floats, all in [0, 1]):
        x[0 .. N_IMG-1]            image corruption scales
        x[N_IMG .. N_IMG+N_TXT-1]  text corruption scales

    Constraints (handled by BudgetRepair, not as pymoo constraints):
        sum(x[0..N_IMG-1])   <= BUDGET_MAX
        sum(x[N_IMG..end])   <= BUDGET_MAX

    Objectives (all minimised):
        F1 = IoU
        F2 = Image Distance
        F3 = 1 - Text Similarity

    Early stopping: when a perfect individual is found inside a batch,
    the rest of that batch is still recorded but all subsequent batches
    are skipped. The EarlyStopCallback then prevents the next generation.
    """

    def __init__(self, evaluator, sample_data, **kwargs):
        super().__init__(
            n_var=N_VAR,
            n_obj=N_OBJ,
            n_ieq_constr=0,
            xl=np.zeros(N_VAR),
            xu=np.ones(N_VAR),
            **kwargs,
        )
        self.evaluator = evaluator
        self.sample_data = sample_data
        self._eval_count = 0
        self._skipped_count = 0
        self.metrics_cache = {}
        self.early_stop_triggered = False
        self._early_stop_eval_id = None

    @staticmethod
    def _cache_key(x):
        return tuple(round(float(v), 6) for v in x)

    def reset(self, sample_data):
        self.sample_data = sample_data
        self._eval_count = 0
        self._skipped_count = 0
        self.metrics_cache.clear()
        self.early_stop_triggered = False
        self._early_stop_eval_id = None

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.full((n, N_OBJ), 1.0)

        for batch_start in range(0, n, BATCH_SIZE):
            if self.early_stop_triggered:
                remaining = n - batch_start
                self._skipped_count += remaining
                break

            batch_end = min(batch_start + BATCH_SIZE, n)
            batch_X = [X[i] for i in range(batch_start, batch_end)]

            metrics_list = self.evaluator.evaluate_batch(
                batch_X, self.sample_data
            )

            for j, metrics in enumerate(metrics_list):
                global_idx = batch_start + j
                x = X[global_idx]

                self.metrics_cache[self._cache_key(x)] = metrics
                self._eval_count += 1

                iou = metrics["iou"]
                img_dist = metrics["img_dist"]
                txt_sim = metrics["txt_sim"]

                F[global_idx] = [iou, img_dist, 1.0 - txt_sim]

                print(
                    f"  Eval #{self._eval_count:>4d} | "
                    f"{_active_corruptions_str(x)} | "
                    f"IoU={iou:.4f}  ImgD={img_dist:.4f}  TxtS={txt_sim:.4f}"
                )

                if not self.early_stop_triggered and _is_perfect(iou, img_dist, txt_sim):
                    self.early_stop_triggered = True
                    self._early_stop_eval_id = self._eval_count
                    print(
                        f"\n  EARLY STOP, perfect adversarial at eval "
                        f"#{self._eval_count}!"
                        f"\n      IoU={iou:.5f}  ImgDist={img_dist:.5f}  "
                        f"TxtSim={txt_sim:.5f}"
                        f"\n      Remaining batches in this generation will "
                        f"be skipped.\n"
                    )
                    # NOTE: we do NOT break here, the rest of the current
                    # batch was already computed so we record their metrics.

        out["F"] = F


def load_sample(folder_path):
    input_json = os.path.join(folder_path, "original.json")
    input_img = os.path.join(folder_path, "data_point.JPEG")
    if not os.path.exists(input_json) or not os.path.exists(input_img):
        raise FileNotFoundError(
            f"Missing original.json or data_point.JPEG in {folder_path}"
        )
    with open(input_json, "r") as f:
        base_data = json.load(f)
    raw_img = Image.open(input_img).convert("RGB")
    orig_w, orig_h = raw_img.size
    resized_img = resize_image_smart(raw_img, MAX_RESOLUTION)
    return {
        "clean_image_pil": resized_img,
        "original_prompt": base_data["prompt"],
        "gt_bboxes": base_data.get("ground_truth", {}),
        "filename": base_data.get("image", ""),
        "baseline_iou": float(base_data["IoU"]),
        "orig_dims": (orig_w, orig_h),
        "curr_dims": resized_img.size,
        "folder_path": folder_path,
    }


def get_all_sample_folders(results_dir=RESULTS_DIR):
    """
    Collect every valid data folder under:
      results_dir/single/solo/NNN
      results_dir/single/multi/NNN
      results_dir/multi/NNN
    Returns list of (folder_path, category_rel, folder_id).
    """
    sample_folders = []
    categories = [
        os.path.join("single", "solo"),
        os.path.join("single", "multi"),
        "multi",
    ]
    for cat_rel in categories:
        cat_abs = os.path.join(results_dir, cat_rel)
        if not os.path.isdir(cat_abs):
            continue
        for fn in os.listdir(cat_abs):
            fp = os.path.join(cat_abs, fn)
            if not os.path.isdir(fp) or not fn.isdigit():
                continue
            if not os.path.exists(os.path.join(fp, "original.json")):
                continue
            if not os.path.exists(os.path.join(fp, "data_point.JPEG")):
                continue
            sample_folders.append((fp, cat_rel, fn))
    sample_folders.sort(key=lambda t: (t[1], int(t[2])))
    return sample_folders


def get_output_dir(category, folder_id):
    return os.path.join(OUTPUT_BASE_DIR, category, folder_id)


def is_already_processed(category, folder_id):
    return os.path.exists(
        os.path.join(get_output_dir(category, folder_id), BEST_L2_RESULT_FILENAME)
    )


def _retrieve_cached_metrics(problem, x):
    key = AdversarialProblem._cache_key(x)
    cached = problem.metrics_cache.get(key)
    if cached is not None:
        return cached
    print(f"  Cache miss: re-evaluating (single) ...")
    return problem.evaluator.evaluate_single(x, problem.sample_data)


def _recreate_adversarial(x, sample_data, evaluator):
    img_scales = x[:N_IMG]
    clean_np = ensure_rgb(np.array(sample_data["clean_image_pil"]))
    current_np = clean_np.copy()
    for i, name in enumerate(IMAGE_ATTACKS):
        current_np = evaluator.image_perturbator.apply_perturbation(
            current_np, name, scale=float(img_scales[i]),
        )
        current_np = ensure_rgb(current_np)
    return Image.fromarray(current_np.astype(np.uint8))


def _build_best_meta(
    label, pareto_idx, sample_data, genome, F_vec, cached, iou_0,
    early_stopped, early_stop_gen, problem,
    folder_path, folder_id, category,
):
    iou_m = float(F_vec[0])
    d_I = float(F_vec[1])
    d_T = float(F_vec[2])
    txt_sim = 1.0 - d_T
    swad, delta_m_plus, phi = compute_swad_metrics(iou_0, iou_m, d_I, d_T)

    return {
        "selection_criterion": label,
        "data_source": {
            "folder_path": folder_path,
            "folder_id": folder_id,
            "category": category,
            "filename": sample_data["filename"],
        },
        "pareto_index": pareto_idx,
        "budget_max": BUDGET_MAX,
        "batch_size": BATCH_SIZE,
        "early_stopped": early_stopped,
        "early_stop_generation": early_stop_gen,
        "total_evaluations": problem._eval_count,
        "skipped_evaluations": problem._skipped_count,
        "genome": genome,
        "objectives": {
            "iou": float(f"{iou_m:.5f}"),
            "img_dist": float(f"{d_I:.5f}"),
            "txt_dist": float(f"{d_T:.5f}"),
            "txt_sim": float(f"{txt_sim:.5f}"),
        },
        "swad_metrics": {
            "iou_0": float(f"{iou_0:.5f}"),
            "delta_m_plus": float(f"{delta_m_plus:.5f}"),
            "phi": float(f"{phi:.5f}"),
            "swad": float(f"{swad:.5f}"),
        },
        "l2_distance": float(f"{np.linalg.norm(F_vec):.5f}"),
        "original_prompt": sample_data["original_prompt"],
        "ground_truth_bboxes": sample_data["gt_bboxes"],
        "vlm_output": {
            "adversarial_prompt": cached["corrupt_prompt"],
            "raw_response": cached["vlm_response"],
            "parsed_predictions": cached["vlm_parsed"],
            "token_count": cached["token_count"],
            "raw_token_count": cached["raw_token_count"],
            "runtime_seconds": cached["runtime_seconds"],
        },
        "applied_img_corruptions": cached["applied_img_corruptions"],
        "applied_txt_corruptions": cached["applied_txt_corruptions"],
    }


def save_pareto_front(
    result, sample_data, evaluator, problem, output_dir,
    early_stopped=False, early_stop_gen=None,
):
    os.makedirs(output_dir, exist_ok=True)

    pareto_X = result.X
    pareto_F = result.F
    if pareto_X.ndim == 1:
        pareto_X = pareto_X.reshape(1, -1)
        pareto_F = pareto_F.reshape(1, -1)

    folder_path = sample_data["folder_path"]
    folder_id = sample_data["folder_id"]
    category = sample_data["category"]
    orig_w, orig_h = sample_data["orig_dims"]

    iou_0 = sample_data["baseline_iou"]
    print(f"  Baseline IoU_0 = {iou_0:.5f} (from original.json)")

    pareto_records = []
    swad_scores = []

    for i in range(len(pareto_X)):
        genome = decode_genome(pareto_X[i])
        cached = _retrieve_cached_metrics(problem, pareto_X[i])

        iou_m = float(pareto_F[i, 0])
        d_I = float(pareto_F[i, 1])
        d_T = float(pareto_F[i, 2])
        txt_sim = 1.0 - d_T

        swad, delta_m_plus, phi = compute_swad_metrics(iou_0, iou_m, d_I, d_T)
        l2_dist = float(np.linalg.norm(pareto_F[i]))
        swad_scores.append(swad)

        record = {
            "index": i,
            "genome": genome,
            "objectives": {
                "iou": float(f"{iou_m:.5f}"),
                "img_dist": float(f"{d_I:.5f}"),
                "txt_dist": float(f"{d_T:.5f}"),
                "txt_sim": float(f"{txt_sim:.5f}"),
            },
            "swad_metrics": {
                "iou_0": float(f"{iou_0:.5f}"),
                "delta_m_plus": float(f"{delta_m_plus:.5f}"),
                "phi": float(f"{phi:.5f}"),
                "swad": float(f"{swad:.5f}"),
            },
            "l2_distance": float(f"{l2_dist:.5f}"),
            "vlm_output": {
                "corrupt_prompt": cached["corrupt_prompt"],
                "raw_response": cached["vlm_response"],
                "parsed_predictions": cached["vlm_parsed"],
                "token_count": cached["token_count"],
                "raw_token_count": cached["raw_token_count"],
                "runtime_seconds": cached["runtime_seconds"],
            },
            "applied_img_corruptions": cached["applied_img_corruptions"],
            "applied_txt_corruptions": cached["applied_txt_corruptions"],
        }
        pareto_records.append(record)

    swad_star = max(swad_scores) if swad_scores else 0.0
    swad_star_idx = int(np.argmax(swad_scores)) if swad_scores else 0

    pareto_output = {
        "data_source": {
            "folder_path": folder_path,
            "folder_id": folder_id,
            "category": category,
            "filename": sample_data["filename"],
        },
        "original_prompt": sample_data["original_prompt"],
        "ground_truth_bboxes": sample_data["gt_bboxes"],
        "baseline_iou": float(f"{iou_0:.5f}"),
        "swad_star": float(f"{swad_star:.5f}"),
        "swad_star_index": swad_star_idx,
        "budget_max": BUDGET_MAX,
        "batch_size": BATCH_SIZE,
        "early_stopped": early_stopped,
        "early_stop_generation": early_stop_gen,
        "total_evaluations": problem._eval_count,
        "skipped_evaluations": problem._skipped_count,
        "n_solutions": len(pareto_records),
        "solutions": pareto_records,
    }

    front_path = os.path.join(output_dir, PARETO_FILENAME)
    with open(front_path, "w") as f:
        json.dump(pareto_output, f, indent=4)
    print(f"\n  Pareto front ({len(pareto_records)} solutions) -> {front_path}")

    distances = np.linalg.norm(pareto_F, axis=1)
    l2_idx = int(np.argmin(distances))

    selections = [
        ("L2",   l2_idx,       BEST_L2_RESULT_FILENAME,   BEST_L2_IMAGE_FILENAME,   BEST_L2_GT_FILENAME,   BEST_L2_GT_ONLY_FILENAME),
        ("SWAD", swad_star_idx, BEST_SWAD_RESULT_FILENAME, BEST_SWAD_IMAGE_FILENAME, BEST_SWAD_GT_FILENAME, BEST_SWAD_GT_ONLY_FILENAME),
    ]

    for label, idx, json_fn, img_fn, gt_fn, gt_only_fn in selections:
        sel_X = pareto_X[idx]
        sel_F = pareto_F[idx]
        sel_genome = decode_genome(sel_X)
        sel_cached = _retrieve_cached_metrics(problem, sel_X)

        iou_m = float(sel_F[0])
        d_I = float(sel_F[1])
        d_T = float(sel_F[2])
        txt_sim = 1.0 - d_T
        swad_val, delta_m_plus, phi = compute_swad_metrics(iou_0, iou_m, d_I, d_T)
        l2_val = float(np.linalg.norm(sel_F))

        print(f"\n  BEST {label} (Pareto index={idx})")
        if early_stopped:
            print(f"  Early-stopped at generation {early_stop_gen}")
        print(f"  IoU={iou_m:.5f}  ImgDist={d_I:.5f}  TxtSim={txt_sim:.5f}  L2={l2_val:.5f}")
        print(f"  Active img: {sel_cached['applied_img_corruptions']}")
        print(f"  Active txt: {sel_cached['applied_txt_corruptions']}")
        print(f"  SWAD: IoU_0={iou_0:.5f}  delta_M+={delta_m_plus:.5f}  phi={phi:.5f}  SWAD={swad_val:.5f}")

        adv_img = _recreate_adversarial(sel_X, sample_data, evaluator)
        img_path = os.path.join(output_dir, img_fn)
        adv_img.save(img_path)

        gt_img = draw_gt_overlay(
            adv_img, sample_data["gt_bboxes"], sel_cached["vlm_parsed"],
            orig_w, orig_h, draw_predictions=True,
        )
        gt_path = os.path.join(output_dir, gt_fn)
        gt_img.save(gt_path, "JPEG", quality=95)

        gt_only_img = draw_gt_overlay(
            adv_img, sample_data["gt_bboxes"], sel_cached["vlm_parsed"],
            orig_w, orig_h, draw_predictions=False,
        )
        gt_only_path = os.path.join(output_dir, gt_only_fn)
        gt_only_img.save(gt_only_path, "JPEG", quality=95)

        meta = _build_best_meta(
            label=label, pareto_idx=idx, sample_data=sample_data,
            genome=sel_genome, F_vec=sel_F, cached=sel_cached, iou_0=iou_0,
            early_stopped=early_stopped, early_stop_gen=early_stop_gen,
            problem=problem, folder_path=folder_path, folder_id=folder_id,
            category=category,
        )
        meta_path = os.path.join(output_dir, json_fn)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        print(f"  Saved: {img_path}")
        print(f"  Saved: {gt_path}")
        print(f"  Saved: {gt_only_path}")
        print(f"  Saved: {meta_path}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    all_samples = get_all_sample_folders()
    print(f"Found {len(all_samples)} valid sample folders under '{RESULTS_DIR}'.\n")
    if not all_samples:
        print("Nothing to process. Exiting.")
        return

    pending = []
    skipped = 0
    for folder_path, category, folder_id in all_samples:
        if is_already_processed(category, folder_id):
            skipped += 1
        else:
            pending.append((folder_path, category, folder_id))

    print(f"Skipping {skipped} already-processed samples.")
    print(f"Remaining: {len(pending)} samples to optimise.\n")
    if not pending:
        print("All samples already processed. Exiting.")
        return

    evaluator = FitnessEvaluator(seed=SEED)

    first_sample_data = load_sample(pending[0][0])
    first_sample_data["category"] = pending[0][1]
    first_sample_data["folder_id"] = pending[0][2]
    problem = AdversarialProblem(evaluator, first_sample_data)

    early_stop_count = 0

    for sample_idx, (folder_path, category, folder_id) in enumerate(pending):
        sample_label = f"{category}/{folder_id}"
        print(f"\n  SAMPLE {sample_idx + 1}/{len(pending)} ,  {sample_label}")
        print(f"  Path: {folder_path}")

        try:
            sample_data = load_sample(folder_path)
            sample_data["category"] = category
            sample_data["folder_id"] = folder_id
        except Exception as e:
            print(f"  Failed to load {sample_label}: {e}. Skipping.\n")
            continue

        problem.reset(sample_data)
        early_stop_cb = EarlyStopCallback(problem)

        algorithm = NSGA2(
            pop_size=POP_SIZE,
            sampling=BudgetAwareSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            repair=BudgetRepair(),
            eliminate_duplicates=True,
        )

        termination = get_termination("n_gen", NUM_GENERATIONS)

        print(
            f"  NSGA-II  |  pop={POP_SIZE}  |  max_gens={NUM_GENERATIONS}  "
            f"|  vars={N_VAR} ({N_IMG} img + {N_TXT} txt)  "
            f"|  objectives={N_OBJ}  |  batch={BATCH_SIZE}"
        )
        print(
            f"  Constraint: sum(img_scales)<={BUDGET_MAX}  "
            f"sum(txt_scales)<={BUDGET_MAX}"
        )
        print(
            f"  Early stop: IoU<={EARLY_STOP_IOU_MAX}  "
            f"ImgDist<{EARLY_STOP_IMG_DIST_MAX}  "
            f"TxtSim>{EARLY_STOP_TXT_SIM_MIN}\n"
        )

        t_start = time.time()
        result = minimize(
            problem, algorithm, termination,
            seed=SEED, verbose=True, callback=early_stop_cb,
        )
        t_elapsed = time.time() - t_start

        did_early_stop = problem.early_stop_triggered
        stop_gen = early_stop_cb.trigger_gen

        if did_early_stop:
            early_stop_count += 1
            print(
                f"\n  Sample {sample_label} EARLY-STOPPED at gen "
                f"{stop_gen} ({problem._eval_count} real evals, "
                f"{problem._skipped_count} skipped) in {t_elapsed:.1f}s"
            )
        else:
            print(
                f"\n  Sample {sample_label} completed all "
                f"{NUM_GENERATIONS} gens in {t_elapsed:.1f}s"
            )
        print(f"     Evaluations: {problem._eval_count}")

        output_dir = get_output_dir(category, folder_id)
        save_pareto_front(
            result, sample_data, evaluator, problem, output_dir,
            early_stopped=did_early_stop, early_stop_gen=stop_gen,
        )

    print(f"\n  ALL DONE, {len(pending)} samples processed")
    print(f"  Early-stopped: {early_stop_count}/{len(pending)}")
    print(f"  Results saved under '{OUTPUT_BASE_DIR}/'")


if __name__ == "__main__":
    main()
