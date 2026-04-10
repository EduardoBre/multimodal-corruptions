import numpy as np
import cv2
from io import BytesIO
from pathlib import Path

# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

# IMPORTANT: this implementation is based on PerturbationDrive https://github.com/ast-fortiss-tum/perturbation-drive


try:
    from kernels.kernels import create_disk_kernel, create_motion_blur_kernel
except ImportError:
    print("Warning: 'kernels.kernels' not found. Blur functions may fail.")


class ImagePerturbator:
    def __init__(self, assets_root=None):
        """
        :param assets_root: Path to the root directory containing 'utils/OverlayImages'.
                            If None, attempts to derive from the current file location.
        """
        if assets_root:
            self.assets_root = Path(assets_root)
        else:
            self.assets_root = Path(__file__).resolve().parent

    def _interpolate(self, factor, values):
        """
        Linearly interpolates a float factor (0.0 to 1.0) across a list of 5 target values.
        Corresponds to the original discrete scales {0, 1, 2, 3, 4}.
        """
        factor = max(0.0, min(1.0, factor))
        xs = [0.0, 0.25, 0.5, 0.75, 1.0]

        for i in range(len(xs) - 1):
            if factor <= xs[i + 1]:
                slope = (values[i + 1] - values[i]) / (xs[i + 1] - xs[i])
                return values[i] + slope * (factor - xs[i])

        return values[-1]

    def apply_perturbation(self, image, attack_type, scale=0.0, **kwargs):
        """
        Central entry point to apply perturbations.

        :param image: Input image (numpy array).
        :param attack_type: String identifier for the attack.
        :param scale: Floating point severity (0.0 to 1.0).
        :param kwargs: Additional arguments (e.g., 'bboxes' for cutout).
        :return: Perturbed image.
        """
        method_map = {
            "jpeg_filter": self.jpeg_filter,
            "pixelate": self.pixelate,
            "defocus_blur": self.defocus_blur,
            "motion_blur": self.motion_blur,
            "gaussian_noise": self.gaussian_noise,
            "fog_filter": self.fog_filter,
            "snow_filter": self.snow_filter,
            "contrast": self.contrast,
            "elastic": self.elastic,
            "cutout": self.cutout_filter_with_bbox,
            "false_color": self.false_color_filter,
            "grayscale": self.grayscale_filter,
        }

        if attack_type not in method_map:
            raise ValueError(f"Unknown attack type: {attack_type}. Available: {list(method_map.keys())}")

        func = method_map[attack_type]

        if attack_type == "cutout":
            bboxes = kwargs.get('bboxes')
            if bboxes is None:
                print("Warning: 'cutout' requires 'bboxes' in kwargs. Returning original image.")
                return image
            return func(scale, image, bboxes)

        return func(scale, image)

    def jpeg_filter(self, scale, image):
        """Introduce JPEG compression artifacts."""
        factor = int(self._interpolate(scale, [30, 18, 15, 10, 5]))

        image = np.asarray(image, dtype=np.uint8)

        _, jpeg_encoded_image = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), factor]
        )
        jpeg_stream = BytesIO(jpeg_encoded_image.tobytes())
        return cv2.imdecode(
            np.frombuffer(jpeg_stream.read(), np.uint8), cv2.IMREAD_COLOR
        )

    def pixelate(self, scale, img):
        """Pixelates the image."""
        factor = self._interpolate(scale, [0.85, 0.55, 0.35, 0.2, 0.1])
        img = np.asarray(img, dtype=np.uint8)
        h, w = img.shape[:2]
        new_w = max(1, int(w * factor))
        new_h = max(1, int(h * factor))
        img = cv2.resize(img, (new_w, new_h), cv2.INTER_AREA)
        return cv2.resize(img, (w, h), cv2.INTER_NEAREST)

    def defocus_blur(self, scale, image):
        """Applies a defocus blur."""
        factor = int(self._interpolate(scale, [2, 5, 6, 9, 12]))
        if factor < 1: factor = 1

        image = np.asarray(image, dtype=np.uint8)
        kernel = create_disk_kernel(factor)
        return cv2.filter2D(image, -1, kernel)

    def motion_blur(self, scale, image):
        """Apply motion blur."""
        s_list = [2, 4, 6, 10, 15]
        a_list = [5, 12, 20, 30, 45]

        size = int(self._interpolate(scale, s_list))
        angle = self._interpolate(scale, a_list)
        if size < 1: size = 1

        image = np.asarray(image, dtype=np.uint8)
        kernel = create_motion_blur_kernel(size, angle)
        return cv2.filter2D(image, -1, kernel)

    def gaussian_noise(self, scale, img):
        """Adds gaussian noise."""
        factor = self._interpolate(scale, [0.03, 0.06, 0.12, 0.18, 0.22])
        x = np.array(img, dtype=np.float32) / 255.0
        noisy = np.clip(x + np.random.normal(size=x.shape, scale=factor), 0, 1).astype(np.float32)
        return (noisy * 255).astype(np.uint8)

    def fog_filter(self, scale, image):
        """Apply a fog effect."""
        int_list = [0.1, 0.2, 0.3, 0.45, 0.65]
        noise_list = [0.05, 0.1, 0.2, 0.3, 0.45]

        intensity = self._interpolate(scale, int_list)
        noise_amount = self._interpolate(scale, noise_list)

        image = np.asarray(image, dtype=np.uint8)
        fog_overlay = np.full_like(image, 255, dtype=np.uint8)
        noise = np.random.normal(scale=noise_amount * 255, size=image.shape).clip(0, 255).astype(np.uint8)
        fog_overlay = cv2.addWeighted(fog_overlay, 1 - noise_amount, noise, noise_amount, 0)
        return cv2.addWeighted(image, 1 - intensity, fog_overlay, intensity, 0)

    def snow_filter(self, scale, image):
        """Apply a snow effect using an overlay image."""
        intensity = self._interpolate(scale, [0.15, 0.22, 0.3, 0.45, 0.6])

        frost_path = self.assets_root / "utils" / "OverlayImages" / "snow.png"

        if not frost_path.exists():
            raise FileNotFoundError(
                f"Could not find `snow.png`. Expected at `{frost_path}`."
            )

        frost_overlay = cv2.imread(str(frost_path), cv2.IMREAD_UNCHANGED)
        if frost_overlay is None:
            raise IOError(f"File `{frost_path}` exists but could not be decoded by OpenCV.")

        image = np.asarray(image, dtype=np.uint8)
        frost_overlay_resized = cv2.resize(frost_overlay, (image.shape[1], image.shape[0]))
        bgr = frost_overlay_resized[:, :, :3]
        alpha = frost_overlay_resized[:, :, 3] / 255.0

        frosted_image = (1 - (intensity * alpha[:, :, np.newaxis])) * image + (intensity * bgr)
        frosted_image = np.clip(frosted_image, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(frosted_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.8
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def contrast(self, scale, img):
        """Increase or decrease contrast."""
        factor = self._interpolate(scale, [1.1, 1.2, 1.3, 1.5, 1.7])
        pivot = 127.5
        img = np.asarray(img, dtype=np.float64)
        return np.clip(pivot + (img - pivot) * factor, 0, 255).astype(np.uint8)

    def elastic(self, scale, img):
        """Applies an elastic deformation."""
        alpha_list = [2, 3, 5, 7, 10]
        sigma_list = [0.4, 0.75, 0.9, 1.2, 1.5]

        alpha = self._interpolate(scale, alpha_list)
        sigma = self._interpolate(scale, sigma_list)

        img = np.asarray(img, dtype=np.uint8)

        dx = np.random.uniform(-1, 1, img.shape[:2]) * alpha
        dy = np.random.uniform(-1, 1, img.shape[:2]) * alpha
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)

        x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        return cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def cutout_filter_with_bbox(self, scale, image, bboxes):
        """
        Applies cutout corruption limiting occlusion of ground truth objects.
        Interpolates 'count' and 'max_coverage'.
        """
        image = np.asarray(image, dtype=np.uint8).copy()
        h, w, _ = image.shape

        count_list = [1, 2, 4, 6, 10]
        cov_list = [0.05, 0.10, 0.15, 0.25, 0.33]

        target_count = int(self._interpolate(scale, count_list))
        target_coverage = self._interpolate(scale, cov_list)

        bbox_masks = [np.zeros((b[3] - b[1], b[2] - b[0])) for b in bboxes]
        attempts = 0
        patches_applied = 0

        while patches_applied < target_count and attempts < target_count * 5:
            attempts += 1
            patch_h = np.random.randint(int(h * 0.05), int(h * 0.2) + 1)
            patch_w = np.random.randint(int(w * 0.05), int(w * 0.2) + 1)

            if h - patch_h <= 0 or w - patch_w <= 0:
                continue

            x = np.random.randint(0, h - patch_h)
            y = np.random.randint(0, w - patch_w)

            valid_patch = True
            temp_mask_updates = []

            for i, box in enumerate(bboxes):
                bx_min, by_min, bx_max, by_max = box
                box_area = (bx_max - bx_min) * (by_max - by_min)
                if box_area <= 0: continue

                inter_row1 = max(x, by_min)
                inter_row2 = min(x + patch_h, by_max)
                inter_col1 = max(y, bx_min)
                inter_col2 = min(y + patch_w, bx_max)

                if inter_row1 < inter_row2 and inter_col1 < inter_col2:
                    local_row1 = inter_row1 - by_min
                    local_row2 = inter_row2 - by_min
                    local_col1 = inter_col1 - bx_min
                    local_col2 = inter_col2 - bx_min

                    current_covered_pixels = np.sum(bbox_masks[i])
                    roi = bbox_masks[i][local_row1:local_row2, local_col1:local_col2]
                    new_pixels = ((inter_row2 - inter_row1) * (inter_col2 - inter_col1)) - np.sum(roi)

                    if (current_covered_pixels + new_pixels) / box_area > target_coverage:
                        valid_patch = False
                        break

                    temp_mask_updates.append((i, slice(local_row1, local_row2), slice(local_col1, local_col2)))

            if valid_patch:
                image[x: x + patch_h, y: y + patch_w, :] = 0
                for idx, slc_row, slc_col in temp_mask_updates:
                    bbox_masks[idx][slc_row, slc_col] = 1
                patches_applied += 1

        return image

    def false_color_filter(self, scale, image):
        """
        Apply false color effect.
        Maps the float range to discrete buckets [0, 4].
        """
        idx = int(scale * 5)
        idx = min(idx, 4)

        image = np.asarray(image, dtype=np.uint8)
        false_color = image.copy()
        if idx == 0:
            false_color[:, :, 0] = image[:, :, 1]
            false_color[:, :, 1] = image[:, :, 2]
            false_color[:, :, 2] = image[:, :, 0]
        elif idx == 1:
            false_color[:, :, 0] = image[:, :, 1]
            false_color[:, :, 1] = image[:, :, 0]
            false_color[:, :, 2] = image[:, :, 2]
        elif idx == 2:
            false_color[:, :, 0] = image[:, :, 2]
            false_color[:, :, 1] = image[:, :, 1]
            false_color[:, :, 2] = image[:, :, 0]
        elif idx == 3:
            false_color[:, :, 0] = 255 - image[:, :, 0]
            false_color[:, :, 1] = 255 - image[:, :, 1]
            false_color[:, :, 2] = 255 - image[:, :, 2]
        elif idx == 4:
            false_color[:, :, 0] = (image[:, :, 0] + image[:, :, 1]) // 2
            false_color[:, :, 1] = (image[:, :, 1] + image[:, :, 2]) // 2
            false_color[:, :, 2] = (image[:, :, 2] + image[:, :, 0]) // 2
        return false_color

    def grayscale_filter(self, scale, image):
        """Apply a grayscale effect."""
        severity = self._interpolate(scale, [0.1, 0.2, 0.35, 0.55, 0.85])
        image = np.asarray(image, dtype=np.uint8)
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grayscale_img_colored = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
        return cv2.addWeighted(image, 1 - severity, grayscale_img_colored, severity, 0)
