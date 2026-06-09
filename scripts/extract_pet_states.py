import argparse
import json
import math
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
BUILTIN_BACKENDS = ("grabcut", "background-diff", "rembg", "rvm")


@dataclass
class ExtractedFrame:
    state: str
    index: int
    rgba: np.ndarray
    bbox: tuple[int, int, int, int]


class SegmentationBackend(ABC):
    name: str

    def prepare(self, video: Path, sampled_frames: list[np.ndarray]) -> None:
        return None

    @abstractmethod
    def alpha(self, bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def extract(self, video: Path, state: str, args: argparse.Namespace) -> list[ExtractedFrame]:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        indices = sample_indices(frame_count, source_fps, args.fps, args.max_frames)
        sampled_frames: list[np.ndarray] = []
        for source_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, bgr = cap.read()
            if ok:
                sampled_frames.append(bgr)

        self.prepare(video, sampled_frames)
        extracted: list[ExtractedFrame] = []
        for output_index, bgr in enumerate(sampled_frames):
            alpha = self.alpha(bgr)
            bbox = frame_bbox(alpha, args.pad)
            if bbox is None:
                continue
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = alpha
            extracted.append(ExtractedFrame(state=state, index=output_index, rgba=rgba, bbox=bbox))

        cap.release()
        if not extracted:
            raise RuntimeError(f"No foreground frames extracted from: {video}")
        return extracted


class GrabCutBackend(SegmentationBackend):
    name = "grabcut"

    def __init__(
        self,
        iterations: int,
        min_component_area: int,
        clean_floor_marker: bool,
    ) -> None:
        self.iterations = iterations
        self.min_component_area = min_component_area
        self.clean_floor_marker = clean_floor_marker

    def alpha(self, bgr: np.ndarray) -> np.ndarray:
        return segment_frame(
            bgr,
            self.iterations,
            self.min_component_area,
            self.clean_floor_marker,
        )


class BackgroundDiffBackend(SegmentationBackend):
    name = "background-diff"

    def __init__(self, min_component_area: int, clean_floor_marker: bool) -> None:
        self.min_component_area = min_component_area
        self.clean_floor_marker = clean_floor_marker
        self.background: np.ndarray | None = None

    def prepare(self, video: Path, sampled_frames: list[np.ndarray]) -> None:
        if not sampled_frames:
            raise RuntimeError(f"No frames sampled for {video}")
        h, w = sampled_frames[0].shape[:2]
        edges = []
        for frame in sampled_frames:
            strip = max(6, min(h, w) // 24)
            bg = estimate_border_background(frame)
            canvas = np.empty_like(frame)
            canvas[:, :] = bg
            canvas[:strip, :, :] = frame[:strip, :, :]
            canvas[-strip:, :, :] = frame[-strip:, :, :]
            canvas[:, :strip, :] = frame[:, :strip, :]
            canvas[:, -strip:, :] = frame[:, -strip:, :]
            edges.append(canvas)
        self.background = np.median(np.stack(edges, axis=0), axis=0).astype(np.uint8)

    def alpha(self, bgr: np.ndarray) -> np.ndarray:
        if self.background is None:
            raise RuntimeError("BackgroundDiffBackend.prepare must run before alpha.")
        diff = cv2.absdiff(bgr, self.background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 24, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        if self.clean_floor_marker:
            mask = remove_floor_marker(mask, bgr)
        return filter_components(mask, self.min_component_area)


class RembgBackend(SegmentationBackend):
    name = "rembg"

    def __init__(self, model: str, min_component_area: int, clean_floor_marker: bool) -> None:
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise RuntimeError(
                "The rembg backend requires the optional 'rembg' package. "
                "Install it, then rerun with --backend rembg."
            ) from exc
        self.remove = remove
        self.session = new_session(model)
        self.model = model
        self.min_component_area = min_component_area
        self.clean_floor_marker = clean_floor_marker

    def alpha(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        cutout = self.remove(image, session=self.session)
        alpha = np.array(cutout.convert("RGBA"))[:, :, 3]
        if self.clean_floor_marker:
            alpha = remove_floor_marker(alpha, bgr)
        return filter_components(alpha, self.min_component_area)


class RobustVideoMattingBackend(SegmentationBackend):
    name = "rvm"

    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        variant: str,
        device: str,
        downsample_ratio: float | None,
        seq_chunk: int,
        min_component_area: int,
        clean_floor_marker: bool,
    ) -> None:
        if not repo.exists():
            raise RuntimeError(f"RobustVideoMatting repo not found: {repo}")
        if not checkpoint.exists():
            raise RuntimeError(
                "RVM checkpoint not found. Download rvm_mobilenetv3.pth or rvm_resnet50.pth "
                f"and pass it with --rvm-checkpoint. Missing: {checkpoint}"
            )
        self.repo = repo
        self.checkpoint = checkpoint
        self.variant = variant
        self.device = device
        self.downsample_ratio = downsample_ratio
        self.seq_chunk = seq_chunk
        self.min_component_area = min_component_area
        self.clean_floor_marker = clean_floor_marker

    def alpha(self, bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError("RVM processes whole videos; call extract instead.")

    def extract(self, video: Path, state: str, args: argparse.Namespace) -> list[ExtractedFrame]:
        repo_path = str(self.repo.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            import torch
            from inference import convert_video
            from model import MattingNetwork
        except ImportError as exc:
            raise RuntimeError(
                "The RVM backend requires PyTorch and RobustVideoMatting inference dependencies. "
                "Install RobustVideoMatting/requirements_inference.txt in the Python environment."
            ) from exc

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()

        with tempfile.TemporaryDirectory(prefix=f"rvm_{state}_") as temp:
            frames_dir = Path(temp) / "rgba"
            model = MattingNetwork(self.variant).eval().to(self.device)
            state_dict = torch.load(self.checkpoint, map_location=self.device)
            model.load_state_dict(state_dict)
            convert_video(
                model,
                input_source=str(video),
                downsample_ratio=self.downsample_ratio,
                output_type="png_sequence",
                output_composition=str(frames_dir),
                seq_chunk=self.seq_chunk,
                progress=not args.disable_progress,
                device=self.device,
            )

            frame_paths = sorted(frames_dir.glob("*.png"))
            if not frame_paths:
                raise RuntimeError(f"RVM did not write any RGBA frames for: {video}")

            indices = sample_indices(len(frame_paths) or frame_count, source_fps, args.fps, args.max_frames)
            extracted: list[ExtractedFrame] = []
            for output_index, source_index in enumerate(indices):
                if source_index >= len(frame_paths):
                    continue
                rgba = np.array(Image.open(frame_paths[source_index]).convert("RGBA"))
                bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                alpha = rgba[:, :, 3]
                if self.clean_floor_marker:
                    alpha = remove_floor_marker(alpha, bgr)
                alpha = filter_components(alpha, self.min_component_area)
                bbox = frame_bbox(alpha, args.pad)
                if bbox is None:
                    continue
                rgba[:, :, 3] = alpha
                extracted.append(ExtractedFrame(state=state, index=output_index, rgba=rgba, bbox=bbox))

        if not extracted:
            raise RuntimeError(f"No foreground frames extracted from RVM output: {video}")
        return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract transparent desktop-pet animation states from videos."
    )
    parser.add_argument("--input-dir", default=".", help="Folder containing state videos.")
    parser.add_argument("--output-dir", default="extracted_assets", help="Output folder.")
    parser.add_argument(
        "--backend",
        choices=BUILTIN_BACKENDS,
        default="grabcut",
        help="Segmentation backend to use.",
    )
    parser.add_argument(
        "--deps-dir",
        default=".python-deps",
        help="Optional local Python dependency folder for plugin backends.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default="models/rembg",
        help="Project-local model cache folder for AI backends.",
    )
    parser.add_argument(
        "--rembg-model",
        default="u2net",
        help="rembg model name, for example u2net, u2netp, birefnet_general_lite, dis_anime.",
    )
    parser.add_argument(
        "--rvm-repo",
        default="../RobustVideoMatting",
        help="Path to the external RobustVideoMatting repository.",
    )
    parser.add_argument(
        "--rvm-checkpoint",
        default="models/rvm/rvm_mobilenetv3.pth",
        help="Path to a RobustVideoMatting .pth checkpoint.",
    )
    parser.add_argument(
        "--rvm-variant",
        default="mobilenetv3",
        choices=("mobilenetv3", "resnet50"),
        help="RVM model variant.",
    )
    parser.add_argument(
        "--rvm-device",
        default="cpu",
        help="Torch device for RVM inference, for example cpu or cuda.",
    )
    parser.add_argument(
        "--rvm-downsample-ratio",
        type=float,
        default=None,
        help="RVM downsample ratio. Leave unset for RVM auto mode.",
    )
    parser.add_argument(
        "--rvm-seq-chunk",
        type=int,
        default=1,
        help="Number of frames RVM processes per chunk.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Target extraction FPS. Use 0 to keep every source frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=48,
        help="Maximum frames per state after sampling. Use 0 for no limit.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=12,
        help="Padding around the extracted pet before normalizing to a shared canvas.",
    )
    parser.add_argument(
        "--grabcut-iterations",
        type=int,
        default=5,
        help="OpenCV GrabCut iterations per sampled frame.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=50,
        help="Ignore tiny foreground components below this pixel area.",
    )
    parser.add_argument(
        "--no-clean-floor-marker",
        action="store_true",
        help="Disable removal of green floor selection markers near the feet.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable progress bars for plugin backends.",
    )
    return parser.parse_args()


def create_backend(args: argparse.Namespace) -> SegmentationBackend:
    deps_dir = Path(args.deps_dir)
    if deps_dir.exists():
        deps_path = str(deps_dir.resolve())
        if deps_path not in sys.path:
            sys.path.insert(0, deps_path)
    model_cache_dir = Path(args.model_cache_dir)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("U2NET_HOME", str(model_cache_dir.resolve()))

    if args.backend == "grabcut":
        return GrabCutBackend(
            args.grabcut_iterations,
            args.min_component_area,
            clean_floor_marker=not args.no_clean_floor_marker,
        )
    if args.backend == "background-diff":
        return BackgroundDiffBackend(
            args.min_component_area,
            clean_floor_marker=not args.no_clean_floor_marker,
        )
    if args.backend == "rembg":
        return RembgBackend(
            args.rembg_model,
            args.min_component_area,
            clean_floor_marker=not args.no_clean_floor_marker,
        )
    if args.backend == "rvm":
        return RobustVideoMattingBackend(
            Path(args.rvm_repo).resolve(),
            Path(args.rvm_checkpoint).resolve(),
            args.rvm_variant,
            args.rvm_device,
            args.rvm_downsample_ratio,
            args.rvm_seq_chunk,
            args.min_component_area,
            clean_floor_marker=not args.no_clean_floor_marker,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def discover_videos(input_dir: Path) -> list[Path]:
    videos = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    order = {"idle": 0, "initial": 0, "初始状态": 0}

    def sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem.lower()
        return order.get(stem, 10), stem

    return sorted(videos, key=sort_key)


def checkerboard(size: tuple[int, int], tile: int = 12) -> Image.Image:
    image = Image.new("RGB", size, (246, 246, 246))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], tile):
        for x in range(0, size[0], tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(222, 222, 222))
    return image


def estimate_border_background(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    strip = max(4, min(h, w) // 40)
    samples = np.concatenate(
        [
            bgr[:strip, :, :].reshape(-1, 3),
            bgr[-strip:, :, :].reshape(-1, 3),
            bgr[:, :strip, :].reshape(-1, 3),
            bgr[:, -strip:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples, axis=0)


def remove_floor_marker(mask: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    y_grid = np.arange(h)[:, None]
    lower = y_grid > int(h * 0.45)

    green = cv2.inRange(hsv, np.array([32, 18, 30]), np.array([110, 255, 255])) > 0
    marker = green & lower

    bg = estimate_border_background(bgr)
    distance = np.linalg.norm(bgr.astype(np.float32) - bg.astype(np.float32), axis=2)
    near_background = distance < 34
    lower_near_background = (distance < 72) & lower

    cleaned = mask.copy()
    cleaned[marker | near_background | lower_near_background] = 0
    return cleaned


def filter_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    h, _ = mask.shape
    if mask.dtype != np.uint8:
        mask = mask.astype("uint8")
    _, mask = cv2.threshold(mask, 8, 255, cv2.THRESH_BINARY)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    for label in range(1, labels_count):
        x, y, width, height, area = stats[label]
        bottom = y + height
        if area < min_component_area:
            continue
        if y <= h * 0.12:
            continue
        if bottom <= h * 0.36:
            continue
        kept[labels == label] = 255
    kernel = np.ones((3, 3), np.uint8)
    kept = cv2.morphologyEx(kept, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(kept, (3, 3), 0)


def segment_frame(
    bgr: np.ndarray,
    grabcut_iterations: int,
    min_component_area: int,
    clean_floor_marker: bool,
) -> np.ndarray:
    h, w = bgr.shape[:2]
    rect = (
        int(w * 0.10),
        int(h * 0.08),
        max(1, int(w * 0.80)),
        max(1, int(h * 0.88)),
    )
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, grabcut_iterations, cv2.GC_INIT_WITH_RECT)

    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    if clean_floor_marker:
        foreground = remove_floor_marker(foreground, bgr)

    return filter_components(foreground, min_component_area)


def frame_bbox(alpha: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return None
    h, w = alpha.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def sample_indices(frame_count: int, source_fps: float, target_fps: float, max_frames: int) -> list[int]:
    if target_fps <= 0 or source_fps <= 0:
        indices = list(range(frame_count))
    else:
        step = max(1, int(round(source_fps / target_fps)))
        indices = list(range(0, frame_count, step))
    if max_frames > 0 and len(indices) > max_frames:
        stride = len(indices) / max_frames
        indices = [indices[min(len(indices) - 1, int(round(i * stride)))] for i in range(max_frames)]
    return sorted(set(indices))


def extract_video(
    video: Path,
    state: str,
    args: argparse.Namespace,
    backend: SegmentationBackend,
) -> list[ExtractedFrame]:
    return backend.extract(video, state, args)


def normalize_frames_by_state(
    extracted_by_state: dict[str, list[ExtractedFrame]]
) -> tuple[dict[str, list[Image.Image]], dict[str, int]]:
    state_crops: dict[str, list[np.ndarray]] = {}
    crop_sizes = []

    for state, frames in extracted_by_state.items():
        x0 = min(item.bbox[0] for item in frames)
        y0 = min(item.bbox[1] for item in frames)
        x1 = max(item.bbox[2] for item in frames)
        y1 = max(item.bbox[3] for item in frames)
        crops = [item.rgba[y0:y1, x0:x1] for item in frames]
        state_crops[state] = crops
        crop_sizes.append((x1 - x0, y1 - y0))

    canvas_w = max(width for width, _ in crop_sizes)
    canvas_h = max(height for _, height in crop_sizes)
    canvas_w = int(math.ceil(canvas_w / 2) * 2)
    canvas_h = int(math.ceil(canvas_h / 2) * 2)

    normalized_by_state: dict[str, list[Image.Image]] = {}
    for state, crops in state_crops.items():
        normalized_by_state[state] = []
        for crop in crops:
            image = Image.fromarray(crop)
            canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            x = (canvas_w - image.width) // 2
            y = canvas_h - image.height
            canvas.alpha_composite(image, (x, y))
            normalized_by_state[state].append(canvas)

    return normalized_by_state, {"width": canvas_w, "height": canvas_h}


def write_state_outputs(
    output_dir: Path,
    state: str,
    images: list[Image.Image],
    preview_duration_ms: int,
) -> dict:
    state_dir = output_dir / "states" / state
    frames_dir = state_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_files = []
    for index, image in enumerate(images):
        path = frames_dir / f"{state}_{index:03d}.png"
        image.save(path)
        frame_files.append(str(path.as_posix()))

    preview_frames = []
    for image in images:
        bg = checkerboard(image.size)
        bg_rgba = bg.convert("RGBA")
        bg_rgba.alpha_composite(image)
        preview_frames.append(bg_rgba.convert("P", palette=Image.Palette.ADAPTIVE))

    preview_path = state_dir / f"{state}_preview.gif"
    preview_frames[0].save(
        preview_path,
        save_all=True,
        append_images=preview_frames[1:],
        duration=preview_duration_ms,
        loop=0,
        disposal=2,
    )

    return {
        "state": state,
        "frame_count": len(images),
        "frames_dir": str(frames_dir.as_posix()),
        "preview": str(preview_path.as_posix()),
        "frames": frame_files,
    }


def write_contact_sheet(output_dir: Path, states: dict[str, list[Image.Image]]) -> Path:
    qa_dir = output_dir / "_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    columns = 6
    cell_w = max(image.width for images in states.values() for image in images) + 24
    cell_h = max(image.height for images in states.values() for image in images) + 44
    rows = sum(1 for _ in states)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for row, (state, images) in enumerate(states.items()):
        sample_count = min(columns, len(images))
        if sample_count == 1:
            indices = [0]
        else:
            indices = [round(i * (len(images) - 1) / (sample_count - 1)) for i in range(sample_count)]
        for col, index in enumerate(indices):
            image = images[index]
            bg = checkerboard(image.size)
            bg_rgba = bg.convert("RGBA")
            bg_rgba.alpha_composite(image)
            x = col * cell_w + (cell_w - image.width) // 2
            y = row * cell_h + 20
            sheet.paste(bg_rgba.convert("RGB"), (x, y))
            draw.text((col * cell_w + 8, row * cell_h + 4), f"{state} #{index}", fill=(0, 0, 0))

    path = qa_dir / "contact-sheet.png"
    sheet.save(path)
    return path


def backend_options(args: argparse.Namespace) -> dict:
    if args.backend == "rembg":
        return {"rembg_model": args.rembg_model}
    if args.backend == "rvm":
        return {
            "rvm_repo": args.rvm_repo,
            "rvm_checkpoint": args.rvm_checkpoint,
            "rvm_variant": args.rvm_variant,
            "rvm_device": args.rvm_device,
            "rvm_downsample_ratio": args.rvm_downsample_ratio,
            "rvm_seq_chunk": args.rvm_seq_chunk,
        }
    return {}


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(input_dir)
    if not videos:
        raise SystemExit(f"No videos found in {input_dir}")

    extracted_by_state: dict[str, list[ExtractedFrame]] = {}
    for video in videos:
        state = video.stem
        print(f"Extracting {state} from {video.name} with {args.backend}...")
        backend = create_backend(args)
        extracted_by_state[state] = extract_video(video, state, args, backend)

    normalized_by_state, canvas = normalize_frames_by_state(extracted_by_state)

    preview_duration_ms = int(1000 / args.fps) if args.fps > 0 else 33
    manifest = {
        "source_dir": str(input_dir),
        "canvas": canvas,
        "target_fps": args.fps,
        "backend": args.backend,
        "backend_options": backend_options(args),
        "states": [],
    }
    for state, images in normalized_by_state.items():
        manifest["states"].append(write_state_outputs(output_dir, state, images, preview_duration_ms))

    contact_sheet = write_contact_sheet(output_dir, normalized_by_state)
    manifest["contact_sheet"] = str(contact_sheet.as_posix())

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()
