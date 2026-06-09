import argparse
import json
import statistics
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


WEBM_SUFFIXES = (".webm",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import already-cut transparent WebM pet states.")
    parser.add_argument("--input-dir", default=".", help="Folder containing transparent WebM files.")
    parser.add_argument("--output-dir", default="transparent_webm_assets", help="Output asset folder.")
    parser.add_argument("--deps-dir", default=".python-deps", help="Folder containing imageio-ffmpeg.")
    parser.add_argument("--fps", type=float, default=12.0, help="Target FPS. Use 0 to keep source FPS.")
    parser.add_argument("--max-frames", type=int, default=48, help="Maximum frames per state. Use 0 for no limit.")
    parser.add_argument("--pad", type=int, default=12, help="Transparent bbox padding before normalization.")
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=24,
        help="Alpha values below this threshold are cleared before measuring and exporting frames.",
    )
    parser.add_argument(
        "--align-mode",
        choices=("frame", "state"),
        default="frame",
        help=(
            "frame centers and bottom-aligns every frame independently; "
            "state preserves each state's source motion inside its union crop."
        ),
    )
    parser.add_argument(
        "--scale-mode",
        choices=("state-height", "none"),
        default="state-height",
        help="state-height scales each state's median transparent height to a shared reference height.",
    )
    parser.add_argument(
        "--reference-state",
        default="idle",
        help="State whose median transparent height should define the default target size.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=0,
        help="Explicit target transparent height in pixels. Overrides --reference-state when set.",
    )
    parser.add_argument(
        "--flip-states",
        default="",
        help="Comma-separated state names to mirror horizontally before normalization.",
    )
    parser.add_argument(
        "--drop-last-frame",
        action="store_true",
        help="Drop the final sampled frame from each state to avoid non-looping video tail snaps.",
    )
    parser.add_argument(
        "--drop-outlier-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop frames whose transparent/body bbox is a strong size outlier for the state.",
    )
    parser.add_argument(
        "--preserve-y-states",
        default="",
        help="Comma-separated state names whose vertical motion should be preserved while x is centered.",
    )
    parser.add_argument(
        "--stabilize-marker-x-states",
        default="",
        help="Comma-separated state names whose green floor marker should stay fixed on the x axis.",
    )
    parser.add_argument(
        "--trim-leading-frames",
        default="",
        help="Comma-separated state=count pairs for removing leading frames, e.g. action_2=6.",
    )
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw extracted PNG frames.")
    return parser.parse_args()


def state_name(path: Path) -> str:
    name = path.stem
    for suffix in ("-Picsart-BackgroundRemover", "_Picsart_BackgroundRemover"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def discover_webms(input_dir: Path) -> list[Path]:
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in WEBM_SUFFIXES]
    order = {"idle": 0, "initial": 0, "初始状态": 0}
    return sorted(files, key=lambda p: (order.get(state_name(p).lower(), 10), state_name(p).lower()))


def get_ffmpeg(deps_dir: Path) -> str:
    deps_path = str(deps_dir.resolve())
    if deps_path not in sys.path:
        sys.path.insert(0, deps_path)
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is required. Install it into the project with: "
            "python -m pip install imageio-ffmpeg --target .python-deps"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def checkerboard(size: tuple[int, int], tile: int = 12) -> Image.Image:
    image = Image.new("RGB", size, (246, 246, 246))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], tile):
        for x in range(0, size[0], tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(222, 222, 222))
    return image


def alpha_bbox(image: Image.Image, pad: int) -> tuple[int, int, int, int] | None:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(image.width, x1 + pad),
        min(image.height, y1 + pad),
    )


def clean_alpha(image: Image.Image, threshold: int) -> Image.Image:
    if threshold <= 0:
        return image
    image = image.copy()
    r, g, b, alpha = image.split()
    alpha = alpha.point(lambda value: 0 if value < threshold else value)
    image.putalpha(alpha)
    return image


def marker_aware_body_bbox(image: Image.Image, pad: int) -> tuple[int, int, int, int] | None:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    mask = Image.new("L", rgba.size, 0)
    out = mask.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            red, green, blue, alpha = pixels[x, y]
            if alpha == 0:
                continue
            is_green_marker = green > 120 and green > red * 1.18 and green > blue * 1.18
            if not is_green_marker:
                out[x, y] = alpha
    bbox = mask.getbbox()
    if bbox is None:
        return alpha_bbox(image, pad)
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(image.width, x1 + pad),
        min(image.height, y1 + pad),
    )


def green_marker_center_x(image: Image.Image) -> float | None:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    xs = []
    for y in range(rgba.height // 2, rgba.height):
        for x in range(rgba.width):
            red, green, blue, alpha = pixels[x, y]
            if alpha <= 20:
                continue
            if green > 60 and green > red * 1.08 and green > blue * 1.08:
                xs.append(x)
    if not xs:
        return None
    return sum(xs) / len(xs)


def shift_rgba(image: Image.Image, dx: int) -> Image.Image:
    if dx == 0:
        return image
    shifted = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shifted.alpha_composite(image, (dx, 0))
    return shifted


def stabilize_marker_x_states(
    normalized: dict[str, list[Image.Image]], states: set[str]
) -> dict[str, float]:
    stabilized: dict[str, float] = {}
    for state in states:
        images = normalized.get(state)
        if not images:
            continue

        centers = [green_marker_center_x(image) for image in images]
        valid_centers = [center for center in centers if center is not None]
        if not valid_centers:
            print(f"No green marker detected for {state}; x stabilization skipped.")
            continue

        target_x = median(valid_centers)
        normalized[state] = [
            shift_rgba(image, round(target_x - center)) if center is not None else image
            for image, center in zip(images, centers)
        ]
        stabilized[state] = target_x
        print(f"Stabilized green marker x for {state}: target={target_x:.2f}")
    return stabilized


def resize_rgba(image: Image.Image, scale: float) -> Image.Image:
    if abs(scale - 1.0) < 0.001:
        return image
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def parse_state_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_state_int_map(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected state=count pair, got: {item}")
        state, count = item.split("=", 1)
        result[state.strip()] = max(0, int(count.strip()))
    return result


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def filter_outlier_items(state: str, items: list[dict], enabled: bool) -> list[dict]:
    if not enabled or len(items) < 5:
        return items
    body_heights = [item["body_height"] for item in items]
    body_widths = [item["body_width"] for item in items]
    areas = [item["body_height"] * item["body_width"] for item in items]
    median_height = median(body_heights)
    median_width = median(body_widths)
    median_area = median(areas)
    kept = []
    dropped = []
    for item in items:
        area = item["body_height"] * item["body_width"]
        too_tall = item["body_height"] > median_height * 1.35
        too_wide = item["body_width"] > median_width * 1.35
        too_large = area > median_area * 1.65
        if too_tall or too_wide or too_large:
            dropped.append(item["source_index"])
        else:
            kept.append(item)
    if dropped:
        print(f"Dropped outlier frames for {state}: {dropped}")
    return kept or items


def extract_png_frames(
    ffmpeg: str,
    video: Path,
    raw_dir: Path,
    fps: float,
    max_frames: int,
) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = raw_dir / "frame_%04d.png"
    cmd = [ffmpeg, "-y", "-c:v", "libvpx", "-i", str(video)]
    if fps > 0:
        cmd.extend(["-vf", f"fps={fps}"])
    if max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.extend(["-pix_fmt", "rgba", str(output_pattern)])
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return sorted(raw_dir.glob("frame_*.png"))


def normalize_states_by_frame(
    raw_by_state: dict[str, list[Image.Image]],
    pad: int,
    alpha_threshold: int,
    scale_mode: str,
    reference_state: str,
    target_height: int,
    preserve_y_states: set[str],
    drop_outlier_frames: bool,
) -> tuple[dict[str, list[Image.Image]], dict]:
    items_by_state: dict[str, list[dict]] = {}
    heights_by_state: dict[str, list[int]] = {}
    union_by_state: dict[str, tuple[int, int, int, int]] = {}

    for state, images in raw_by_state.items():
        items_by_state[state] = []
        heights_by_state[state] = []
        boxes = []
        for source_index, image in enumerate(images):
            image = clean_alpha(image, alpha_threshold)
            full_box = alpha_bbox(image, pad)
            body_box = marker_aware_body_bbox(image, pad)
            if full_box is None or body_box is None:
                continue
            boxes.append(full_box)
            fx0, fy0, fx1, fy1 = full_box
            bx0, by0, bx1, by1 = body_box
            crop = image.crop(full_box)
            body_center_x = ((bx0 + bx1) / 2) - fx0
            full_bottom_y = fy1 - fy0
            items_by_state[state].append(
                {
                    "crop": crop,
                    "body_center_x": body_center_x,
                    "bottom_y": full_bottom_y,
                    "body_height": by1 - by0,
                    "body_width": bx1 - bx0,
                    "full_box": full_box,
                    "source_index": source_index,
                }
            )
            heights_by_state[state].append(by1 - by0)
        if not items_by_state[state]:
            raise RuntimeError(f"No non-transparent pixels found for state: {state}")
        items_by_state[state] = filter_outlier_items(
            state,
            items_by_state[state],
            drop_outlier_frames and state not in preserve_y_states,
        )
        heights_by_state[state] = [item["body_height"] for item in items_by_state[state]]
        boxes = [item["full_box"] for item in items_by_state[state]]
        union_by_state[state] = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    if scale_mode == "state-height":
        median_heights = {
            state: statistics.median(heights) for state, heights in heights_by_state.items()
        }
        if target_height > 0:
            reference_height = target_height
        else:
            reference_height = median_heights.get(reference_state)
            if reference_height is None:
                reference_height = statistics.median(median_heights.values())
        scale_by_state = {
            state: reference_height / max(1, median_height)
            for state, median_height in median_heights.items()
        }
    else:
        reference_height = 0
        scale_by_state = {state: 1.0 for state in items_by_state}

    scaled_by_state: dict[str, list[dict]] = {}
    max_left = 0
    max_right = 0
    max_height = 0
    max_preserved_union_height = 0
    for state, items in items_by_state.items():
        scale = scale_by_state[state]
        ux0, uy0, ux1, uy1 = union_by_state[state]
        union_height = (uy1 - uy0) * scale
        if state in preserve_y_states:
            max_preserved_union_height = max(max_preserved_union_height, union_height)
        scaled_by_state[state] = []
        for item in items:
            crop = resize_rgba(item["crop"], scale)
            body_center_x = item["body_center_x"] * scale
            bottom_y = item["bottom_y"] * scale
            fx0, fy0, fx1, fy1 = item["full_box"]
            preserve_y_offset = (fy0 - uy0) * scale
            scaled_by_state[state].append(
                {
                    "crop": crop,
                    "body_center_x": body_center_x,
                    "bottom_y": bottom_y,
                    "preserve_y_offset": preserve_y_offset,
                    "preserve_y_union_height": union_height,
                }
            )
            max_left = max(max_left, body_center_x)
            max_right = max(max_right, crop.width - body_center_x)
            max_height = max(max_height, crop.height)

    canvas_w = round(max_left + max_right)
    canvas_h = round(max(max_height, max_preserved_union_height))
    baseline_y = canvas_h
    normalized: dict[str, list[Image.Image]] = {}
    for state, items in scaled_by_state.items():
        normalized[state] = []
        for item in items:
            crop = item["crop"]
            canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            x = round((canvas_w / 2) - item["body_center_x"])
            if state in preserve_y_states:
                y = round(
                    baseline_y
                    - item["preserve_y_union_height"]
                    + item["preserve_y_offset"]
                )
            else:
                y = round(baseline_y - item["bottom_y"])
            canvas.alpha_composite(crop, (x, y))
            normalized[state].append(canvas)
    return normalized, {
        "width": canvas_w,
        "height": canvas_h,
        "align_mode": "frame",
        "scale_mode": scale_mode,
        "reference_height": reference_height,
        "scale_by_state": scale_by_state,
        "preserve_y_states": sorted(preserve_y_states),
    }


def normalize_states_by_state_union(
    raw_by_state: dict[str, list[Image.Image]], pad: int
) -> tuple[dict[str, list[Image.Image]], dict]:
    state_crops: dict[str, list[Image.Image]] = {}
    crop_sizes = []

    for state, images in raw_by_state.items():
        boxes = [alpha_bbox(image, pad) for image in images]
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            raise RuntimeError(f"No non-transparent pixels found for state: {state}")
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        crops = [image.crop((x0, y0, x1, y1)) for image in images]
        state_crops[state] = crops
        crop_sizes.append((x1 - x0, y1 - y0))

    canvas_w = max(width for width, _ in crop_sizes)
    canvas_h = max(height for _, height in crop_sizes)
    normalized: dict[str, list[Image.Image]] = {}
    for state, crops in state_crops.items():
        normalized[state] = []
        for crop in crops:
            canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            x = (canvas_w - crop.width) // 2
            y = canvas_h - crop.height
            canvas.alpha_composite(crop, (x, y))
            normalized[state].append(canvas)
    return normalized, {"width": canvas_w, "height": canvas_h, "align_mode": "state"}


def to_manifest_path(output_dir: Path, path: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def write_outputs(output_dir: Path, state: str, images: list[Image.Image], duration_ms: int) -> dict:
    state_dir = output_dir / "states" / state
    frames_dir = state_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for index, image in enumerate(images):
        path = frames_dir / f"{state}_{index:03d}.png"
        image.save(path)
        frame_paths.append(to_manifest_path(output_dir, path))

    preview_frames = []
    for image in images:
        bg = checkerboard(image.size).convert("RGBA")
        bg.alpha_composite(image)
        preview_frames.append(bg.convert("P", palette=Image.Palette.ADAPTIVE))

    preview_path = state_dir / f"{state}_preview.gif"
    preview_frames[0].save(
        preview_path,
        save_all=True,
        append_images=preview_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    return {
        "state": state,
        "frame_count": len(images),
        "frames_dir": to_manifest_path(output_dir, frames_dir),
        "preview": to_manifest_path(output_dir, preview_path),
        "frames": frame_paths,
    }


def write_contact_sheet(output_dir: Path, states: dict[str, list[Image.Image]]) -> Path:
    qa_dir = output_dir / "_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    columns = 6
    cell_w = max(image.width for images in states.values() for image in images) + 24
    cell_h = max(image.height for images in states.values() for image in images) + 44
    sheet = Image.new("RGB", (columns * cell_w, len(states) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for row, (state, images) in enumerate(states.items()):
        sample_count = min(columns, len(images))
        indices = [0] if sample_count == 1 else [
            round(i * (len(images) - 1) / (sample_count - 1)) for i in range(sample_count)
        ]
        for col, index in enumerate(indices):
            image = images[index]
            bg = checkerboard(image.size).convert("RGBA")
            bg.alpha_composite(image)
            x = col * cell_w + (cell_w - image.width) // 2
            y = row * cell_h + 20
            sheet.paste(bg.convert("RGB"), (x, y))
            draw.text((col * cell_w + 8, row * cell_h + 4), f"{state} #{index}", fill=(0, 0, 0))

    path = qa_dir / "contact-sheet.png"
    sheet.save(path)
    return path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg(Path(args.deps_dir))
    videos = discover_webms(input_dir)
    if not videos:
        raise SystemExit(f"No WebM files found in {input_dir}")

    raw_root = output_dir / "_raw"
    raw_by_state: dict[str, list[Image.Image]] = {}
    flip_states = parse_state_set(args.flip_states)
    preserve_y_states = parse_state_set(args.preserve_y_states)
    stabilize_marker_x_states_arg = parse_state_set(args.stabilize_marker_x_states)
    trim_leading_frames = parse_state_int_map(args.trim_leading_frames)
    for video in videos:
        state = state_name(video)
        with tempfile.TemporaryDirectory(dir=output_dir) as temp:
            temp_dir = Path(temp)
            print(f"Importing {state} from {video.name}...")
            frame_paths = extract_png_frames(ffmpeg, video, temp_dir, args.fps, args.max_frames)
            if args.drop_last_frame and len(frame_paths) > 1:
                frame_paths = frame_paths[:-1]
            frames = [Image.open(path).convert("RGBA") for path in frame_paths]
            if state in flip_states:
                frames = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in frames]
            trim_count = trim_leading_frames.get(state, 0)
            if trim_count > 0 and trim_count < len(frames):
                frames = frames[trim_count:]
                print(f"Trimmed {trim_count} leading frames for {state}.")
            raw_by_state[state] = frames
            if args.keep_raw:
                keep_dir = raw_root / state
                keep_dir.mkdir(parents=True, exist_ok=True)
                for path in frame_paths:
                    shutil.copy2(path, keep_dir / path.name)

    if args.align_mode == "frame":
        normalized, canvas = normalize_states_by_frame(
            raw_by_state,
            args.pad,
            args.alpha_threshold,
            args.scale_mode,
            args.reference_state,
            args.target_height,
            preserve_y_states,
            args.drop_outlier_frames,
        )
    else:
        normalized, canvas = normalize_states_by_state_union(raw_by_state, args.pad)
    stabilized_marker_x = stabilize_marker_x_states(normalized, stabilize_marker_x_states_arg)
    duration_ms = int(1000 / args.fps) if args.fps > 0 else 33
    manifest = {
        "source_dir": ".",
        "importer": "transparent-webm",
        "canvas": canvas,
        "target_fps": args.fps,
        "align_mode": args.align_mode,
        "scale_mode": args.scale_mode,
        "flipped_states": sorted(flip_states),
        "drop_last_frame": args.drop_last_frame,
        "drop_outlier_frames": args.drop_outlier_frames,
        "preserve_y_states": sorted(preserve_y_states),
        "stabilize_marker_x_states": sorted(stabilize_marker_x_states_arg),
        "stabilized_marker_x": stabilized_marker_x,
        "trim_leading_frames": trim_leading_frames,
        "states": [write_outputs(output_dir, state, images, duration_ms) for state, images in normalized.items()],
    }
    contact_sheet = write_contact_sheet(output_dir, normalized)
    manifest["contact_sheet"] = to_manifest_path(output_dir, contact_sheet)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()
