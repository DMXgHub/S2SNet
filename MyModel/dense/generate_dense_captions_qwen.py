import argparse
import base64
import json
import os
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CAPTION_PROMPT = """You are describing paired RGB and thermal images from the same autonomous-driving scene.

The first image is the RGB view and the second image is the thermal view.
Write one detailed English paragraph that describes the scene as faithfully as possible for semantic segmentation research.

Include:
- scene type, time-of-day cues, weather or illumination if visible
- road layout, sidewalk or curb structure, lane geometry, intersections, and drivable area
- visible people, cars, bicycles, traffic participants, and their approximate positions
- spatial relations among major objects
- differences between RGB appearance and thermal response when visible
- small objects, boundaries, occlusions, and background structures that may matter for segmentation

Rules:
- Do not invent objects that are not visible.
- Do not mention uncertainty about the task itself.
- Do not use bullet points.
- Produce a single rich paragraph only."""


def encode_image(path):
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def load_existing(path):
    candidates = [path, path.with_suffix(path.suffix + ".tmp")]
    valid_payloads = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            valid_payloads.append((candidate.stat().st_mtime, len(payload), payload))
        except (OSError, json.JSONDecodeError):
            continue
    if not valid_payloads:
        return {}
    return max(valid_payloads, key=lambda item: (item[1], item[0]))[2]


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    try:
        tmp_path.replace(path)
    except PermissionError:
        print(
            f"warning: could not replace {path.name}; keeping the latest checkpoint in {tmp_path.name}",
            flush=True,
        )


def iter_pairs(image_dir):
    for rgb_path in sorted(image_dir.glob("*_rgb.jpg")):
        thermal_path = rgb_path.with_name(rgb_path.name.replace("_rgb.jpg", "_th.jpg"))
        if thermal_path.exists():
            yield rgb_path, thermal_path


def request_caption(api_key, model, rgb_path, thermal_path, timeout, max_retries, trust_env):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(rgb_path)}},
                    {"type": "image_url", "image_url": {"url": encode_image(thermal_path)}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    session = requests.Session()
    session.trust_env = trust_env

    last_error = None
    for attempt in range(max_retries):
        try:
            response = session.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 200:
                body = response.json()
                text = body["choices"][0]["message"]["content"].strip()
                if text:
                    return text
                raise RuntimeError(f"empty content: {body}")

            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = f"{response.status_code}: {response.text[:500]}"
                time.sleep(min(2 ** attempt, 30))
                continue

            raise RuntimeError(f"{response.status_code}: {response.text[:1000]}")
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"caption request failed after {max_retries} retries: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(PROJECT_ROOT / "seperated_images"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "dense_captions_qwen.json"))
    parser.add_argument("--model", default="qwen3-vl-32b-thinking")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--trust-env-proxy", action="store_true", help="Use proxy settings from the environment.")
    parser.add_argument("--limit", type=int, default=0, help="Use a positive number for a smoke test.")
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    image_dir = Path(args.image_dir)
    output_path = Path(args.output)
    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    captions = load_existing(output_path)
    pairs = list(iter_pairs(image_dir))
    if args.limit > 0:
        pairs = pairs[:args.limit]

    total = len(pairs)
    print(f"image_dir: {image_dir}", flush=True)
    print(f"output: {output_path}", flush=True)
    print(f"paired images found: {total}", flush=True)
    if total == 0:
        raise RuntimeError("no paired '*_rgb.jpg' and '*_th.jpg' images were found")

    for index, (rgb_path, thermal_path) in enumerate(pairs, start=1):
        key = rgb_path.name
        if key in captions and captions[key].get("dense_caption"):
            print(f"[{index}/{total}] skip {key}", flush=True)
            continue

        print(f"[{index}/{total}] caption {key}", flush=True)
        caption = request_caption(
            api_key=api_key,
            model=args.model,
            rgb_path=rgb_path,
            thermal_path=thermal_path,
            timeout=args.timeout,
            max_retries=args.max_retries,
            trust_env=args.trust_env_proxy,
        )
        captions[key] = {
            "rgb_image": rgb_path.name,
            "thermal_image": thermal_path.name,
            "dense_caption": caption,
        }
        save_json(output_path, captions)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
