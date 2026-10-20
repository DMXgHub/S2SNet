import argparse
import json
import os
import re
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COMPRESSION_SYSTEM_PROMPT = """You rewrite dense RGB-thermal scene descriptions into compact CLIP-ready captions for semantic segmentation.

Return one concise English sentence only.

Requirements:
- Keep the caption between 120 and 180 English words.
- Put scene type, time-of-day or illumination, and the main road/sidewalk layout near the beginning.
- Preserve visible people, cars, bicycles, and their approximate positions when mentioned.
- Preserve useful RGB-thermal differences, warm targets, boundaries, occlusions, and drivable regions when mentioned.
- Do not invent objects or details.

- Do not add explanations, bullet points, quotation marks, or labels."""


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


def count_words(text):
    return len(re.findall(r"[A-Za-z']+", text))


def request_clip_caption(api_key, model, dense_caption, timeout, max_retries, trust_env):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": dense_caption},
        ],
        "temperature": 0.2,
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
                text = text.strip('"').strip("'").strip()
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

    raise RuntimeError(f"compression request failed after {max_retries} retries: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str("./dense_captions_qwen.json"))
    parser.add_argument("--output", default=str("./dense_captions_qwen.json"))
    parser.add_argument("--model", default="qwen3-vl-32b-thinking")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--trust-env-proxy", action="store_true", help="Use proxy settings from the environment.")
    parser.add_argument("--limit", type=int, default=0, help="Use a positive number for a smoke test.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate clip_caption even if it already exists.")
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"caption file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        source_payload = json.load(f)
    if not isinstance(source_payload, dict):
        raise ValueError("input JSON must be an object mapping image keys to caption payloads")

    captions = load_existing(output_path) if output_path.exists() else {}
    if not captions:
        captions = source_payload

    keys = list(source_payload.keys())
    if args.limit > 0:
        keys = keys[:args.limit]

    total = len(keys)
    print(f"input: {input_path}", flush=True)
    print(f"output: {output_path}", flush=True)
    print(f"captions selected: {total}", flush=True)

    for index, key in enumerate(keys, start=1):
        source_item = source_payload[key]
        if not isinstance(source_item, dict):
            source_item = {"dense_caption": source_item}
        dense_caption = source_item.get("dense_caption")
        if not isinstance(dense_caption, str) or not dense_caption.strip():
            raise ValueError(f"{key} is missing a non-empty dense_caption")

        item = captions.setdefault(key, dict(source_item))
        if not args.overwrite and item.get("clip_caption"):
            print(f"[{index}/{total}] skip {key}", flush=True)
            continue

        print(f"[{index}/{total}] compress {key}", flush=True)
        clip_caption = request_clip_caption(
            api_key=api_key,
            model=args.model,
            dense_caption=dense_caption,
            timeout=args.timeout,
            max_retries=args.max_retries,
            trust_env=args.trust_env_proxy,
        )
        item.update(source_item)
        item["clip_caption"] = clip_caption
        item["clip_caption_word_count"] = count_words(clip_caption)
        save_json(output_path, captions)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
