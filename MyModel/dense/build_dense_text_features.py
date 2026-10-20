import argparse
import json
import os
import sys
from pathlib import Path

import torch

import clip


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def load_text_encoder(args):
    encoder = args.text_encoder.lower()
    if encoder == "clip":
        model, _ = clip.load(args.clip_model, device=args.device)
        return model, lambda texts: clip.tokenize(texts, truncate=True), "clip"

    if encoder == "longclip":
        repo = args.longclip_repo or os.getenv("LONGCLIP_REPO", "")
        checkpoint = args.longclip_checkpoint or os.getenv("LONGCLIP_CHECKPOINT", "")
        if not repo:
            raise RuntimeError("Long-CLIP requires --longclip-repo or LONGCLIP_REPO")
        if not checkpoint:
            raise RuntimeError("Long-CLIP requires --longclip-checkpoint or LONGCLIP_CHECKPOINT")
        repo_path = Path(repo).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"Long-CLIP repo not found: {repo_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Long-CLIP checkpoint not found: {checkpoint_path}")
        sys.path.insert(0, str(repo_path))
        from LongCLIP.model import longclip

        model, _ = longclip.load(str(checkpoint_path), device=args.device)
        return model, longclip.tokenize, "longclip"

    raise ValueError(f"unsupported text encoder: {args.text_encoder}")


def select_caption(payload, encoder):
    if not isinstance(payload, dict):
        return payload
    if encoder == "longclip":
        return payload.get("longclip_caption") or payload.get("dense_caption") or payload.get("clip_caption")
    return payload.get("clip_caption") or payload.get("dense_caption") or payload.get("longclip_caption")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", default="MyModel/dense/dense_captions_qwen.json", help="JSON mapping image keys to dense captions.")
    parser.add_argument("--output", default="./dense_text_features_longclip.pt", help="Output .pt path for CLIP text features.")
    parser.add_argument("--text-encoder", choices=["clip", "longclip"], default="longclip")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--longclip-repo", default="LongCLIP", help="Path to the official Long-CLIP repository.")
    parser.add_argument("--longclip-checkpoint", default="weights/longclip-B.pt", help="Path to longclip-B.pt or longclip-L.pt.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    captions_path = Path(args.captions)
    output_path = Path(args.output)

    with captions_path.open("r", encoding="utf-8") as f:
        captions = json.load(f)

    if not isinstance(captions, dict):
        raise ValueError("captions JSON must be an object mapping image keys to caption strings")

    model, tokenizer, encoder_name = load_text_encoder(args)
    model.eval()

    keys = list(captions.keys())
    features = {}
    with torch.no_grad():
        for batch_keys in batched(keys, args.batch_size):
            texts = [select_caption(captions[key], encoder_name) for key in batch_keys]
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ValueError("every caption entry must contain a non-empty caption string")
            tokens = tokenizer(texts).to(args.device)
            batch_features = model.encode_text(tokens).float()
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            batch_features = batch_features.cpu()
            for key, feature in zip(batch_keys, batch_features):
                features[key] = feature

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)


if __name__ == "__main__":
    main()
