"""Interactive QA over an extracted DMP using a local Ollama model.

Usage:
  python scripts/ask_dmp.py --sample 1 --extractor dmponline --model gemma4:e4b

The script loads `data/output/1_extracted/<extractor>/sample{N}.json` (or
the structured JSON if present) and opens an interactive prompt. Type a
question and press Enter; an empty line exits.
"""
import argparse
import json
from pathlib import Path

from dmpbridge.core import paths
from dmpbridge.models.ollama import OllamaModel


def load_document(extractor: str, sample: int) -> list[dict]:
    # Prefer structured JSON if available
    tag = paths.make_tag("gemma4:e4b", extractor)
    struct_path = paths.structured_path(tag, sample)
    if struct_path.exists():
        try:
            return json.loads(struct_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    extracted = paths.extracted_path(extractor, sample)
    if not extracted.exists():
        raise FileNotFoundError(f"Extracted JSON not found: {extracted}")
    return json.loads(extracted.read_text(encoding="utf-8"))


def build_context(blocks: list[dict], max_chars: int = 60_000) -> str:
    # Join blocks with page markers; trim to max_chars from the end
    parts = []
    for b in blocks:
        page = b.get("page")
        text = b.get("text", "").strip()
        if not text:
            continue
        parts.append(f"[page {page}] {text}")
    ctx = "\n\n".join(parts)
    if len(ctx) > max_chars:
        # keep the tail which often contains later sections
        ctx = ctx[-max_chars:]
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--extractor", default="dmponline")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--host", default=None, help="Ollama host URL (e.g. http://localhost:11434)")
    args = ap.parse_args()

    blocks = load_document(args.extractor, args.sample)
    ctx = build_context(blocks)

    host = args.host or None
    # OllamaModel expects a host string; if None it will use default from config
    model = OllamaModel(args.model, host or "http://localhost:11434")

    system = (
        "You are a helpful assistant. Use the provided document context to answer "
        "the user's question. When appropriate, mention page numbers from the context."
    )

    print("Loaded document — start asking questions (empty line to exit).")
    while True:
        try:
            q = input("Question: ").strip()
        except EOFError:
            break
        if not q:
            break
        prompt = ctx + "\n\nQuestion: " + q + "\nAnswer:"
        try:
            resp = model.complete(system, prompt)
        except Exception as exc:
            print("Error querying Ollama:", exc)
            break
        print("\n" + resp + "\n")


if __name__ == "__main__":
    main()
