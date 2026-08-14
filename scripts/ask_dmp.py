"""Interactive QA over an extracted DMP using a local Ollama model.

Usage:
  python scripts/ask_dmp.py --sample 1 --extractor dmponline --model gemma4:e4b

The script loads `data/output/1_extracted/<extractor>/sample{N}.json` (or
the structured JSON if present) and opens an interactive prompt. Type a
question and press Enter; an empty line exits.

Before running:
  1. Start Ollama with: ollama serve
  2. In another terminal, verify it's running: ollama ps
  3. Make sure your model is installed: ollama pull gemma4:e4b
"""
import argparse
import json
import time
import requests
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


def wait_for_ollama(host: str, timeout: int = 30) -> bool:
    """Wait for Ollama to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{host}/api/tags", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print()
    return False


def check_model_available(host: str, model: str) -> bool:
    """Check if the model is available in Ollama."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name") for m in models]
            return any(model in name for name in model_names for model in [model.split(":")[0]])
    except requests.exceptions.RequestException:
        pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--extractor", default="dmponline")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    ap.add_argument("--timeout", type=int, default=30, help="Seconds to wait for Ollama to start")
    args = ap.parse_args()

    print(f"\n=== DMP Question Answering ===")
    print(f"Sample: {args.sample}")
    print(f"Extractor: {args.extractor}")
    print(f"Model: {args.model}")
    print(f"Host: {args.host}")

    # Load document first
    print("\nLoading document...")
    try:
        blocks = load_document(args.extractor, args.sample)
        ctx = build_context(blocks)
        print(f"✓ Loaded {len(blocks)} blocks (~{len(ctx):,} chars)")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    # Wait for Ollama to be ready
    print("\nChecking Ollama server...")
    if not wait_for_ollama(args.host, args.timeout):
        print(f"✗ Ollama is not responding at {args.host}")
        print(f"\nStart Ollama in another terminal with:")
        print(f"  ollama serve")
        print(f"\nThen verify it's running:")
        print(f"  ollama ps")
        return

    print("✓ Ollama is running")

    # Check if model is available
    print(f"\nChecking for model '{args.model}'...")
    if not check_model_available(args.host, args.model):
        print(f"✗ Model not found. Install it with:")
        print(f"  ollama pull {args.model}")
        return

    print(f"✓ Model '{args.model}' is available")

    # Connect to model
    print("\nConnecting to model...")
    try:
        model = OllamaModel(args.model, args.host)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return

    system = (
        "You are a helpful assistant. Use the provided document context to answer "
        "the user's question. When appropriate, mention page numbers from the context."
    )

    print("\n" + "="*50)
    print("Ready! Start asking questions (empty line to exit)")
    print("="*50 + "\n")
    
    while True:
        try:
            q = input("Question: ").strip()
        except EOFError:
            break
        if not q:
            break
        
        prompt = ctx + "\n\nQuestion: " + q + "\nAnswer:"
        try:
            print("...", end=" ", flush=True)
            resp = model.complete(system, prompt)
            print("\n" + resp + "\n")
        except requests.exceptions.ConnectionError:
            print("✗ Connection lost to Ollama")
            print("Make sure Ollama is still running: ollama ps")
            break
        except requests.exceptions.Timeout:
            print("✗ Request timed out (model taking too long)")
        except Exception as exc:
            print(f"✗ Error: {exc}")
            break


if __name__ == "__main__":
    main()
