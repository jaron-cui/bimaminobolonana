# encoder/scripts/model_stats.py
import os
import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
import torch

from encoder import build_encoder

CFG_DIR = REPO_ROOT / "configs"

def load_cfg(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = CFG_DIR / p
    with open(p, "r") as f:
        return yaml.safe_load(f)


def count_params(m, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in m.parameters() if getattr(p, "requires_grad", False))
    return sum(p.numel() for p in m.parameters())


def summarize_cfg(cfg: dict) -> dict:
    """Normalize keys we care about for the table."""
    return {
        "name": cfg.get("name"),
        "model_name": cfg.get("model_name"),   # CLIP
        "variant": cfg.get("variant"),         # Pri3D
        "pretrained": cfg.get("pretrained"),
        "ckpt_path": cfg.get("ckpt_path") or cfg.get("checkpoint_path"),
        "freeze": cfg.get("freeze"),
        "train_last_n_blocks": cfg.get("train_last_n_blocks"),
        "out_dim": cfg.get("out_dim"),
        "fuse": cfg.get("fuse"),
    }


def _count_params_module(m: torch.nn.Module | None) -> int:
    if m is None:
        return 0
    return sum(p.numel() for p in m.parameters())


def _clip_visual_modules(enc):
    """
    Return (visual, trunk, heads, text) if this is a CLIP encoder (open-clip style),
    otherwise (None, None, None, None).
    """
    model = getattr(enc, "model", None)
    vis = getattr(model, "visual", None)
    if vis is None or not isinstance(vis, torch.nn.Module):
        return None, None, None, None

    # ResNet-based CLIP in open-clip exposes .trunk for the conv body.
    trunk = getattr(vis, "trunk", None)
    if trunk is not None and not isinstance(trunk, torch.nn.Module):
        trunk = None

    heads = []
    # Common RN heads in open-clip
    if hasattr(vis, "attnpool") and isinstance(vis.attnpool, torch.nn.Module):
        heads.append(vis.attnpool)
    if hasattr(vis, "proj") and isinstance(vis.proj, torch.nn.Module):
        heads.append(vis.proj)

    # Text tower (best-effort; may vary by open-clip version)
    text = None
    if model is not None:
        # Try the transformer block
        text_main = getattr(model, "transformer", None)
        # Token embedding (some models have .token_embedding as a Module; others as Parameter)
        tok = getattr(model, "token_embedding", None)
        if isinstance(text_main, torch.nn.Module):
            # If token_embedding is a Parameter, wrap a tiny module to count it
            class _TokWrap(torch.nn.Module):
                def __init__(self, p): super().__init__(); self.w = p
            tok_mod = tok if isinstance(tok, torch.nn.Module) else (_TokWrap(tok) if isinstance(tok, torch.nn.Parameter) else None)

            # Combine for counting
            class _TextCombo(torch.nn.Module):
                def __init__(self, a, b): super().__init__();
                # NOTE: we register as submodules only if they are Modules
            text = torch.nn.Module()
            if isinstance(text_main, torch.nn.Module): text.add_module("main", text_main)
            if isinstance(tok_mod, torch.nn.Module):  text.add_module("tok", tok_mod)

    return vis, trunk, heads, text


# replace clip_param_breakdown with this improved version
def clip_param_breakdown(enc) -> dict:
    """Return dict with CLIP visual/text breakdown; robustly derive trunk for RN models."""
    d = {}
    vis, trunk, heads, text = _clip_visual_modules(enc)
    if vis is not None:
        vis_total = _count_params_module(vis)
        d["params_visual_total"] = vis_total

        heads_total = 0
        if heads:
            heads_total = sum(_count_params_module(h) for h in heads)
            d["params_visual_heads"] = heads_total

        # Prefer an explicit trunk module if present
        if trunk is not None:
            d["params_visual_trunk"] = _count_params_module(trunk)
        else:
            # Fallback: estimate trunk = visual_total - heads (works for RN50)
            d["params_visual_trunk"] = vis_total - heads_total

    if text is not None:
        d["params_text_tower"] = _count_params_module(text)

    return d



def row_for(cfg_path: Path) -> dict:
    cfg = load_cfg(cfg_path)
    enc = build_encoder(cfg)

    total = count_params(enc)
    trainable = count_params(enc, trainable_only=True)
    info = summarize_cfg(cfg)

    # backbone label
    try:
        backbone = getattr(enc, "model_name", None) or info.get("model_name") or info.get("variant")
    except Exception:
        backbone = info.get("model_name") or info.get("variant") or "n/a"

    row = {
        "config": cfg_path.name,
        "encoder": info["name"],
        "backbone": backbone or "n/a",
        "pretrained": info["pretrained"],
        "freeze": info["freeze"],
        "train_last_n_blocks": info["train_last_n_blocks"],
        "out_dim": info["out_dim"],
        "fuse": info["fuse"],
        "ckpt_path": (Path(info["ckpt_path"]).name if info.get("ckpt_path") else None),
        "params_total": total,
        "params_trainable": trainable,
    }
    # Add CLIP breakdown if applicable
    row.update(clip_param_breakdown(enc))
    return row


def print_table(rows: list[dict], headers: list[str]) -> None:
    col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    def fmt_row(r): return " | ".join(str(r.get(h, "")).ljust(col_widths[h]) for h in headers)
    print("\n== Encoder Parameter & Hyperparameter Summary ==\n")
    print(fmt_row({h: h for h in headers}))
    print("-+-".join("-" * col_widths[h] for h in headers))
    for r in rows:
        print(fmt_row(r))


def save_markdown_and_json(rows: list[dict], headers: list[str], out_md: Path, out_json: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        md_lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    out_md.write_text("\n".join(md_lines))
    out_json.write_text(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Print encoder parameter counts & hyperparameters.")
    ap.add_argument(
        "--configs",
        nargs="*",
        default=[
            "encoder_clip_b32.yaml",
            "encoder_clip_b32_openai.yaml",
            "encoder_clip_rn50_openai.yaml",
            "encoder_pri3d_random.yaml",
            "encoder_pri3d_pretrained.yaml",
        ],
        help="List of config file names or paths (relative to repo configs/ by default).",
    )
    ap.add_argument("--out-md", type=str, default="runs/model_stats.md", help="Markdown table output path")
    ap.add_argument("--out-json", type=str, default="runs/model_stats.json", help="JSON output path")
    args = ap.parse_args()

    rows: list[dict] = []
    for c in args.configs:
        p = Path(c)
        p_full = p if p.is_absolute() else (CFG_DIR / p)
        if not p_full.exists():
            print(f"[skip] {p} not found")
            continue
        try:
            rows.append(row_for(p))
        except Exception as e:
            print(f"[error] {p}: {e}")

    if not rows:
        print("No rows generated (no configs found?).")
        return

    headers = [
        "config", "encoder", "backbone", "pretrained", "freeze", "train_last_n_blocks",
        "out_dim", "fuse", "ckpt_path", "params_total", "params_trainable",
        "params_visual_total", "params_visual_trunk", "params_visual_heads", "params_text_tower",
    ]

    print_table(rows, headers)
    out_md = REPO_ROOT / args.out_md
    out_json = REPO_ROOT / args.out_json
    save_markdown_and_json(rows, headers, out_md, out_json)
    print(f"\nSaved: {out_md.relative_to(REPO_ROOT)}")
    print(f"Saved: {out_json.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
