"""
check_coreml_ops.py
--------------------
Ubuntu-friendly CoreML EP compatibility checker.

Fetches the official ORT supported-op list from the onnxruntime gh-pages
branch (always up-to-date), cross-checks every node in your ONNX model,
prints a colour-coded terminal report, and writes a detailed Markdown report.

Usage:
    pip install onnx onnxruntime requests beautifulsoup4
    python check_coreml_ops.py --model ./deployed_model/model.onnx
    python check_coreml_ops.py --model ./deployed_model/model.onnx --format NeuralNetwork
    python check_coreml_ops.py --model ./deployed_model/model.onnx --out ./report.md
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import onnx
import onnx.shape_inference
import onnxruntime as ort


# ── ANSI colours ─────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Fetch supported-op list
# ═══════════════════════════════════════════════════════════════════════════════

# The CoreML EP docs live on the gh-pages branch.
# This is the single authoritative source — always current, no version pinning needed.
_GH_PAGES_URL = (
    "https://raw.githubusercontent.com/microsoft/onnxruntime"
    "/gh-pages/docs/execution-providers/CoreML-ExecutionProvider.md"
)


def fetch_coreml_supported_ops() -> tuple[dict[str, dict[str, str]], str]:
    """
    Fetches and parses the ORT CoreML EP documentation.

    Returns:
        supported  — {"NeuralNetwork": {op: note, ...}, "MLProgram": {op: note, ...}}
        source_url — human-readable string of which source was used

    Tries two sources:
        1. gh-pages raw markdown  (most accurate, version-independent)
        2. Built-in hard-coded baseline (offline fallback, ORT gh-pages snapshot)

    NOTE: We intentionally do NOT try the rendered HTML (onnxruntime.ai) because
    that page returns 403 in many CI/server environments and its HTML structure
    is fragile to parse reliably across Jekyll rebuilds.
    """
    supported: dict[str, dict[str, str]] = {"NeuralNetwork": {}, "MLProgram": {}}

    # --- Layer 1: gh-pages raw markdown ---
    try:
        r = requests.get(_GH_PAGES_URL, timeout=10)
        r.raise_for_status()
        _parse_markdown(r.text, supported)
        # Validate: both sections must have ops; a single-section result means
        # the page structure changed and we should fall back.
        if supported["NeuralNetwork"] and supported["MLProgram"]:
            # Fix a known docs typo: "ai.onnx.ReduceSum" should be "ReduceSum"
            _fix_malformed_op_names(supported)
            return supported, _GH_PAGES_URL
        else:
            print(f"[!] gh-pages parse incomplete "
                  f"(NN={len(supported['NeuralNetwork'])}, "
                  f"ML={len(supported['MLProgram'])}). Using built-in fallback.")
    except Exception as e:
        print(f"[!] Could not fetch gh-pages doc ({e}). Using built-in fallback.")

    # --- Layer 2: built-in baseline ---
    return _builtin_fallback(), "built-in baseline (gh-pages snapshot, may lag latest ORT)"


def _parse_markdown(md: str, supported: dict[str, dict[str, str]]) -> None:
    """
    Parses the two markdown tables under '### NeuralNetwork' and '### MLProgram'.
    Handles rows of the form:
        | ai.onnx:Add          | optional note |
        | ai.onnx.ReduceSum    | ...           |   ← docs typo (period instead of colon)
    """
    current = None
    for line in md.splitlines():
        if "### NeuralNetwork" in line:
            current = "NeuralNetwork"
        elif "### MLProgram" in line:
            current = "MLProgram"
        elif current and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not cells:
                continue
            raw = cells[0]
            # Match both "ai.onnx:OpName" and the typo "ai.onnx.OpName"
            if raw.startswith("ai.onnx"):
                # Extract the last token after ':' or '.'
                op = raw.replace("ai.onnx:", "").replace("ai.onnx.", "").strip()
                if op and op != "---":   # skip header/separator rows
                    note = cells[1].strip() if len(cells) > 1 else ""
                    supported[current][op] = note


def _fix_malformed_op_names(supported: dict[str, dict[str, str]]) -> None:
    """Remove any op names that are clearly malformed (contain spaces, slashes, etc.)."""
    for section in supported:
        bad = [op for op in supported[section] if " " in op or "/" in op or not op]
        for op in bad:
            del supported[section][op]


def _builtin_fallback() -> dict[str, dict[str, str]]:
    """
    Hard-coded snapshot from onnxruntime gh-pages (June 2025).
    NeuralNetwork: 38 ops.  MLProgram: 45 ops.
    BatchNormalization is NeuralNetwork-ONLY — intentionally absent from MLProgram.
    """
    _nn: dict[str, str] = {
        "Add": "",
        "ArgMax": "",
        "AveragePool": "Only 2D Pool is supported.",
        "BatchNormalization": "",                          # NeuralNetwork only
        "Cast": "",
        "Clip": "",
        "Concat": "",
        "Conv": "Only 1D/2D Conv. Weights and bias should be constant.",
        "DepthToSpace": "Only DCR mode.",
        "Div": "",
        "Flatten": "",
        "Gather": "Input indices with scalar value not supported.",
        "Gemm": "Input B should be constant.",
        "GlobalAveragePool": "Only 2D Pool.",
        "GlobalMaxPool": "Only 2D Pool.",
        "LRN": "",
        "LeakyRelu": "",
        "MatMul": "Input B should be constant.",
        "MaxPool": "Only 2D Pool.",
        "Mul": "",
        "PRelu": "Slope must be constant with shape [C,1,1] or 1 element.",
        "Pad": "Constant mode only. Last two dim padding. pads/constant_value must be constant.",
        "Pow": "Only supports fp32 inputs.",
        "Reciprocal": "",
        "ReduceSum": "",
        "Relu": "",
        "Reshape": "",
        "Resize": "nearest or bilinear only.",
        "Shape": "",
        "Sigmoid": "",
        "Slice": "",
        "Softmax": "",
        "Split": "",
        "Sqrt": "",
        "Squeeze": "",
        "Sub": "",
        "Tanh": "",
        "Transpose": "",
    }
    _ml: dict[str, str] = {
        # MLProgram: BatchNormalization intentionally NOT included (unsupported)
        "Add": "",
        "Argmax": "",
        "AveragePool": "Only 2D Pool.",
        "Cast": "",
        "Clip": "",
        "Concat": "",
        "Conv": "Weights and bias should be constant.",
        "ConvTranspose": "Weights and bias should be constant.",
        "DepthToSpace": "",
        "Div": "",
        "Erf": "",
        "Gelu": "",
        "Gemm": "Input B should be constant.",
        "GlobalAveragePool": "Only 2D Pool.",
        "GlobalMaxPool": "Only 2D Pool.",
        "GridSample": "",
        "GroupNormalization": "",
        "InstanceNormalization": "",
        "LayerNormalization": "",
        "LeakyRelu": "",
        "MatMul": "Input B should be constant.",
        "Max": "",
        "MaxPool": "Only 2D Pool.",
        "Mul": "",
        "PRelu": "Slope must be constant.",
        "Pow": "",
        "Reciprocal": "",
        "ReduceMax": "",
        "ReduceMean": "",
        "ReduceSum": "",
        "Relu": "",
        "Reshape": "",
        "Resize": "nearest or bilinear only.",
        "Round": "",
        "Shape": "",
        "Sigmoid": "",
        "Slice": "",
        "Softmax": "",
        "Split": "",
        "Sqrt": "",
        "Squeeze": "",
        "Sub": "",
        "Tanh": "",
        "Transpose": "",
        "Unsqueeze": "",
    }
    return {"NeuralNetwork": _nn, "MLProgram": _ml}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Inspect the ONNX model
# ═══════════════════════════════════════════════════════════════════════════════

def extract_model_ops(model_path: str):
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)
    ops: dict[str, list[str]] = defaultdict(list)
    for node in model.graph.node:
        name = node.name or f"<{node.op_type}>"
        ops[node.op_type].append(name)
    return dict(ops), model


def get_model_meta(model_path: str, model: onnx.ModelProto) -> dict:
    path    = Path(model_path)
    size_mb = path.stat().st_size / 1024 / 1024
    return {
        "path":     str(path.resolve()),
        "filename": path.name,
        "size_mb":  f"{size_mb:.2f}",
        "opset":    model.opset_import[0].version if model.opset_import else "?",
        "ir":       model.ir_version,
        "nodes":    sum(1 for _ in model.graph.node),
        "inputs":   [i.name for i in model.graph.input],
        "outputs":  [o.name for o in model.graph.output],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Core analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse(model_path: str, model_format: str) -> dict:
    ort_ver             = ort.__version__
    supported_ops, src  = fetch_coreml_supported_ops()
    target_set          = supported_ops.get(model_format, {})

    model_ops, model_proto = extract_model_ops(model_path)
    meta                   = get_model_meta(model_path, model_proto)

    unsupported: dict[str, dict] = {}
    supported:   dict[str, dict] = {}
    custom:      dict[str, list] = {}

    for op, nodes in model_ops.items():
        if op.startswith("com.microsoft") or op.startswith("org."):
            custom[op] = nodes
        elif op in target_set:
            supported[op] = {"nodes": nodes, "note": target_set[op]}
        else:
            unsupported[op] = {"nodes": nodes}

    total   = sum(len(v) for v in model_ops.values())
    covered = sum(len(v["nodes"]) for v in supported.values())

    return {
        "ort_version":   ort_ver,
        "model_format":  model_format,
        "source":        src,
        "meta":          meta,
        "unsupported":   unsupported,
        "supported":     supported,
        "custom":        custom,
        "total_nodes":   total,
        "covered_nodes": covered,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Terminal report
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(covered: int, total: int, width: int = 30) -> str:
    pct    = 100 * covered / total if total else 0
    filled = int(width * pct / 100)
    bar    = "█" * filled + "░" * (width - filled)
    colour = GREEN if pct >= 90 else YELLOW if pct >= 60 else RED
    return f"{colour}{bar}{RESET} {pct:.1f}%"


def print_terminal_report(result: dict, out_path: str) -> None:
    u  = result["unsupported"]
    s  = result["supported"]
    c  = result["custom"]
    m  = result["meta"]
    tn = result["total_nodes"]
    cv = result["covered_nodes"]

    W = 68  # report width

    print()
    print(f"{BOLD}{'═'*W}{RESET}")
    print(f"{BOLD}  CoreML EP Compatibility Report{RESET}")
    print(f"{'═'*W}")
    print(f"  Model    : {m['filename']}  ({m['size_mb']} MB)")
    print(f"  Opset    : {m['opset']}   IR version: {m['ir']}")
    print(f"  ORT ver  : {result['ort_version']}")
    print(f"  Format   : {result['model_format']}")
    print(f"  Op source: {result['source']}")
    print(f"{'─'*W}")
    print(f"  Coverage : {_bar(cv, tn)}  ({cv}/{tn} nodes)")
    print(f"{'═'*W}\n")

    # ── Unsupported table ──────────────────────────────────────────────────────
    if u:
        un_nodes = sum(len(v["nodes"]) for v in u.values())
        print(f"{RED}{BOLD}  ❌  UNSUPPORTED OPERATORS — {len(u)} op type(s), {un_nodes} node(s) total{RESET}\n")

        # Column widths
        cw = [30, 7, 7, 28]   # Op | Count | % of total | Example nodes
        hdr = (f"  {'Operator':<{cw[0]}}  {'Nodes':>{cw[1]}}  {'% Total':>{cw[2]}}  "
               f"{'Example Node Names':<{cw[3]}}")
        sep = f"  {'─'*cw[0]}  {'─'*cw[1]}  {'─'*cw[2]}  {'─'*cw[3]}"
        print(hdr)
        print(sep)
        for op in sorted(u.keys()):
            nodes   = u[op]["nodes"]
            count   = len(nodes)
            pct     = 100 * count / tn if tn else 0
            example = ", ".join(nodes[:2])
            if len(nodes) > 2:
                example += f" (+{len(nodes)-2} more)"
            if len(example) > cw[3]:
                example = example[:cw[3]-1] + "…"
            print(f"  {RED}{op:<{cw[0]}}{RESET}  {count:>{cw[1]}}  {pct:>{cw[2]-1}.1f}%  {example}")
        print()

        # Per-op fix hints
        hints_shown = False
        for op in sorted(u.keys()):
            hint = _op_fix_hint(op, result["model_format"])
            if hint:
                if not hints_shown:
                    print(f"{CYAN}  💡  Fix hints:{RESET}")
                    hints_shown = True
                print(f"  • {op}: {hint}")
        if hints_shown:
            print()
    else:
        print(f"{GREEN}{BOLD}  ✅  All operators are supported by CoreML EP ({result['model_format']}).{RESET}\n")

    # ── Custom / contrib ops ───────────────────────────────────────────────────
    if c:
        print(f"{YELLOW}{BOLD}  ⚠️   CUSTOM / CONTRIB OPS — {len(c)} type(s), always CPU fallback{RESET}\n")
        for op in sorted(c.keys()):
            print(f"  {YELLOW}{op:<32}{RESET}  {len(c[op])} node(s)")
        print()

    # ── Supported op types (compact) ──────────────────────────────────────────
    if s:
        print(f"{GREEN}  ✅  Supported by CoreML EP ({len(s)} op types):{RESET}")
        line, indent = "     ", "     "
        for w in sorted(s.keys()):
            token = w + ", "
            if len(line) + len(token) > 72:
                print(line.rstrip(", "))
                line = indent
            line += token
        print(line.rstrip(", "))
        print()

    # ── General recommendations ────────────────────────────────────────────────
    if u or c:
        print(f"{CYAN}  💡  General recommendations:{RESET}")
        if result["model_format"] == "NeuralNetwork" and u:
            print("  • Try --format MLProgram (iOS 15+ / macOS 12+) — it has a wider op set.")
        print("  • Unsupported nodes fall back to CPU EP automatically — model still runs,")
        print("    but those nodes won't use ANE/GPU acceleration.")
        print()

    # ── Footer — show the actual output path ───────────────────────────────────
    print(f"  📄  Markdown report → {out_path}")
    print(f"{'═'*W}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Markdown report
# ═══════════════════════════════════════════════════════════════════════════════

def write_markdown_report(result: dict, out_path: str) -> None:
    u   = result["unsupported"]
    s   = result["supported"]
    c   = result["custom"]
    m   = result["meta"]
    tn  = result["total_nodes"]
    cv  = result["covered_nodes"]
    pct = 100 * cv / tn if tn else 0
    fmt     = result["model_format"]
    ort_ver = result["ort_version"]

    status_icon = "✅" if not u and not c else ("❌" if u else "⚠️")

    lines: list[str] = []
    a = lines.append

    # ── Title & metadata ───────────────────────────────────────────────────────
    a(f"# {status_icon} CoreML EP Compatibility Report")
    a("")
    a("| | |")
    a("|:--|:--|")
    a(f"| **Generated** | {result['timestamp']} |")
    a(f"| **ONNX Runtime** | {ort_ver} |")
    a(f"| **CoreML Format** | {fmt} |")
    a(f"| **Op list source** | {result['source']} |")
    a("")

    # ── Model info ─────────────────────────────────────────────────────────────
    a("## 📦 Model Information")
    a("")
    a("| Field | Value |")
    a("|:------|:------|")
    a(f"| File | `{m['filename']}` |")
    a(f"| Path | `{m['path']}` |")
    a(f"| Size | {m['size_mb']} MB |")
    a(f"| ONNX Opset | {m['opset']} |")
    a(f"| IR Version | {m['ir']} |")
    a(f"| Total Nodes | {tn} |")
    a(f"| Inputs | `{'`, `'.join(m['inputs'])}` |")
    a(f"| Outputs | `{'`, `'.join(m['outputs'])}` |")
    a("")

    # ── Coverage summary ───────────────────────────────────────────────────────
    a("## 📊 Coverage Summary")
    a("")
    bar_filled = int(pct / 5)
    bar_str    = "█" * bar_filled + "░" * (20 - bar_filled)
    a("```")
    a(f"  CoreML ({fmt}) coverage")
    a(f"  {bar_str}  {pct:.1f}%  ({cv} / {tn} nodes)")
    a("```")
    a("")
    un_nodes = sum(len(v["nodes"]) for v in u.values())
    cu_nodes = sum(len(v)          for v in c.values())
    a("| Status | Op Types | Nodes |")
    a("|:-------|:--------:|------:|")
    a(f"| ✅ Supported by CoreML EP | {len(s)} | {cv} |")
    a(f"| ❌ Unsupported — CPU fallback | {len(u)} | {un_nodes} |")
    a(f"| ⚠️ Custom / contrib ops | {len(c)} | {cu_nodes} |")
    a(f"| **Total** | **{len(s)+len(u)+len(c)}** | **{tn}** |")
    a("")

    # ── Unsupported ops ────────────────────────────────────────────────────────
    if u:
        a("## ❌ Unsupported Operators")
        a("")
        a(f"> **{len(u)} operator type(s)** accounting for **{un_nodes} node(s)** "
          f"({100*un_nodes/tn:.1f}% of the graph) will fall back to CPU EP on macOS.")
        a("")

        # ── Quick summary table ────────────────────────────────────────────────
        a("### Quick Summary")
        a("")
        a("| # | Operator | Node Count | % of Graph | Suggested Fix |")
        a("|--:|:---------|:----------:|:----------:|:--------------|")
        for i, op in enumerate(sorted(u.keys()), 1):
            count   = len(u[op]["nodes"])
            pct_op  = 100 * count / tn if tn else 0
            fix     = _op_fix_hint_short(op, fmt)
            a(f"| {i} | `{op}` | {count} | {pct_op:.1f}% | {fix} |")
        a("")

        # ── Detailed breakdown — one card per op ──────────────────────────────
        a("### Detailed Breakdown")
        a("")
        for op in sorted(u.keys()):
            nodes = u[op]["nodes"]
            count = len(nodes)
            pct_op = 100 * count / tn if tn else 0
            a(f"#### `{op}`")
            a("")
            a(f"| Property | Value |")
            a(f"|:---------|:------|")
            a(f"| Node count | **{count}** ({pct_op:.1f}% of graph) |")
            a(f"| CoreML format | {fmt} |")
            a(f"| Status | ❌ Not supported — falls back to CPU EP |")
            hint = _op_fix_hint(op, fmt)
            if hint:
                a(f"| Fix | {hint} |")
            a("")
            # All affected node names
            if count <= 20:
                a("**Affected nodes:**")
                a("")
                a("| # | Node Name |")
                a("|--:|:----------|")
                for j, n in enumerate(nodes, 1):
                    a(f"| {j} | `{n}` |")
            else:
                # Too many to list individually — show first 10 + count
                a("**Affected nodes** (first 10 shown):")
                a("")
                a("| # | Node Name |")
                a("|--:|:----------|")
                for j, n in enumerate(nodes[:10], 1):
                    a(f"| {j} | `{n}` |")
                a(f"| … | *…and {count - 10} more* |")
            a("")
    else:
        a("## ✅ Unsupported Operators")
        a("")
        a(f"> **None.** All {tn} nodes in this model are supported by the "
          f"CoreML EP ({fmt}).")
        a("")

    # ── Custom / contrib ops ───────────────────────────────────────────────────
    if c:
        a("## ⚠️ Custom / Contrib Operators")
        a("")
        a("> These operators use a non-standard domain (`com.microsoft` or similar).")
        a("> CoreML EP never handles them — they always run on CPU EP regardless of format.")
        a("")
        a("| Operator | Domain | Node Count |")
        a("|:---------|:-------|:----------:|")
        for op in sorted(c.keys()):
            domain = ".".join(op.split(".")[:2]) if "." in op else "custom"
            a(f"| `{op}` | `{domain}` | {len(c[op])} |")
        a("")

    # ── Supported ops ──────────────────────────────────────────────────────────
    a("## ✅ Supported Operators")
    a("")
    a(f"> **{len(s)} operator type(s)** are handled by the CoreML EP ({fmt}).")
    a("")
    a("| Operator | Node Count | Constraints / Notes |")
    a("|:---------|:----------:|:--------------------|")
    for op in sorted(s.keys()):
        count = len(s[op]["nodes"])
        note  = s[op]["note"] or "—"
        a(f"| `{op}` | {count} | {note} |")
    a("")

    # ── Recommendations ────────────────────────────────────────────────────────
    if u or c:
        a("## 💡 Recommendations")
        a("")
        recs: list[str] = []
        if fmt == "NeuralNetwork" and u:
            recs.append(
                "**Switch to MLProgram format** (`--format MLProgram`). "
                "It targets iOS 15+ / macOS 12+ and supports a wider op set "
                "(e.g. adds `ConvTranspose`, `LayerNormalization`, `Gelu`, `GridSample`)."
            )
        if "BatchNormalization" in u and fmt == "MLProgram":
            recs.append(
                "**`BatchNormalization`** is not supported in MLProgram. "
                "Fuse it into the preceding `Conv` weight via `torch.nn.utils.fuse_conv_bn_eval()` "
                "before export, or switch to `NeuralNetwork` format where it is supported."
            )
        if "Resize" in u:
            recs.append(
                "**`Resize`** — set `mode='nearest'` or `'linear'` in "
                "`torch.nn.Upsample` / `F.interpolate`. Cubic is unsupported."
            )
        if "Pad" in u:
            recs.append(
                "**`Pad`** — only `constant` mode is supported. "
                "Replace `reflect` or `edge` padding before export."
            )
        if "LSTM" in u or "GRU" in u:
            recs.append(
                "**`LSTM` / `GRU`** — bidirectional is unsupported. "
                "Use forward-only or split into two unidirectional layers."
            )
        recs.append(
            "Unsupported nodes are **automatically partitioned to CPU EP** at runtime "
            "— the model still runs correctly, but those nodes won't use ANE or GPU acceleration."
        )
        recs.append(
            "Run [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier) "
            "before export to fold constants and eliminate unnecessary ops: "
            "`pip install onnxsim && onnxsim model.onnx model_sim.onnx`."
        )
        for i, rec in enumerate(recs, 1):
            a(f"{i}. {rec}")
        a("")

    # ── Footer ─────────────────────────────────────────────────────────────────
    a("---")
    a("")
    a(f"*Generated by `check_coreml_ops.py` | ORT {ort_ver} | {result['timestamp']}*")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  📄  Markdown report saved → {out_path}\n")


def _op_fix_hint(op: str, fmt: str) -> str:
    """Full-length fix hint for MD report."""
    hints = {
        "BatchNormalization": (
            "Fuse into Conv via `torch.nn.utils.fuse_conv_bn_eval()` before export, "
            "or switch to `--format NeuralNetwork` where it is supported."
            if fmt == "MLProgram" else ""
        ),
        "Resize":     "Use `mode='nearest'` or `'linear'` in `F.interpolate`. Cubic is unsupported.",
        "Pad":        "Use `F.pad(..., mode='constant')` only. `reflect`/`edge` are unsupported.",
        "LSTM":       "Replace bidirectional LSTM with two forward LSTMs and concatenate outputs.",
        "GRU":        "Replace bidirectional GRU with two forward GRUs and concatenate outputs.",
        "GridSample": "Not supported in MLProgram. Decompose into affine + `Resize` if possible." if fmt == "MLProgram" else "",
        "NonMaxSuppression": "Run NMS post-processing in Python / Swift after CoreML inference.",
        "RoiAlign":   "Run ROI pooling on CPU EP as a post-processing step.",
        "ScatterND":  "Replace with `Gather` + arithmetic if the scatter pattern is simple.",
        "Einsum":     ("Replace with equivalent `MatMul` + `Transpose` chains."
                       if fmt == "NeuralNetwork" else ""),
        "Range":      ("Materialise as a constant initializer before export."
                       if fmt == "NeuralNetwork" else ""),
    }
    return hints.get(op, "No specific guidance — check the CoreML EP docs for alternatives.")


def _op_fix_hint_short(op: str, fmt: str) -> str:
    """One-liner hint for the quick-summary table."""
    full = _op_fix_hint(op, fmt)
    if not full or full.startswith("No specific"):
        return "See CoreML EP docs"
    # Truncate at first sentence
    short = full.split(".")[0] + "."
    return short if len(short) < 80 else short[:77] + "…"


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check ONNX model CoreML EP compatibility (Ubuntu-safe).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the .onnx model file.",
    )
    parser.add_argument(
        "--format", default="MLProgram",
        choices=["NeuralNetwork", "MLProgram"],
        help=(
            "CoreML model format to target.\n"
            "  NeuralNetwork — Core ML 3+  (iOS 13+ / macOS 10.15+)\n"
            "  MLProgram     — Core ML 5+  (iOS 15+ / macOS 12+)  [default]"
        ),
    )
    parser.add_argument(
        "--out", default=None,
        help="Path for the Markdown report (default: <model_dir>/<model_stem>_coreml_<format>_report.md).",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[error] Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Default output path encodes the format so two runs never overwrite each other
    if args.out:
        out_path = args.out
    else:
        stem     = Path(args.model).stem
        fmt_tag  = args.format.lower()         # "neuralnetwork" or "mlprogram"
        out_path = str(
            Path(args.model).with_name(f"{stem}_coreml_{fmt_tag}_report.md")
        )

    print(f"\n  Analysing model : {args.model}")
    print(f"  Target format   : {args.format}")
    print(f"  Report output   : {out_path}\n")

    result = analyse(args.model, args.format)
    print_terminal_report(result, out_path)
    write_markdown_report(result, out_path)


if __name__ == "__main__":
    main()