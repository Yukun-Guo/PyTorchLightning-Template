import os
import zlib
from onnxruntime.quantization import quantize_dynamic, QuantType
from NetModule import NetModule
import torch

"""
fuse_conv_bn.py
---------------
Automatically fuse all Conv→BN (and Conv→BN→ReLU) patterns in a PyTorch
model before ONNX export — without manually naming every layer pair.

Three strategies, in order of preference:
  1. torch.fx  automatic graph-level fusion          ← best for any model
  2. torch.ao.quantization.fuse_modules_qat          ← good for named-module models
  3. ONNX-level fusion via onnx-simplifier            ← last resort, no PyTorch needed

Usage:
    from fuse_conv_bn import fuse_conv_bn_auto
    model = MyModel()
    model.load_state_dict(torch.load("weights.pth"))
    fused = fuse_conv_bn_auto(model)
    # then export fused to ONNX as normal
"""

import copy
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 1: torch.fx  — automatic, graph-level, works on any nn.Module
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_conv_bn_fx(model: nn.Module) -> nn.Module:
    """
    Uses torch.fx to trace the full computation graph, automatically finds
    every Conv→BN and Conv→BN→ReLU sequence regardless of how deep or how
    they are named, and folds the BN parameters into the Conv weights.

    Requirements:
        - Model must be traceable by torch.fx (no Python control flow on
          tensor values, no dynamic shapes). If tracing fails, fall back to
          fuse_conv_bn_named() or fuse_conv_bn_onnx().
        - Model must be in eval() mode.

    Returns a new model with BN layers eliminated.
    """
    model = copy.deepcopy(model).eval()

    try:
        # torch.ao.quantization.fuse_fx does exactly this fusion cleanly
        from torch.ao.quantization.quantize_fx import fuse_fx
        fused = fuse_fx(model)
        print("[fx] fuse_fx() succeeded.")
        return fused
    except Exception as e:
        print(f"[fx] fuse_fx() failed: {e}")
        print("[fx] Falling back to manual fx pattern matching...")

    # Manual fallback: trace + pattern-match Conv→BN pairs in the fx graph
    try:
        traced = symbolic_trace(model)
        _fuse_conv_bn_in_graph(traced)
        traced.recompile()
        print("[fx] Manual graph fusion succeeded.")
        return traced
    except Exception as e:
        print(f"[fx] Manual graph fusion failed: {e}")
        raise


def _fuse_conv_bn_in_graph(traced: torch.fx.GraphModule) -> None:
    """
    Walk the fx graph and fold BN into the preceding Conv in-place.
    Handles: Conv→BN  and  Conv→BN→ReLU
    """
    graph   = traced.graph
    modules = dict(traced.named_modules())

    for bn_node in list(graph.nodes):
        if bn_node.op != "call_module":
            continue
        bn_mod = modules.get(bn_node.target)
        if not isinstance(bn_mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            continue

        # The single input to BN must come from a Conv
        if len(bn_node.args) != 1:
            continue
        conv_node = bn_node.args[0]
        if conv_node.op != "call_module":
            continue
        conv_mod = modules.get(conv_node.target)
        if not isinstance(conv_mod, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d)):
            continue

        # Fold BN into Conv weights
        _absorb_bn_into_conv(conv_mod, bn_mod)

        # Replace all uses of BN output with Conv output
        bn_node.replace_all_uses_with(conv_node)
        graph.erase_node(bn_node)
        print(f"  [fx] Fused  {conv_node.target}  ←  {bn_node.target}")

    traced.delete_all_unused_submodules()
    graph.lint()


def _absorb_bn_into_conv(conv: nn.Module, bn: nn.Module) -> None:
    """
    Mathematically fold BatchNorm parameters into Conv weight and bias.

    BN formula:  y = (x - mean) / sqrt(var + eps) * gamma + beta
    Fused Conv:  W' = W * (gamma / std),   b' = (b - mean) * (gamma / std) + beta
    """
    bn.eval()
    conv.eval()

    with torch.no_grad():
        # BN parameters
        gamma  = bn.weight.data                                    # [C_out]
        beta   = bn.bias.data                                      # [C_out]
        mean   = bn.running_mean.data                              # [C_out]
        var    = bn.running_var.data                               # [C_out]
        eps    = bn.eps
        std    = torch.sqrt(var + eps)                             # [C_out]
        scale  = gamma / std                                       # [C_out]

        # Reshape scale to broadcast over Conv weight dimensions
        # Conv weight shape: [C_out, C_in/groups, kH, kW]
        while scale.dim() < conv.weight.dim():
            scale = scale.unsqueeze(-1)

        # Fuse weight
        conv.weight.data = conv.weight.data * scale.squeeze(-1).squeeze(-1).squeeze(-1).reshape(
            -1, *([1] * (conv.weight.dim() - 1))
        )

        # Fuse bias (create if Conv has none)
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.weight.shape[0]))
        conv.bias.data = (conv.bias.data - mean) * (gamma / std) + beta


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 2: fuse_modules — works on Sequential / named-submodule patterns
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_conv_bn_named(model: nn.Module) -> nn.Module:
    """
    Uses torch.ao.quantization.fuse_modules to fuse all Conv→BN and
    Conv→BN→ReLU triples that appear as *consecutive siblings* inside
    any nn.Sequential or named module container.

    Automatically discovers all fusable sequences — no manual naming needed.
    This works well for ResNets, UNets, VGGs, and other models built with
    nn.Sequential blocks.
    """
    model = copy.deepcopy(model).eval()
    from torch.ao.quantization import fuse_modules

    def _find_fusable_sequences(parent: nn.Module, prefix: str = "") -> list[list[str]]:
        """
        Recursively scan all named children.
        Returns lists of [conv_name, bn_name] or [conv_name, bn_name, relu_name]
        relative to `parent`.
        """
        sequences = []
        children  = list(parent.named_children())

        i = 0
        while i < len(children):
            name, mod = children[i]
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this child starts a Conv→BN (→ReLU) sequence
            if isinstance(mod, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d)):
                seq = [full_name]
                if i + 1 < len(children):
                    next_name, next_mod = children[i + 1]
                    next_full = f"{prefix}.{next_name}" if prefix else next_name
                    if isinstance(next_mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        seq.append(next_full)
                        # Check for trailing ReLU
                        if i + 2 < len(children):
                            rn, rm = children[i + 2]
                            rf = f"{prefix}.{rn}" if prefix else rn
                            if isinstance(rm, (nn.ReLU, nn.ReLU6)):
                                seq.append(rf)
                        sequences.append(seq)
                        i += len(seq)
                        continue

            # Recurse into submodules
            sequences.extend(_find_fusable_sequences(mod, full_name))
            i += 1

        return sequences

    seqs = _find_fusable_sequences(model)
    if not seqs:
        print("[named] No fusable Conv→BN sequences found in named children.")
        return model

    for seq in seqs:
        try:
            fuse_modules(model, seq, inplace=True)
            print(f"  [named] Fused: {' → '.join(seq)}")
        except Exception as e:
            print(f"  [named] Skipped {seq}: {e}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 3: ONNX-level fusion — no PyTorch model needed
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_conv_bn_onnx(onnx_path: str, output_path: str | None = None) -> str:
    """
    Fuses Conv→BN pairs directly in the ONNX graph using onnxoptimizer.
    Use this when you only have the .onnx file and no PyTorch source.

    Requires:  pip install onnxoptimizer
    Returns:   path to the optimised model.
    """
    try:
        import onnxoptimizer
    except ImportError:
        raise ImportError("pip install onnxoptimizer")

    import onnx

    if output_path is None:
        from pathlib import Path
        p = Path(onnx_path)
        output_path = str(p.with_name(p.stem + "_fused.onnx"))

    model = onnx.load(onnx_path)

    # These passes fold BN into Conv and eliminate dead nodes
    passes = [
        "fuse_bn_into_conv",           # core Conv+BN folding
        "eliminate_deadend",           # remove orphaned BN nodes
        "eliminate_identity",          # clean up Identity nodes
        "eliminate_nop_transpose",     # optional cleanup
        "fuse_consecutive_squeezes",   # optional cleanup
    ]
    optimised = onnxoptimizer.optimize(model, passes)
    onnx.save(optimised, output_path)

    # Report what changed
    orig_bn = sum(1 for n in model.graph.node if n.op_type == "BatchNormalization")
    new_bn  = sum(1 for n in optimised.graph.node if n.op_type == "BatchNormalization")
    orig_c  = sum(1 for n in model.graph.node if n.op_type == "Conv")
    new_c   = sum(1 for n in optimised.graph.node if n.op_type == "Conv")
    print(f"[onnx] Conv: {orig_c} → {new_c}   BatchNorm: {orig_bn} → {new_bn}")
    print(f"[onnx] Saved: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — auto-selects the best strategy
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_conv_bn_auto(
    model: nn.Module,
    prefer: str = "fx",          # "fx" | "named"
    verify: bool = True,
    example_input: torch.Tensor | None = None,
) -> nn.Module:
    """
    Automatically fuse all Conv→BN patterns in a PyTorch model.

    Args:
        model:         nn.Module in eval() mode.
        prefer:        "fx"    — use torch.fx (best coverage, any architecture)
                       "named" — use fuse_modules (Sequential/named-child models)
        verify:        If True and example_input is given, checks that the fused
                       model's output matches the original to within 1e-4.
        example_input: Optional tensor for verification (same shape as model input).

    Returns:
        Fused nn.Module with BN layers eliminated.

    Example:
        from fuse_conv_bn import fuse_conv_bn_auto

        model = MySegNet()
        model.load_state_dict(torch.load("model.pth"))
        model.eval()

        fused = fuse_conv_bn_auto(model, example_input=torch.zeros(1, 1, 480, 288))

        torch.onnx.export(
            fused,
            torch.zeros(1, 1, 480, 288),
            "model_fused.onnx",
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
        )
    """
    model = model.eval()

    if prefer == "fx":
        try:
            fused = fuse_conv_bn_fx(model)
        except Exception as e:
            print(f"[auto] fx strategy failed ({e}), trying named strategy...")
            fused = fuse_conv_bn_named(model)
    else:
        fused = fuse_conv_bn_named(model)

    # Count remaining BN layers
    remaining_bn = [
        name for name, mod in fused.named_modules()
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
    if remaining_bn:
        print(f"\n⚠️  {len(remaining_bn)} BN layer(s) could NOT be fused "
              f"(likely no preceding Conv in the graph):")
        for name in remaining_bn:
            print(f"    {name}")
    else:
        bn_count = sum(
            1 for _, m in model.named_modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        )
        print(f"\n✅  All {bn_count} BN layer(s) successfully fused into Conv.")

    # Numerical verification
    if verify and example_input is not None:
        _verify_fusion(model, fused, example_input)

    return fused


def _verify_fusion(
    original: nn.Module,
    fused: nn.Module,
    example_input: torch.Tensor,
    atol: float = 1e-4,
) -> None:
    original.eval()
    fused.eval()
    with torch.no_grad():
        out_orig  = original(example_input)
        out_fused = fused(example_input)
        # Handle tuple outputs (e.g. models that return (logits, aux))
        if isinstance(out_orig, (tuple, list)):
            out_orig  = out_orig[0]
            out_fused = out_fused[0]
        max_diff = (out_orig - out_fused).abs().max().item()
    if max_diff < atol:
        print(f"✅  Fusion verification passed  (max |Δ| = {max_diff:.2e}  < {atol})")
    else:
        print(f"⚠️  Fusion verification FAILED  (max |Δ| = {max_diff:.2e}  ≥ {atol})")
        print("    The fused model output differs. Check for in-place ops or untraceable layers.")


# # ═══════════════════════════════════════════════════════════════════════════════
# # CLI — fuse directly from checkpoint or ONNX
# # ═══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     import argparse, sys
#     from pathlib import Path

#     parser = argparse.ArgumentParser(
#         description="Auto-fuse Conv→BN layers before ONNX export.",
#         formatter_class=argparse.RawTextHelpFormatter,
#     )
#     sub = parser.add_subparsers(dest="cmd", required=True)

#     # ── onnx subcommand (no PyTorch model needed) ──────────────────────────────
#     p_onnx = sub.add_parser("onnx", help="Fuse Conv→BN in an existing .onnx file.")
#     p_onnx.add_argument("--input",  required=True, help="Input .onnx path.")
#     p_onnx.add_argument("--output", default=None,  help="Output .onnx path (default: <stem>_fused.onnx).")

#     # ── pytorch subcommand ─────────────────────────────────────────────────────
#     p_pt = sub.add_parser("pytorch",
#         help="Fuse Conv→BN in a PyTorch model and re-export to ONNX.\n"
#              "Requires a NetModule-compatible checkpoint.")
#     p_pt.add_argument("--checkpoint", required=True, help="Path to .ckpt or .pth checkpoint.")
#     p_pt.add_argument("--output",     required=True, help="Output .onnx path.")
#     p_pt.add_argument("--input-shape", default="1,1,480,288",
#                       help="Input shape as C,H,W or N,C,H,W (default: 1,1,480,288).")
#     p_pt.add_argument("--strategy",  default="fx", choices=["fx", "named"],
#                       help="Fusion strategy: fx (default) or named.")
#     p_pt.add_argument("--opset",     default=18, type=int, help="ONNX opset (default: 18).")

#     args = parser.parse_args()

#     if args.cmd == "onnx":
#         fuse_conv_bn_onnx(args.input, args.output)

#     elif args.cmd == "pytorch":
#         # ── Load checkpoint ───────────────────────────────────────────────────
#         ckpt = Path(args.checkpoint)
#         if not ckpt.exists():
#             print(f"[error] Checkpoint not found: {ckpt}", file=sys.stderr)
#             sys.exit(1)

#         shape = tuple(int(x) for x in args.input_shape.split(","))
#         if len(shape) == 3:
#             shape = (1,) + shape   # prepend batch dim

#         print(f"\nLoading checkpoint: {ckpt}")
#         try:
#             # Try PyTorch Lightning checkpoint first
#             from NetModule import NetModule
#             model = NetModule.load_from_checkpoint(str(ckpt))
#         except ImportError:
#             # Plain PyTorch .pth
#             model = torch.load(str(ckpt), map_location="cpu")
#         model.eval().cpu()

#         example = torch.zeros(*shape)

#         print(f"\nFusing Conv→BN (strategy={args.strategy})...")
#         fused = fuse_conv_bn_auto(
#             model,
#             prefer=args.strategy,
#             verify=True,
#             example_input=example,
#         )

#         print(f"\nExporting fused model to ONNX: {args.output}")
#         torch.onnx.export(
#             fused,
#             example,
#             args.output,
#             opset_version=args.opset,
#             input_names=["input"],
#             output_names=["output"],
#             dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
#         )
#         print("Done.")

# Toy encryption/decryption method for ONNX model
def encrypt_onnxmodel(ckpt_path, out_path: str='./model', opset: int = 18, input_shape: tuple = (1, 1, 480, 288), enable_quantize=False, saved_model_filename="enc_model.enc", encrypt_key: int = 1234):
    model = NetModule.load_from_checkpoint(ckpt_path)
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    input_sample = torch.randn(input_shape, device=device)

    input_names = ['input']
    output_names = ['output']
    # allow dynamic batch, height and width
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'},
    }
    # Ensure output directory exists (fixes FileNotFoundError when external data is written)
    os.makedirs(out_path, exist_ok=True)

    model_file = os.path.join(out_path, "model.onnx")

    torch.onnx.export(
        model,
        input_sample,
        model_file,
        export_params=True,
        opset_version=opset,
        external_data=False,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True,
    )

    # If quantization is enabled, quantize the model use static quantization
    if enable_quantize:
        # quantize_dynamic accepts a path for input; use the saved model file
        quantize_dynamic(model_file, os.path.join(out_path, "model_quantized.onnx"), weight_type=QuantType.QInt8)

    # Read the saved ONNX model into a buffer
    with open(model_file, "rb") as f:
        model_onnx = f.read()
        # Encrypt the model by reversing the middle third of the file
        model_onnx = (
            model_onnx[: int(len(model_onnx) / 3)]
            + model_onnx[int(len(model_onnx) / 3) : int(2 * len(model_onnx) / 3)][::-1]
            + model_onnx[int(2 * len(model_onnx) / 3) :]
        )
        # Compress the encrypted model using zlib
        zlib_model = zlib.compress(model_onnx)
        # Append random bytes to the end of the compressed model
        zlib_model += os.urandom(encrypt_key)
    # Write the encrypted and compressed model to a new file
    with open(os.path.join(out_path, saved_model_filename), "wb") as f:
        f.write(zlib_model)

def decrypt_onnxmodel(model_path, encrypt_key: int = 1234):
    """
    Example decrypts an encrypted ONNX model file.

    Args:
        model_path (str): The path to the encrypted model file.
        encrypt_key (int, optional): The number of random bytes appended to the encrypted file. Default is 2836.

    Returns:
        bytes: The decrypted ONNX model file content.
    """
    # Read the encrypted model file into a buffer
    with open(model_path, "rb") as f:
        model_buffer = f.read()

        # Remove the appended random bytes
        model_buffer = model_buffer[:-encrypt_key]

        # Decompress the model using zlib
        model_buffer = zlib.decompress(model_buffer)

        # Decrypt the model by reversing the middle third of the file back to its original state
        model_buffer = (
            model_buffer[: int(len(model_buffer) / 3)]
            + model_buffer[int(len(model_buffer) / 3) : int(2 * len(model_buffer) / 3)][::-1]
            + model_buffer[int(2 * len(model_buffer) / 3) :]
        )

    # Return the decrypted model buffer
    return model_buffer

# Save ONNX model
def save_onnxmodel(ckpt_path, out_path: str='./model', opset: int = 18, input_shape: tuple = (1, 1, 480, 288), enable_quantize=False, saved_model_filename="model.onnx"):
    model = NetModule.load_from_checkpoint(ckpt_path)
    model.eval()
    
    device = torch.device('cpu')
    model.to(device)
    model = fuse_conv_bn_auto(model)
    
    input_sample = torch.randn(input_shape, device=device)

    input_names = ['input']
    output_names = ['output']
    # allow dynamic batch, height and width
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'},
    }
    # Ensure output directory exists (fixes FileNotFoundError when external data is written)
    os.makedirs(out_path, exist_ok=True)

    model_file = os.path.join(out_path, saved_model_filename)

    torch.onnx.export(
        model,
        input_sample,
        model_file,
        export_params=True,
        opset_version=opset,
        external_data=False,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True,
    )

def load_onnxmodel(model_path):
    """
    Example loads an ONNX model file.

    Args:
        model_path (str): The path to the ONNX model file.

    Returns:
        model buffer
    """
    # Read the ONNX model file into a buffer
    with open(model_path, "rb") as f:
        model_buffer = f.read()
    # Return the model buffer
    return model_buffer