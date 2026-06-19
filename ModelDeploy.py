"""
Export a trained checkpoint to ONNX and verify it.

    python ModelDeploy.py

Reads the ``[Deployment]`` config section, exports the (best) checkpoint to
ONNX, runs the ONNX model with onnxruntime, compares its output against the
PyTorch model, and writes a CoreML-compatibility report.
"""

import numpy as np
import onnxruntime as ort
import torch

from NetModule import NetModule
from Utils.check_CoreML_ops import analyse, print_terminal_report, write_markdown_report
from Utils.deploy_onnxmodel import load_onnxmodel, save_onnxmodel
from Utils.training import find_best_checkpoint, load_config


def main(config_path: str = "config.toml"):
    config = load_config(config_path)
    deploy = config["Deployment"]

    checkpoint = deploy.get("checkpoint") or find_best_checkpoint(config)
    if not checkpoint:
        raise FileNotFoundError("No checkpoint found. Train a model first or set [Deployment].checkpoint.")
    print(f"Exporting checkpoint: {checkpoint}")

    c, h, w = config["DataModule"]["image_shape"]
    n_class = config["DataModule"]["n_class"]
    input_shape = (1, c, h, w)
    out_dir = deploy["output_dir"]
    onnx_name = deploy.get("onnx_filename", "model.onnx")
    onnx_path = f"{out_dir}/{onnx_name}"

    save_onnxmodel(
        checkpoint,
        out_path=out_dir,
        opset=deploy.get("opset", 18),
        input_shape=input_shape,
        enable_quantize=deploy.get("quantize", False),
        saved_model_filename=onnx_name,
    )

    # Sanity check the exported graph with onnxruntime.
    ort_sess = ort.InferenceSession(load_onnxmodel(onnx_path))
    dummy = np.random.rand(*input_shape).astype(np.float32)
    y = ort_sess.run(None, {"input": dummy})
    print("ONNX output shape:", y[0].shape, "(expected", (1, n_class, h, w), ")")

    # Compare PyTorch vs ONNX outputs on the same input.
    print("Checking inference correctness...")
    model = NetModule.load_from_checkpoint(checkpoint).eval().to("cpu")
    inp = torch.randn(input_shape, dtype=torch.float32)
    with torch.no_grad():
        pt_out = model(inp).cpu().numpy().astype(np.float32)
    onnx_out = ort_sess.run(None, {"input": inp.numpy()})[0]
    print("Max abs difference (PyTorch vs ONNX):", np.max(np.abs(pt_out - onnx_out)))
    print("Outputs match:", np.allclose(pt_out, onnx_out, atol=1e-4))

    # CoreML compatibility report.
    report_path = f"{out_dir}/compatibility_report.md"
    result = analyse(onnx_path, "MLProgram")
    print_terminal_report(result, report_path)
    write_markdown_report(result, report_path)


if __name__ == "__main__":
    main()
