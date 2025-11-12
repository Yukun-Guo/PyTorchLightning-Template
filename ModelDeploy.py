from NetModule import NetModule
import torch
import onnxruntime as ort
import numpy as np
import os
import glob
from Utils.deploy_onnxmodel import save_onnxmodel, load_onnxmodel

# load model from checkpoint
# get the latest checkpoint in the logs folder
logs_folder = "./logs/layer_segmentation"
checkpoint_files = glob.glob(os.path.join(logs_folder, "*.ckpt"))
checkpoint_files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time
checkpoint_path = checkpoint_files[0] if checkpoint_files else None

save_onnxmodel(checkpoint_path, out_path="./deployed_model",opset=18, input_shape=(1,1,480,288), enable_quantize=False, saved_model_filename="model.onnx")

# decrypt model and test
model_buffer = load_onnxmodel("./deployed_model/model.onnx")
ort_sess = ort.InferenceSession(model_buffer)
y = ort_sess.run(None, {"input": np.random.rand(1, 1, 480, 288).astype(np.float32)})
print("Test: onnx: ", np.equal(y[0].shape, np.array([1,11, 480, 288])))

# check inference correctness, compare with pytorch model
print("Checking inference correctness...")
model = NetModule.load_from_checkpoint(checkpoint_path)
model.eval()
model.to('cpu')
inp = torch.randn((1, 1, 480, 288), dtype=torch.float32)
with torch.no_grad():
    pt_out = model(inp)
pt_out_np = pt_out.cpu().numpy().astype(np.float32)
onnx_outs = ort_sess.run(None, {"input": inp.numpy()})
onnx_out = onnx_outs[0]
print("Max absolute difference between PyTorch and ONNX Runtime outputs: ", np.max(np.abs(pt_out_np - onnx_out)))
print("Inference correctness check passed:", np.allclose(pt_out_np, onnx_out, atol=1e-5))