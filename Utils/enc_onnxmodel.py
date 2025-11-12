import onnx
import os
import zlib
from onnxruntime.quantization import quantize_dynamic, QuantType
from NetModule import NetModule
import torch

# a simple encryption method for ONNX model


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

    # Load the exported model proto from the saved ONNX file
    model_proto = onnx.load(model_file)
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
