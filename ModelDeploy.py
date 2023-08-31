from NetModule import NetModule
# load model
model = NetModule.load_from_checkpoint(
    'logs/model-epoch=028-val_loss=0.71930.ckpt')
model.to_onnx('RetinalSegModel_v2.onnx')
