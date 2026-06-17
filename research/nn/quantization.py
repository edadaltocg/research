import torch


def post_training_static_quantization(model, backend="qnnpack"):
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    prepare_fn = torch.quantization.prepare
    convert_fn = torch.quantization.convert
    model_static_quantized = prepare_fn(model, inplace=False)
    model_static_quantized = convert_fn(model_static_quantized, inplace=False)
    return model_static_quantized
