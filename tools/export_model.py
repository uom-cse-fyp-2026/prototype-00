"""Export an in-memory PyTorch Sequential model to the NeuroMinds text format."""

from pathlib import Path


def export_sequential(model, output):
    """Export exact Linear layers, each optionally followed by an exact ReLU."""
    import torch
    from torch import nn

    # Exact types keep custom forward methods from being silently approximated.
    if type(model) is not nn.Sequential:
        raise ValueError("expected torch.nn.Sequential")
    layers = []
    parameters = 0
    for module in model:
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("modules with forward hooks are unsupported")
        if type(module) is nn.ReLU:
            if not layers or layers[-1][1]:
                raise ValueError("ReLU must immediately follow a Linear")
            layers[-1] = (layers[-1][0], True)
            continue
        if type(module) is not nn.Linear:
            raise ValueError(f"unsupported module type: {type(module).__name__}")
        if layers and layers[-1][0].out_features != module.in_features:
            raise ValueError("adjacent layer dimensions do not match")
        if not 1 <= module.in_features <= 16384 or not 1 <= module.out_features <= 16384:
            raise ValueError("linear dimensions must be in [1, 16384]")
        parameters += module.weight.numel()
        if module.bias is not None:
            parameters += module.bias.numel()
        if len(layers) == 16 or parameters > 1_000_000:
            raise ValueError("model exceeds 16 layers or 1,000,000 parameters")
        layers.append((module, False))
    if not layers:
        raise ValueError("expected at least one Linear layer")
    if model._forward_hooks or model._forward_pre_hooks:
        raise ValueError("models with forward hooks are unsupported")

    def values(tensor, shape):
        if tuple(tensor.shape) != shape or not tensor.is_floating_point():
            raise ValueError("expected a floating-point parameter with the declared Linear shape")
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(tensor).all().item():
            raise ValueError("FP32 model parameters must be finite")
        return " ".join(format(value, ".9g") for value in tensor.reshape(-1).tolist())

    lines = [f"NM_MODEL_V1 {len(layers)}"]
    for layer, relu in layers:
        activation = "relu" if relu else "identity"
        bias = "bias" if layer.bias is not None else "no_bias"
        lines.append(f"linear {layer.in_features} {layer.out_features} {activation} {bias}")
        lines.append(values(layer.weight, (layer.out_features, layer.in_features)))
        if layer.bias is not None:
            lines.append(values(layer.bias, (layer.out_features,)))
    Path(output).write_text("\n".join(lines) + "\n", encoding="ascii")
