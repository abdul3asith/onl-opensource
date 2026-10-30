import torch
from policy import act


def make_step_fn(model, variant, device):
    if variant == "baseline":
        return lambda image, state: act(model, image, state)

    if variant == "fp16":

        def act_fn(image, state):
            with torch.autocast(device_type=device, dtype=torch.float16):
                return act(model, image, state)

        return act_fn

    if variant == "bf16":

        def act_fn(image, state):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                return act(model, image, state)

        return act_fn

    if variant == "compile":
        compiled = torch.compile(model)
        return lambda image, state: act(compiled, image, state)

    raise ValueError(f"unknown variant: {variant}")
