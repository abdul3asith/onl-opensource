from policy import make_observation


def drift(baseline_fn, variant_fn, device, samples=50):
    """Average difference in actions between baseline and variant, same inputs."""
    total = 0.0
    for _ in range(samples):
        image, state = make_observation(device)
        a = baseline_fn(image, state).float()
        b = variant_fn(image, state).float()
        total += (a - b).abs().mean().item()
    return total / samples


def success_rate(variant_fn):
    # TODO: run real episodes in LIBERO and count successes
    return None
