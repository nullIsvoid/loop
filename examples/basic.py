import torch

from latent_loop import RingLatentProcessor, RingMixConfig

# Typical video latent layout: [batch, channels, time, height, width]
latents = torch.randn(1, 16, 81, 60, 104, device="cuda", dtype=torch.float16)

ring = RingLatentProcessor(
    RingMixConfig(
        radius=1,
        maximum_strength=0.12,
        active_fraction=0.70,
        temporal_dim=2,
    )
)

# Integration point: after a sampler updates x_t, before the next denoise step.
for step in range(30):
    # latents = scheduler_step(model(...), latents, ...)
    latents = ring(latents, step=step, total_steps=30)
