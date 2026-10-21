"""Bounded CPU mock of the EDIT protocol; no native/pretrained qualification."""
from types import SimpleNamespace
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q, self.to_k, self.to_v = [nn.Linear(4, 4, bias=False) for _ in range(3)]
        self.to_out = nn.ModuleList([nn.Linear(4, 4, bias=False)])

    def forward(self, value):
        return self.to_out[0](torch.sin(self.to_q(value) + self.to_k(value) + self.to_v(value)))


class Block(nn.Module):
    def __init__(self, cross=False):
        super().__init__()
        self.attention = Attention()
        self.cross_attention = Attention() if cross else None

    def forward(self, value):
        value = value + self.attention(value)
        return value if self.cross_attention is None else value + self.cross_attention(value)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_layers=1, n_refiner_layers=1, injection_layer_ids=[0],
                                      cap_feat_dim=4, in_channels=4, patch_size=2)
        self.noise_refiner = nn.ModuleList([Block()])
        self.layers = nn.ModuleList([Block(True)])
        self.seen = []

    def forward(self, *, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask, context_latents, return_dict):
        self.seen.append(dict(time=timestep.detach().clone(), references=[r.detach().clone() for r in context_latents],
                              mask=None if encoder_attention_mask is None else encoder_attention_mask.clone(),
                              shape=hidden_states.shape))
        assert return_dict is False and context_latents
        mask = (torch.ones(1, encoder_hidden_states.shape[2], dtype=torch.bool)
                if encoder_attention_mask is None else encoder_attention_mask)
        embedding = (encoder_hidden_states * mask[:, None, :, None]).sum((1, 2)) / mask.sum()
        reference = sum((i + 1) * r.mean((2, 3)) for i, r in enumerate(context_latents))
        value = hidden_states.permute(0, 2, 3, 1) + embedding[:, None, None] + reference[:, None, None]
        for module in [*self.noise_refiner, *self.layers]:
            value = module(value)
        return (value.permute(0, 3, 1, 2) * .2 + timestep[:, None, None, None] * .1,)


class Pipeline:
    """Mock orchestration, deliberately not a pretrained/native qualification."""
    def __init__(self):
        with torch.random.fork_rng():
            torch.manual_seed(123)
            self.transformer = Transformer()
        self.scheduler = SimpleNamespace(calls=0)
        self.text_encoder = nn.Linear(1, 1)
        self.vae = nn.Linear(1, 1)
        self.calls = []

    def __call__(self, *, prompt, image, mask, output_type, num_inference_steps, guidance_scale, generator,
                 height=None, width=None):
        assert num_inference_steps == 4 and guidance_scale == 1
        self.calls.append(dict(prompt=prompt, image=image, mask=mask, output_type=output_type))
        embeds = torch.full((1, 2, 3, 4), (sum(prompt.encode()) % 101) / 100.)
        # The last embedding is padded and deliberately huge.
        embeds[:, :, -1] = 10000
        contexts = [torch.full((1, 4, 2 + 2 * i, 2), float(img.getpixel((0, 0))[0]) / 255)
                    for i, img in enumerate(image)]
        if mask is not None:
            contexts[0] = contexts[0] + .05
        latent = torch.randn((1, 4, 2, 2), generator=generator)
        for time, delta in [(1000., -.18), (820., -.32), (500., -.35), (150., -.15)]:
            prediction = self.transformer(hidden_states=latent, timestep=torch.tensor([time / 1000]),
                encoder_hidden_states=embeds, encoder_attention_mask=torch.tensor([[True, True, False]]),
                context_latents=contexts, return_dict=False)[0].float()
            latent = latent.float() + delta * prediction
            self.scheduler.calls += 1
        return SimpleNamespace(images=[latent if output_type == 'latent' else 'native decoder output'])

