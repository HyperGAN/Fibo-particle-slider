"""Run with python -m fibo_particle_slider.example OUTPUT; CPU mock only."""
import sys
from pathlib import Path
import torch
from particle_sliders.lifecycle import open_session
from particle_sliders.inference import open_inference
from PIL import Image
from .lifecycle import synthetic_config


def main():
    torch.set_num_threads(1)
    output = Path(sys.argv[1])
    config = synthetic_config(output / 'cache')
    with open_session(config) as session:
        session.train(stop_after=2)
        session.checkpoint(output / 'training.pt')
    with open_session(config, resume=output / 'training.pt') as session:
        session.train()
        session.export(output / 'adapter.pt', selection='ema')
        session.export_inference(output / 'native.pt', selection='ema')
    with open_session(config, artifact=output / 'adapter.pt', operation='evaluate') as session:
        assert session.evaluate().shape == (1, 4, 2, 2)
    native = dict(prompt='keep jacket', images=[Image.new('RGB', (8, 8), (64, 20, 30))])
    with open_inference(output / 'native.pt', references=True, output_type='latent', native_request=native) as session:
        assert session.render(**native, strength=.5).images[0].shape == (1, 4, 2, 2)
    print('CPU mock: trained, resumed, reloaded response and rendered native-only EMA export')


if __name__ == '__main__':
    main()
