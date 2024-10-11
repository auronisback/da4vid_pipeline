import os

from docker import DockerClient

from da4vid.docker.base import BaseContainer


class OmegaFoldContainer(BaseContainer):
  MODELS_FOLDER = '/root/.cache/omegafold_ckpt'
  INPUT_DIR = '/Omegafold/run/inputs'
  OUTPUT_DIR = '/Omegafold/run/outputs'

  def __init__(self, model_dir, input_dir, output_dir, model_weights: str = "2",
               num_recycles: int = 5, device: str = 'cpu'):
    super().__init__(
      image='ameg/omegafold:latest',
      entrypoint='/bin/bash',
      with_gpus=True,
      volumes={
        model_dir: OmegaFoldContainer.MODELS_FOLDER,
        input_dir: OmegaFoldContainer.INPUT_DIR,
        output_dir: OmegaFoldContainer.OUTPUT_DIR
      },
      detach=True
    )
    self.model_dir = model_dir
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.num_recycles = num_recycles
    self.model_weights = model_weights
    self.device = device

  def run(self, client: DockerClient = None):
    for f in os.listdir(self.input_dir):  # Cycling all fasta files
      if f.endswith('.fa'):
        basename = '.'.join(os.path.basename(f).split('.')[:-1])
        self.commands.append((f'python3 /OmegaFold/main.py '
                              f'--model {self.model_weights} '
                              f'--device {self.device} '
                              f'--num_cycle {self.num_recycles} '
                              f'{OmegaFoldContainer.INPUT_DIR}/{f} '
                              f'{OmegaFoldContainer.OUTPUT_DIR}/{basename}'))
    super()._run_container(client)
