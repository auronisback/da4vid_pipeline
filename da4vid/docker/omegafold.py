import os

from docker import DockerClient

from da4vid.docker.base import BaseContainer


class OmegaFoldContainer(BaseContainer):
  MODELS_FOLDER = '/root/.cache/omegafold_ckpt'
  INPUT_DIR = '/Omegafold/run/inputs'
  OUTPUT_DIR = '/Omegafold/run/outputs'

  def __init__(self, model_dir, input_dir, output_dir,
               running_model: str = "2"):
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
    self.model = running_model

  def run(self, num_cycle: int = 5, device: str = 'cpu',
          client: DockerClient = None):
    for f in os.listdir(self.input_dir):  # Cycling all fasta files
      if f.endswith('.fa'):
        basename = '.'.join(os.path.basename(f).split('.')[:-1])
        self.commands.append((f'python3 /OmegaFold/main.py '
                              f'--model {self.model} '
                              f'--device {device} '
                              f'--num_cycle {num_cycle} '
                              f'{OmegaFoldContainer.INPUT_DIR}/{f} '
                              f'{OmegaFoldContainer.OUTPUT_DIR}/{basename}'))
    super()._run_container(client)
