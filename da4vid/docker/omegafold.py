import concurrent.futures
import math
import os
import sys
from typing import List

from docker import DockerClient

from da4vid.docker.base import BaseContainer


class OmegaFoldContainer(BaseContainer):
  MODELS_FOLDER = '/root/.cache/omegafold_ckpt'
  INPUT_DIR = '/Omegafold/run/inputs'
  OUTPUT_DIR = '/Omegafold/run/outputs'

  def __init__(self, model_dir, input_dir, output_dir, model_weights: str = "2",
               num_recycles: int = 5, device: str = 'cpu', max_parallel: int = 1,
               image: str = 'da4vid/omegafold:latest'):
    super().__init__(
      image=image,
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
    self.max_parallel = max_parallel

  def run(self, client: DockerClient = None):
    container = super()._create_container(client)
    chunks = self.__get_fasta_chunks()
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
      for i, chunk in enumerate(chunks):  # Cycling on all chunks
        device = f'cuda:{i % 2}'  # TODO: get devices automatically
        executor.submit(self.__run_on_fasta_list, fasta_base_names=chunk, container=container, device=device)
    # After everything is finished, change permissions on generated files
    super()._execute_command(container,
                             f'/usr/bin/chmod 0777 --recursive {self.OUTPUT_DIR}',
                             file=sys.stdout)
    super()._stop_container(container)
    return True

  def __run_on_fasta_list(self, fasta_base_names: List[str], container, device: str):
    print(f'Running {fasta_base_names} on device {device}')
    for fasta_basename in fasta_base_names:
      command = self.__create_command(fasta_basename, device)
      super()._execute_command(container, command, file=sys.stdout)

  def __create_command(self, fasta_basename, device):
    fasta_no_ext = '.'.join(fasta_basename.split('.')[:-1])
    return (f'python3 main.py '
            f'--model {self.model_weights} '
            f'--device {device} '
            f'--num_cycle {self.num_recycles} '
            f'{OmegaFoldContainer.INPUT_DIR}/{fasta_basename} '
            f'{OmegaFoldContainer.OUTPUT_DIR}/{fasta_no_ext}')

  def __get_fasta_chunks(self) -> List[List[str]]:
      files = [f for f in os.listdir(self.input_dir) if f.endswith('.fa')]
      n = len(files) // self.max_parallel
      rem = len(files) % self.max_parallel
      ff = []
      for i in range(self.max_parallel):
        if i < rem:
          ff.append(files[i:i + n + 1])
        else:
          ff.append(files[i:i + n])
      return ff
