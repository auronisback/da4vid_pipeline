import concurrent.futures
import os
import threading
from typing import List

import docker

from da4vid.docker.base import BaseContainer
from da4vid.gpus.cuda import CudaDeviceManager


class OmegaFoldContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/omegafold:latest'

  MODELS_FOLDER = '/root/.cache/omegafold_ckpt'
  INPUT_DIR = '/Omegafold/run/inputs'
  OUTPUT_DIR = '/Omegafold/run/outputs'

  def __init__(self, model_dir, input_dir, output_dir, client: docker.DockerClient, gpu_manager: CudaDeviceManager,
               model_weights: str = "2", num_recycles: int = 5, max_parallel: int = 1,
               image: str = DEFAULT_IMAGE):
    super().__init__(
      image=image,
      entrypoint='/bin/bash',
      volumes={
        model_dir: OmegaFoldContainer.MODELS_FOLDER,
        input_dir: OmegaFoldContainer.INPUT_DIR,
        output_dir: OmegaFoldContainer.OUTPUT_DIR
      },
      client=client,
      gpu_manager=gpu_manager
    )
    self.model_dir = model_dir
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.num_recycles = num_recycles
    self.model_weights = model_weights
    self.max_parallel = max_parallel

  def run(self):
    chunks = self.__get_fasta_chunks()
    res = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
      futures = [executor.submit(self.__create_and_run_container, fasta_basenames=chunk) for chunk in chunks]
      for future in concurrent.futures.as_completed(futures):
        res &= future.result()
        if not res:
          break
    return res

  def __create_and_run_container(self, fasta_basenames) -> bool:
    container, device = super()._create_container()
    print(f'[{threading.current_thread().name}] Running {fasta_basenames} on device {device.name}')
    # Run commands in the container
    res = self.__run_on_fasta_list(fasta_basenames, container)
    # After everything is finished, change permissions on generated files
    res &= super()._execute_command(container,
                                    f'/usr/bin/chmod 0777 --recursive {self.OUTPUT_DIR}')
    super()._stop_container(container)
    return res

  def __run_on_fasta_list(self, fasta_basenames: List[str], container) -> bool:
    res = True
    for fasta_basename in fasta_basenames:
      command = self.__create_command(fasta_basename)
      res &= super()._execute_command(container, command)
      if not res:
        break
    return res

  def __create_command(self, fasta_basename):
    fasta_no_ext = '.'.join(fasta_basename.split('.')[:-1])
    return (f'python3 main.py '
            f'--model {self.model_weights} '
            #f'--device {device} '
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
