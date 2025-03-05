import concurrent.futures
import math
import os.path
import threading
from typing import List

import docker

from da4vid import io
from da4vid.containers.base import BaseContainer
from da4vid.containers.executor import ContainerExecutorBuilder, ContainerExecutor
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.model.proteins import Protein
from tmp.singularity.singularity import CONTAINER_OUTPUT_FOLDER


class ChaiContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/chai-lab:latest'

  CONTAINER_INPUT_DIR = '/chai-lab/inputs'
  CONTAINER_OUTPUT_DIR = '/chai-lab/outputs'

  def __init__(self, builder: ContainerExecutorBuilder, gpu_manager: CudaDeviceManager, input_dir: str, output_dir: str,
               trunk_recycles: int, diffusion_steps: int, esm_embeddings: bool, max_parallel: int = 1,
               out_logfile: str = None, err_logfile: str = None):
    super().__init__(builder, gpu_manager)
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.trunk_recycles = trunk_recycles
    self.diffusion_steps = diffusion_steps
    self.esm_embeddings = esm_embeddings
    self.max_parallel = max_parallel
    self.out_logfile = out_logfile
    self.err_logfile = err_logfile

  def run(self, client: docker.DockerClient = None):
    self.builder.set_volumes({
      self.input_dir: ChaiContainer.CONTAINER_INPUT_DIR,
      self.output_dir: ChaiContainer.CONTAINER_OUTPUT_DIR
    })
    tmp_dir = self.__split_fasta_files()
    containers = self.__build_containers()
    chunks = self.__get_fasta_chunks(tmp_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
      for container, chunk in zip(containers, chunks):  # Cycling on all chunks
        executor.submit(self.__run_on_fasta_list, fasta_basenames=chunk, container=container)

  def __split_fasta_files(self) -> str:
    tmp_dir = os.path.join(self.input_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    sequences: List[Protein] = []
    for f in os.listdir(self.input_dir):
      if f.endswith('.fa'):
        sequences += io.read_protein_mpnn_fasta(os.path.join(self.input_dir, f))
    for s in sequences:
      seq_fasta = os.path.join(tmp_dir, f'{s.name}.fa')
      with open(seq_fasta, 'w') as f:
        f.write(f'>protein|{s.name}\n{s.sequence()}\n')
        f.flush()
    return tmp_dir

  def __get_fasta_chunks(self, tmp_dir) -> List[List[str]]:
    files = [f for f in os.listdir(tmp_dir) if f.endswith('.fa')]
    n = math.ceil(len(files) / self.max_parallel)
    return [files[i:i + n] for i in range(0, len(files), n)]

  def __run_on_fasta_list(self, fasta_basenames: List[str], container: ContainerExecutor):
    if not fasta_basenames:  # Nothing to do
      return True
    with container as executor:
      print(f'[HOST {threading.current_thread().name}] Running predictions '
            f'for {fasta_basenames} on {executor.device().name}')
      for fasta_basename in fasta_basenames:
        res = self.__execute_command_for_single_fasta(container, fasta_basename)
        res &= executor.execute(f'/usr/bin/chmod 777 --recursive {ChaiContainer.CONTAINER_OUTPUT_DIR}')
      return res
      for fasta_basename in fasta_basenames:
        command = (f'python3 /chai-lab/chai_lab/main.py fold '
                   f'{fasta_basename}')

  def __build_containers(self) -> List[ContainerExecutor]:
    return [
      self.builder.set_logs(
        out_log_stream=f'{self.out_logfile}.{i}' if self.out_logfile else None,
        err_log_stream=f'{self.err_logfile}.{i}' if self.err_logfile else None,
      ).set_device(self.gpu_manager.next_device()).build()
      for i in range(self.max_parallel)
    ]

  def __execute_command_for_single_fasta(self, container: ContainerExecutor, fasta_basename: str):
    command = ('python3 /chai-lab/chai_lab/main.py fold '
               f'--'
               f'{os.path.join(ChaiContainer.CONTAINER_INPUT_DIR, fasta_basename)} '
               f'{os.path.join(ChaiContainer.CONTAINER_OUTPUT_DIR, os.path.splitext(fasta_basename)[0])}')
