import concurrent.futures
import math
import os.path
from typing import List

import docker

from da4vid import io
from da4vid.docker.base import BaseContainer
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.model import Protein


class ChaiContainer(BaseContainer):

  INPUT_DIR = '/chai-lab/inputs'
  OUTPUT_DIR = '/chai-lab/outputs'

  def __init__(self, input_dir: str, output_dir: str, client: docker.DockerClient, gpu_manager: CudaDeviceManager,
               trunk_recycles: int, diffusion_steps: int, esm_embeddings: bool, max_parallel: int = 1):
    super().__init__(
      image='ameg/chai_gradio',
      entrypoint='/bin/bash',
      volumes={
        input_dir: self.INPUT_DIR,
        output_dir: self.OUTPUT_DIR
      },
      client=client,
      gpu_manager=gpu_manager
    )
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.trunk_recycles = trunk_recycles
    self.diffusion_steps = diffusion_steps
    self.esm_embeddings = esm_embeddings
    self.max_parallel = max_parallel

  def run(self, client: docker.DockerClient = None):
    tmp_dir = self.__split_fasta_files()
    container = super()._create_container()
    chunks = self.__get_fasta_chunks(tmp_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
      for i, chunk in enumerate(chunks):  # Cycling on all chunks
        device = f'cuda:{i % 2}'
        executor.submit(self.__run_on_fasta_list, fasta_basenames=chunk, container=container, device=device)

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

  def __run_on_fasta_list(self, fasta_basenames: List[str], container, device: str):
    print(f'Running {fasta_basenames} on device {device}')
    for fasta_basename in fasta_basenames:
      command = (f'python3 /chai-lab/run_inference.py '
                 f'--')
