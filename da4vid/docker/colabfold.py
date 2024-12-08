import concurrent.futures
import os
import sys
from typing import List, Tuple

import docker

from da4vid.docker.base import BaseContainer


class ColabFoldContainer(BaseContainer):
  MODELS_FOLDER = '/colabfold/weights'
  INPUT_DIR = '/colabfold/inputs'
  OUTPUT_DIR = '/colabfold/outputs'
  COPY_MSA_SCRIPT = '/colabfold/scripts'
  COLABFOLD_BATCH_COMMAND = '/usr/local/envs/colabfold/bin/colabfold_batch'

  COLABFOLD_API_URL = 'https://api.colabfold.com'
  MODEL_NAMES = ['auto', 'alphafold2', 'alphafold2_ptm,alphafold2_multimer_v1', 'alphafold2_multimer_v2',
                 'alphafold2_multimer_v3', 'deepfold_v1']

  def __init__(self, model_dir: str, input_dir: str, output_dir: str,
               num_recycle: int = 5, zip_outputs: bool = False,
               model_name: str = MODEL_NAMES[0], num_models: int = 5,
               msa_host_url: str = COLABFOLD_API_URL, max_parallel: int = 1,
               image: str = 'da4vid/colabfold:latest'):
    super().__init__(
      image=image,
      entrypoint='/bin/bash',
      with_gpus=True,
      volumes={
        model_dir: ColabFoldContainer.MODELS_FOLDER,
        input_dir: ColabFoldContainer.INPUT_DIR,
        output_dir: ColabFoldContainer.OUTPUT_DIR
      },
      detach=True
    )
    self.num_recycle = num_recycle
    self.zip_outputs = zip_outputs
    # Checking valid model
    if model_name not in ColabFoldContainer.MODEL_NAMES:
      raise ValueError(f'given model "{model_name}" is invalid '
                       f'(choices: {", ".join(ColabFoldContainer.MODEL_NAMES)})')
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.model_name = model_name
    self.num_models = num_models
    # Initializing list of MSA endpoint URLs
    self.msa_host_url = msa_host_url
    # Setting number of max parallel jobs
    self.max_parallel = max_parallel

  def run(self, client: docker.DockerClient = None):
    # TODO: Differentiate containers in order to use different devices when refactoring with CUDAManager
    container = super()._create_container(client)
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
      chunks = self.__get_fasta_chunks()
      for i, chunk in enumerate(chunks):
        device = f'cuda:{i % 2}'
        executor.submit(self.__run_on_fasta_list, fasta_basenames=chunk,
                        container=container, device=device)
    # Modifying permissions after works
    super()._execute_command(container,
                             f'/usr/bin/chmod 777 --recursive {ColabFoldContainer.OUTPUT_DIR}',
                             file=sys.stdout)
    super()._stop_container(container)
    return True

  def __run_on_fasta_list(self, fasta_basenames: List[str], container, device: str):
    print(f'Running predictions for {fasta_basenames} on {device}')
    for fasta_basename in fasta_basenames:
      for command in self.__commands_for_single_fasta(fasta_basename):
        super()._execute_command(container, command, file=sys.stdout)

  def __get_fasta_chunks(self) -> List[List[str]]:
    files = [f for f in os.listdir(self.input_dir) if f.endswith('.fa')]
    n = len(files) // self.max_parallel
    rem = len(files) % self.max_parallel
    ff = []
    start = 0
    for i in range(self.max_parallel):
      if i < rem:
        ff.append(files[start:start+n+1])
        start = start+n+1
      else:
        ff.append(files[start:start+n])
        start = start + n
    return ff

  def __commands_for_single_fasta(self, f: str) -> List[str]:
    return [
      self.__create_msa_fasta_command(f),
      self.__msa_only_command(f),
      self.__copy_msa_command(f),
      self.__remove_msa_fasta_command(f),
      self.__prediction_command(f),
    ]

  def __create_msa_fasta_command(self, f: str) -> str:
    input_fasta, _ = self.__get_input_and_output_folder(f)
    tmp_fasta = self.__get_tmp_msa_fasta_path(f)
    return f'/bin/bash -c "/usr/bin/head -n 2 {input_fasta} > {tmp_fasta}"'

  def __msa_only_command(self, f: str) -> str:
    _, output_folder = self.__get_input_and_output_folder(f)
    tmp_fasta = self.__get_tmp_msa_fasta_path(f)
    return f'{self.COLABFOLD_BATCH_COMMAND} --msa-only {tmp_fasta} {output_folder}'

  def __copy_msa_command(self, f: str) -> str:
    input_fasta, output_folder = self.__get_input_and_output_folder(f)
    return f'python {self.COPY_MSA_SCRIPT} {input_fasta} {output_folder} 1'

  def __prediction_command(self, f: str) -> str:
    input_fasta, output_folder = self.__get_input_and_output_folder(f)
    return (f'{self.COLABFOLD_BATCH_COMMAND} '
            f'--data {self.MODELS_FOLDER} '
            f'--model-type {self.model_name} '
            f'--num-recycle {self.num_recycle} '
            f'--num-models {self.num_models} '
            f'--host-url {self.msa_host_url} '
            f'{"--zip-outputs" if self.zip_outputs else ""} '
            f'{input_fasta} {output_folder}')

  def __remove_msa_fasta_command(self, f: str) -> str:
    return f'/usr/bin/rm {self.__get_tmp_msa_fasta_path(f)}'

  def __get_input_and_output_folder(self, f: str) -> Tuple[str, str]:
    return os.path.join(self.INPUT_DIR, f), os.path.join(self.OUTPUT_DIR, '.'.join(f.split('.')[:-1]))

  def __get_tmp_msa_fasta_path(self, f: str) -> str:
    return os.path.join(self.INPUT_DIR, f'msa_{f}')
