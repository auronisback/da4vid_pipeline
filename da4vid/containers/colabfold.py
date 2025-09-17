import concurrent.futures
import os
import threading
from typing import List

from da4vid.containers.base import BaseContainer
from da4vid.containers.executor import ContainerExecutorBuilder, ContainerExecutor
from da4vid.gpus.cuda import CudaDeviceManager


class ColabFoldContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/colabfold:latest'

  CONTAINER_MODELS_DIR = '/colabfold/weights'
  CONTAINER_INPUT_DIR = '/colabfold/inputs'
  CONTAINER_OUTPUT_DIR = '/colabfold/outputs'
  CONTAINER_DB_DIR = '/colabfold/msa_dbs'

  # Scripts and commands
  __COPY_MSA_SCRIPT = '/colabfold/scripts/copy_msa.py'
  __COLABFOLD_BATCH_COMMAND = '/usr/local/envs/colabfold/bin/colabfold_batch'
  __COLABFOLD_SEARCH_COMMAND = '/usr/local/envs/colabfold/bin/colabfold_search'
  __MMSEQS_COMMAND = '/usr/local/envs/colabfold/bin/mmseqs'

  # Default API URL
  COLABFOLD_API_URL = 'https://api.colabfold.com'

  # Available ColabFold models
  MODEL_NAMES = ['auto', 'alphafold2', 'alphafold2_ptm,alphafold2_multimer_v1', 'alphafold2_multimer_v2',
                 'alphafold2_multimer_v3', 'deepfold_v1']

  def __init__(self, builder: ContainerExecutorBuilder, gpu_manager: CudaDeviceManager,
               model_dir: str, input_dir: str, output_dir: str, num_recycle: int = 5,
               zip_outputs: bool = False, model_name: str = MODEL_NAMES[0], num_models: int = 5,
               local_msa_db: str = None,
               msa_host_url: str = COLABFOLD_API_URL, max_parallel: int = 1,
               out_logfile: str = None, err_logfile: str = None):
    super().__init__(builder, gpu_manager)
    # Checking valid model
    if model_name not in ColabFoldContainer.MODEL_NAMES:
      raise ValueError(f'given model "{model_name}" is invalid '
                       f'(choices: {", ".join(ColabFoldContainer.MODEL_NAMES)})')
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.model_dir = model_dir
    self.model_name = model_name
    self.num_models = num_models
    self.num_recycle = num_recycle
    self.zip_outputs = zip_outputs
    self.out_logfile = out_logfile
    self.err_logfile = err_logfile
    # Checking if local DB is used
    self.local_msa_db = local_msa_db
    # Initializing list of MSA endpoint URLs
    self.msa_host_url = msa_host_url
    # Setting number of max parallel jobs
    self.max_parallel = max_parallel

  def run(self) -> bool:
    volumes = {
      self.model_dir: ColabFoldContainer.CONTAINER_MODELS_DIR,
      self.input_dir: ColabFoldContainer.CONTAINER_INPUT_DIR,
      self.output_dir: ColabFoldContainer.CONTAINER_OUTPUT_DIR
    }
    if self.local_msa_db:
      volumes[self.local_msa_db] = ColabFoldContainer.CONTAINER_DB_DIR
    self.builder.set_volumes(volumes)
    res = True
    containers = self.__build_containers()
    chunks = self.__get_fasta_chunks()
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as tpe:
      futures = [tpe.submit(self.__create_and_execute_container, container=container, fasta_basenames=chunk)
                 for container, chunk in zip(containers, chunks)]
      for future in concurrent.futures.as_completed(futures):
        res = future.result()
        if not res:
          break
    return res

  def __create_and_execute_container(self, container: ContainerExecutor, fasta_basenames: List[str]) -> bool:
    if not fasta_basenames:  # Nothing to do if no FASTA file has to be evaluated
      return True
    with container as executor:
      print(f'[HOST {threading.current_thread().name}] Running predictions '
            f'for {fasta_basenames} on {executor.device().name}')
      res = self.__run_on_fasta_list(fasta_basenames, container)
      res &= executor.execute(f'/usr/bin/chmod 777 --recursive {ColabFoldContainer.CONTAINER_OUTPUT_DIR}')
    return res

  def __run_on_fasta_list(self, fasta_basenames: List[str], container) -> bool:
    for fasta_basename in fasta_basenames:
      res = self.__execute_commands_for_single_fasta(container, fasta_basename)
      if not res:
        return False
    return True

  def __get_fasta_chunks(self) -> List[List[str]]:
    files = [f for f in os.listdir(self.input_dir) if f.endswith('.fa')]
    n = len(files) // self.max_parallel
    rem = len(files) % self.max_parallel
    ff = []
    start = 0
    for i in range(self.max_parallel):
      end = start + n + 1 if i < rem else start + n
      ff.append(files[start:end])
      start = end
    return ff

  def __execute_commands_for_single_fasta(self, container, f: str) -> bool:
    if self.local_msa_db:
      commands = self.__get_commands_for_local_msa(f)
    else:
      commands = self.__get_commands_for_remote_msa(f)
    res = True
    for command in commands:
      res = container.execute(command)
      if not res:
        break
    return res

  def __get_commands_for_local_msa(self, f: str) -> List[str]:
    return [
      self.__create_msa_search_command(f),
      self.__prediction_command(f)
    ]

  def __get_commands_for_remote_msa(self, f: str) -> List[str]:
    return [
      self.__create_msa_fasta_command(f),
      self.__msa_only_command(f),
      self.__copy_msa_command(f),
      self.__remove_msa_fasta_command(f),
      self.__prediction_command(f)
    ]

  def __create_msa_search_command(self, fasta: str) -> str:
    fasta_path = self.__get_input_fasta(fasta)
    msa_out = self.__get_local_msa_out(fasta)
    return f'colabfold_batch --mmseqs {self.__MMSEQS_COMMAND} {fasta_path} {self.CONTAINER_DB_DIR} {msa_out}'

  def __create_msa_fasta_command(self, f: str) -> str:
    input_fasta = self.__get_input_fasta(f)
    tmp_fasta = self.__get_tmp_msa_fasta_path(f)
    return f'/bin/bash -c "/usr/bin/head -n 2 {input_fasta} > {tmp_fasta}"'

  def __msa_only_command(self, f: str) -> str:
    output_folder = self.__get_output_folder_for_fasta(f)
    tmp_fasta = self.__get_tmp_msa_fasta_path(f)
    return f'{self.__COLABFOLD_BATCH_COMMAND} --msa-only {tmp_fasta} {output_folder}'

  def __copy_msa_command(self, f: str) -> str:
    input_fasta = self.__get_input_fasta(f)
    output_folder = self.__get_output_folder_for_fasta(f)
    # Extracting MSA index: searching an env folder
    msa_index = self.__get_msa_index(f)
    return f'python {self.__COPY_MSA_SCRIPT} {input_fasta} {output_folder} {msa_index}'

  def __get_msa_index(self, fasta_basename: str) -> int:
    with open(os.path.join(self.input_dir, fasta_basename), 'r') as f:
      return int(f.readline().strip().split('_')[-1])

  def __prediction_command(self, f: str) -> str:
    if self.local_msa_db:
      input_data = self.__get_local_msa_out(f)
    else:
      input_data = self.__get_input_fasta(f)
    output_folder = self.__get_output_folder_for_fasta(f)
    return (f'{self.__COLABFOLD_BATCH_COMMAND} '
            f'--data {self.CONTAINER_MODELS_DIR} '
            f'--model-type {self.model_name} '
            f'--num-recycle {self.num_recycle} '
            f'--num-models {self.num_models} '
            f'--host-url {self.msa_host_url} '
            f'{"--zip-outputs" if self.zip_outputs else ""} '
            f'{input_data} {output_folder}')

  def __remove_msa_fasta_command(self, f: str) -> str:
    return f'/usr/bin/rm {self.__get_tmp_msa_fasta_path(f)}'

  def __get_input_fasta(self, f: str) -> str:
    return os.path.join(self.CONTAINER_INPUT_DIR, f)

  def __get_output_folder_for_fasta(self, f: str) -> str:
    return os.path.join(self.CONTAINER_OUTPUT_DIR, '.'.join(f.split('.')[:-1]))

  def __get_tmp_msa_fasta_path(self, f: str) -> str:
    return os.path.join(self.CONTAINER_INPUT_DIR, f'msa_{f}')

  def __get_local_msa_out(self, f: str) -> str:
    return os.path.join(self.CONTAINER_OUTPUT_DIR, 'msa', '.'.join(os.path.basename(f).split('.')[:-1]))

  def __build_containers(self) -> List[ContainerExecutor]:
    containers = []
    for i in range(self.max_parallel):
      self.builder.set_logs(
        out_log_stream=f'{self.out_logfile}.{i}' if self.out_logfile else None,
        err_log_stream=f'{self.err_logfile}.{i}' if self.err_logfile else None,
      ).set_device(self.gpu_manager.next_device())
      containers.append(self.builder.build())
    return containers
