import concurrent.futures
import os
import threading
from typing import List

from da4vid.containers.base import BaseContainer
from da4vid.containers.executor import ContainerExecutorBuilder, ContainerExecutor
from da4vid.gpus.cuda import CudaDeviceManager


class OmegaFoldContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/omegafold:latest'

  CONTAINER_MODEL_FOLDER = '/root/.cache/omegafold_ckpt'
  CONTAINER_INPUT_DIR = '/Omegafold/run/inputs'
  CONTAINER_OUTPUT_DIR = '/Omegafold/run/outputs'
  __MAIN_SCRIPT = '/Omegafold/main.py'

  def __init__(self, builder: ContainerExecutorBuilder, gpu_manager: CudaDeviceManager,
               model_dir: str, input_dir: str, output_dir: str, model_weights: str = "2",
               num_recycles: int = 5, max_parallel: int = 1, out_logfile: str = None,
               err_logfile: str = None):
    super().__init__(builder=builder, gpu_manager=gpu_manager)
    self.model_dir = model_dir
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.num_recycles = num_recycles
    self.model_weights = model_weights
    self.max_parallel = max_parallel
    self.out_logfile = out_logfile
    self.err_logfile = err_logfile
    # Setting builder data
    self.builder.set_volumes({
      self.input_dir: self.CONTAINER_INPUT_DIR,
      self.output_dir: self.CONTAINER_OUTPUT_DIR,
      self.model_dir: self.CONTAINER_MODEL_FOLDER,
    })

  def run(self) -> bool:
    containers = self.__build_containers()
    chunks = self.__get_fasta_chunks()
    res = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as tpe:
      futures = [tpe.submit(self.__create_and_run_container, container=container, fasta_basenames=chunk)
                 for container, chunk in zip(containers, chunks)]
      for future in concurrent.futures.as_completed(futures):
        res &= future.result()
        if not res:
          break
    return res

  def __create_and_run_container(self, container: ContainerExecutor, fasta_basenames: List[str]) -> bool:
    if not fasta_basenames:  # Nothing to do if no fastas should be executed
      return True
    with container as executor:
      print(f'[HOST {threading.current_thread().name}] Running {fasta_basenames} on device {executor.device().name}')
      # Run commands in the container
      res = self.__run_on_fasta_list(fasta_basenames, executor)
      # After everything is finished, change permissions on generated files
      executor.execute(f'/usr/bin/chmod 0777 --recursive {self.CONTAINER_OUTPUT_DIR}')
    return res

  def __run_on_fasta_list(self, fasta_basenames: List[str], container: ContainerExecutor) -> bool:
    for fasta_basename in fasta_basenames:
      command = self.__create_command(fasta_basename)
      if not container.execute(command):
        return False
    return True

  def __create_command(self, fasta_basename):
    fasta_no_ext = '.'.join(fasta_basename.split('.')[:-1])
    return (f'python3 {self.__MAIN_SCRIPT} '
            f'--model {self.model_weights} '
            f'--num_cycle {self.num_recycles} '
            f'{OmegaFoldContainer.CONTAINER_INPUT_DIR}/{fasta_basename} '
            f'{OmegaFoldContainer.CONTAINER_OUTPUT_DIR}/{fasta_no_ext}')

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

  def __build_containers(self) -> List[ContainerExecutor]:
    containers = []
    for i in range(self.max_parallel):
      self.builder.set_logs(
        out_log_stream=f'{self.out_logfile}.{i}' if self.out_logfile else None,
        err_log_stream=f'{self.err_logfile}.{i}' if self.err_logfile else None,
      ).set_device(self.gpu_manager.next_device())
      containers.append(self.builder.build())
    return containers
