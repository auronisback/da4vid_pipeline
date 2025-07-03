from typing import List

from da4vid.containers.base import BaseContainer
from da4vid.containers.executor import ContainerExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager


class PestoContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/pesto:latest'

  CONTAINER_INPUT_FOLDER = '/PeSTo/inputs'
  CONTAINER_OUTPUT_FOLDER = '/PeSTo/outputs'
  INFERENCE_SCRIPT = '/PeSTo/run_inference.py'

  def __init__(self,  builder: ContainerExecutorBuilder, gpu_manager: CudaDeviceManager,
               input_folder: str, output_folder: str, out_logfile: str = None, err_logfile: str = None):
    super().__init__(
      builder=builder,
      gpu_manager=gpu_manager
    )
    self.input_folder = input_folder
    self.output_folder = output_folder
    self.out_logfile = out_logfile
    self.err_logfile = err_logfile

  def run(self) -> bool:
    self.builder.set_logs(self.out_logfile, self.err_logfile).set_volumes({
      self.input_folder: self.CONTAINER_INPUT_FOLDER,
      self.output_folder: self.CONTAINER_OUTPUT_FOLDER
    }).set_device(self.gpu_manager.next_device())
    res = True
    with self.builder.build() as executor:
      for cmd in self.__get_commands():
        res = executor.execute(cmd)
        if not res:
          break
      executor.execute(f'/bin/chmod 0777 --recursive {self.CONTAINER_OUTPUT_FOLDER}')
    return res

  def __get_commands(self) -> List[str]:
    return [
      f'python {self.INFERENCE_SCRIPT} {self.CONTAINER_INPUT_FOLDER} {self.CONTAINER_OUTPUT_FOLDER}'
    ]