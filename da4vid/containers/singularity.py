import logging
import os
import subprocess
import sys
from typing import Dict, TextIO, List

from spython.main.base import Client

from da4vid.containers.executor import ContainerExecutor, ContainerExecutorBuilder
from da4vid.gpus.cuda import CudaDevice


class SingularityExecutor(ContainerExecutor):
  class SifFileNotFoundException(Exception):
    def __init__(self, message: str):
      super().__init__(message)

  class SingularityContainerAlreadyCreatedException(Exception):
    def __init__(self, message: str):
      super().__init__(message)

  class SingularityContainerNotRunningException(Exception):
    def __init__(self, message: str):
      super().__init__(message)

  def __init__(self, sif_path: str, client: Client, device: CudaDevice, volumes: Dict[str, str] = None,
               out_stream: TextIO = sys.stdout, err_stream: TextIO = sys.stderr, preserved_quotes: List[str] = None):
    self.sif_path = sif_path
    self.client = client
    self.__device = device
    self.volumes = volumes or {}
    self.out_stream = out_stream
    self.err_stream = err_stream
    self.preserved_quotes = preserved_quotes or []
    self.__check_sif_path()  # Check if the container SIF file exists and can be read
    # Caching container
    self.__container = None

  def create(self, **kwargs) -> None:
    if self.__container:
      raise self.SingularityContainerAlreadyCreatedException('Container already exists')
    volumes = []
    for host_folder, container_folder in self.volumes.items():
      volumes += ['--bind', f'{host_folder}:{container_folder}']
    self.__container = self.client.instance(self.sif_path, options=[
      *volumes, '--nv'
    ])

  def execute(self, cmd: str, **kwargs) -> bool:
    if not self.__container:
      raise self.SingularityContainerNotRunningException('Container not created')
    try:
      command = self.__split_command(cmd)
      logging.debug(f'[HOST] Executing command: {command}')
      exc = self.client.execute(self.__container, command, options=[
        '--env', f'CUDA_VISIBLE_DEVICES={self.__device.index}'
      ], stream=True, stream_type='both')  # SPython merges out and err :'(
      for line in exc:
        self.out_stream.write(line)
    except subprocess.CalledProcessError as e:
      logging.warning(f'Error executing command: {e}')
      return False
    return True

  def stop(self, **kwargs) -> None:
    if not self.__container:
      raise self.SingularityContainerNotRunningException('Container not created')
    self.__container.stop()
    self.__container = None

  def device(self) -> CudaDevice | None:
    return self.__device

  def __check_sif_path(self) -> None:
    if not os.path.isfile(self.sif_path):
      raise self.SifFileNotFoundException(f'File {self.sif_path} not exists or is not a regular file')

  def __split_command(self, cmd: str) -> List[str]:
    """
    Splits a command string into an array of various arguments.
    :param cmd: The command string
    :return: The list of arguments for singularity exec command
    """
    command = []
    command_part = ''
    stack = []
    for c in cmd:
      if c == '"' or c == "'":
        # Different quote symbol: appending
        if not stack or stack[-1] != c:
          stack.append(c)
        else:
          stack.pop()
        if c in self.preserved_quotes:
          command_part += c
      elif c == ' ':
        if not stack:  # Empty stack: outside of quotes, found another piece of the command
          command.append(command_part)
          command_part = ''
        else:
          command_part += c
      else:  # Normal character, appending
        command_part += c
    # Adding last command if needed
    if command_part:
      command.append(command_part)
    return command


class SingularityExecutorBuilder(ContainerExecutorBuilder):

  def __init__(self):
    super().__init__()
    self.__preserved_quotes = []
    self.sif_path = None
    self.client = None
    self.volumes = None
    self.device = None

  def set_sif_path(self, sif_path: str) -> 'SingularityExecutorBuilder':
    self.sif_path = sif_path
    return self

  def set_client(self, client: Client) -> 'SingularityExecutorBuilder':
    self.client = client
    return self

  def set_logs(self, out_log_stream: str | TextIO, err_log_stream: str | TextIO) -> 'SingularityExecutorBuilder':
    super().set_logs(out_log_stream, err_log_stream)
    return self

  def set_device(self, device: CudaDevice | None = None) -> 'SingularityExecutorBuilder':
    super().set_device(device)
    return self

  def preserve_quotes_in_cmds(self, preserved_quotes: List[str]) -> 'SingularityExecutorBuilder':
    self.__preserved_quotes = preserved_quotes
    return self

  def set_volumes(self, volumes: Dict[str, str]) -> 'SingularityExecutorBuilder':
    super().set_volumes(volumes)
    return self

  def build(self) -> ContainerExecutor:
    if self.sif_path is None or self.client is None or self.device is None:
      raise self.ContainerBuilderException(f'Unable to build Singularity container: (SIF path: {self.sif_path}, '
                                           f'client: {self.client} or device: {self.device} invalid')
    return SingularityExecutor(
      sif_path=self.sif_path,
      client=self.client,
      device=self.device,
      volumes=self.volumes,
      out_stream=self.out_stream,
      err_stream=self.err_stream,
      preserved_quotes=self.__preserved_quotes
    )
