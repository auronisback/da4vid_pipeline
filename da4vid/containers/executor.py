import abc
import sys
from typing import TextIO, Dict

from da4vid.gpus.cuda import CudaDevice


class ContainerExecutor(abc.ABC):
  """
  Interface abstracting a container executor.
  Provides definitions of method which can be used when executing a container.
  """

  @abc.abstractmethod
  def create(self, **kwargs) -> None:
    pass

  @abc.abstractmethod
  def execute(self, cmd: str, **kwargs) -> bool:
    pass

  @abc.abstractmethod
  def stop(self, **kwargs) -> None:
    pass

  @abc.abstractmethod
  def device(self) -> CudaDevice | None:
    pass

  def __enter__(self) -> 'ContainerExecutor':
    self.create()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()


class ContainerExecutorBuilder(abc.ABC):
  """
  Class abstracting builders for container executors.
  """

  def __init__(self):
    self.device = None
    self.volumes = None
    self.err_stream = sys.stdout
    self.out_stream = sys.stderr

  def set_volumes(self, volumes: Dict[str, str]) -> 'ContainerExecutorBuilder':
    self.volumes = volumes
    return self

  def set_device(self, device: CudaDevice | None = None) -> 'ContainerExecutorBuilder':
    self.device = device
    return self

  def set_logs(self, out_log_stream: str | TextIO, err_log_stream: str | TextIO) -> 'ContainerExecutorBuilder':
    if out_log_stream is None:
      self.out_stream = sys.stdout
    elif isinstance(out_log_stream, str):
      self.out_stream = open(out_log_stream, 'w')
    else:
      self.out_stream = out_log_stream
    if err_log_stream is None:
      self.err_stream = sys.stderr
    elif isinstance(err_log_stream, str):
      self.err_stream = open(err_log_stream, 'w')
    else:
      self.err_stream = sys.stderr
    return self

  @abc.abstractmethod
  def build(self) -> ContainerExecutor:
    pass

  class ContainerBuilderException(Exception):
    def __init__(self, message: str):
      super().__init__(message)