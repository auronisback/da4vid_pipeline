import abc
import sys
from typing import Dict, Union, List, Tuple, Any, IO

import docker
from docker.types import DeviceRequest, Mount

from da4vid.containers.executor import ContainerExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager, CudaDevice


class BaseContainer(abc.ABC):
  """
  Abstract class for Docker containers.
  """

  def __init__(self, builder: ContainerExecutorBuilder, gpu_manager: CudaDeviceManager):
    self.builder = builder
    self.gpu_manager = gpu_manager

  @abc.abstractmethod
  def run(self) -> bool:
    pass
