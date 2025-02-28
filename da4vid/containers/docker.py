import logging
import sys
from typing import TextIO, Dict

import docker
from docker.types import Mount, DeviceRequest

from da4vid.containers import executor
from da4vid.containers.executor import ContainerExecutorBuilder, ContainerExecutor
from da4vid.gpus.cuda import CudaDevice


class DockerExecutor(executor.ContainerExecutor):
  """
  Class implementing the execution logic for docker containers.
  """

  class DockerImageNotFoundException(Exception):
    def __init__(self, message: str):
      super().__init__(message)

  class DockerContainerAlreadyCreatedException(Exception):
    def __init__(self, message: str):
      super().__init__(message)

  class DockerContainerNotRunningException(Exception):
    def __init__(self, message: str):
      super().__init__(message)

  def __init__(self, image: str, client: docker.DockerClient,
               device: CudaDevice, volumes: Dict[str, str] = None,
               out_stream: TextIO = sys.stdout, err_stream: TextIO = sys.stderr):
    super().__init__()
    self.image = image
    self.client = client
    self.__device = device
    self.volumes = volumes or {}
    self.out_stream = out_stream
    self.err_stream = err_stream
    self.__check_image()  # Checking image is present
    # Caching container
    self.__container = None

  def create(self, **kwargs) -> None:
    """
    Creates the actual container.
    :param kwargs: Unused arguments
    :raises DockerContainerAlreadyCreatedException: if this method is called when the container is already created
    """
    if self.__container is not None:
      raise self.DockerContainerAlreadyCreatedException('Container already created')
    self.__container = self.client.containers.run(
      image=self.image,
      mounts=[Mount(target, source, type='bind') for source, target in self.volumes.items()],
      device_requests=[DeviceRequest(capabilities=[['gpu']], device_ids=[f'{self.__device.index}'])],
      detach=True,
      auto_remove=True,
      tty=True
    )

  def execute(self, cmd: str, **kwargs) -> bool:
    """
    Executes a command in the container.
    :param cmd: The command which needs to be executed (as the argument of 'docker exec')
    :param kwargs: Unused arguments
    :return: True if the command has been successfully executed, or False otherwise
    :raises DockerContainerNotRunningException: if this method has been called before the 'create' method
    """
    self.__check_running()
    logging.debug(f'[HOST] Executing command {cmd}')
    exec_handle = self.client.api.exec_create(
      container=self.__container.id,
      cmd=cmd,
    )
    # Starting stream
    stream = self.client.api.exec_start(exec_handle, stream=True, demux=True)
    for out, err in stream:
      if out is not None:
        self.out_stream.write(f'{out.decode()}\n')
      if err is not None:
        self.err_stream.write(f'{err.decode()}\n')
    # Retrieving command return value
    res = self.client.api.exec_inspect(exec_handle['Id']).get('ExitCode')
    return res == 0  # 0 means OK

  def device(self) -> CudaDevice | None:
    return self.__device

  def stop(self, **kwargs) -> None:
    """
    Stops the container.
    :param kwargs: Unused arguments
    :raises DockerContainerNotRunningException: if this method has been called before the 'create' method
    """
    self.__check_running()
    self.__container.stop()
    self.__container = None
    # Conditionally closing streams
    if self.out_stream and self.out_stream != sys.stdout:
      self.out_stream.close()
    if self.err_stream and self.err_stream != sys.stderr:
      self.err_stream.close()

  def __check_image(self) -> None:
    if not self.client.images.list(filters={'reference': self.image}):
      raise self.DockerImageNotFoundException(f'Image not found: {self.image}')

  def __check_running(self) -> None:
    if not self.__container:
      raise self.DockerContainerNotRunningException(f'Container not created')


class DockerExecutorBuilder(ContainerExecutorBuilder):
  def __init__(self):
    super().__init__()
    self.device: CudaDevice | None = None
    self.image: str | None = None
    self.volumes: Dict[str, str] | None = None
    self.client: docker.DockerClient | None = None

  def set_device(self, device: CudaDevice | None = None) -> 'DockerExecutorBuilder':
    super().set_device(device)
    return self

  def set_logs(self, out_log_stream: str | TextIO, err_log_stream: str | TextIO) -> 'DockerExecutorBuilder':
    super().set_logs(out_log_stream, err_log_stream)
    return self

  def set_image(self, image: str) -> 'DockerExecutorBuilder':
    self.image = image
    return self

  def set_volumes(self, volumes: Dict[str, str] = None) -> 'DockerExecutorBuilder':
    self.volumes = volumes
    return self

  def set_client(self, client: docker.DockerClient) -> 'DockerExecutorBuilder':
    self.client = client
    return self

  def build(self) -> DockerExecutor:
    if self.image is None or self.client is None or self.device is None:
      raise self.ContainerBuilderException(f'Unable to build Docker container: (image: {self.image}, '
                                           f'client: {self.client} or device: {self.device} invalid')
    return DockerExecutor(
      image=self.image,
      client=self.client,
      device=self.device,
      volumes=self.volumes,
      out_stream=self.out_stream,
      err_stream=self.err_stream,
    )
