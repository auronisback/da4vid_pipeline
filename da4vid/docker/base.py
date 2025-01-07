import abc
import sys
from typing import Dict, Union, List, Tuple, Any, IO

import docker
from docker.types import DeviceRequest, Mount

from da4vid.gpus.cuda import CudaDeviceManager, CudaDevice


class BaseContainer(abc.ABC):
  """
  Abstract class for Docker containers.
  """

  def __init__(self, image: str, entrypoint: str,
               client: docker.DockerClient, gpu_manager: CudaDeviceManager,
               commands: Union[List[str], str] = None,
               volumes: Dict[str, str] = None):
    if commands is None:
      commands = []
    self.image = image
    self.entrypoint = entrypoint
    self.client = client
    self.gpu_manager = gpu_manager
    self.commands = commands
    if volumes is None:
      volumes = {}
    self.volumes = volumes
    self.auto_remove = True

  def _create_container(self) -> Tuple[Any, CudaDevice]:
    self.__check_image()
    device = self.gpu_manager.next_device()
    container = self.client.containers.run(
      image=self.image,
      entrypoint=self.entrypoint,
      mounts=[Mount(target, source, type='bind') for source, target in self.volumes.items()],
      device_requests=[DeviceRequest(capabilities=[['gpu']],
                                     device_ids=[f'{device.index}'])],
      detach=True,
      auto_remove=self.auto_remove,
      tty=True
    )
    return container, device

  def _run_container(self, output_log: IO = sys.stdout, error_log: IO = sys.stderr) -> bool:
    """
    Runs a container to execute all specified commands in this object, closing it afterwards.
    :param output_log: The file on which print output logs. Defaults to STDOUT
    :param error_log: The file on which print error logs. Defaults to STDERR
    :return: True if the container has been successfully executed, or false otherwise
    """
    container, device = self._create_container()
    ok = True
    for cmd in self.commands:
      ok = self._execute_command(container, cmd, output_log, error_log)
      if not ok:  # Error in executing the command
        break
    # Stopping the container
    self._stop_container(container)
    return ok

  def _execute_command(self, container, command: str,
                       output_log: IO = sys.stdout, error_log: IO = sys.stderr) -> bool:
    """
    Executes a command in the container.
    :param container: The container on which execute the command
    :param command: The command which needs to be executed (as the argument of 'docker exec')
    :param output_log: The file on which print output logs. Defaults to STDOUT
    :param error_log: The file on which print error logs. Defaults to STDERR
    :return: True if the command has been successfully executed, or False otherwise
    """
    print(f'Executing command: {command}')
    # Creating the handle for the exec command
    exec_handle = self.client.api.exec_create(
      container=container.id,
      cmd=command
    )
    # Starting the command and streaming
    stream = self.client.api.exec_start(exec_handle, stream=True, demux=True)
    for out, err in stream:
      if out is not None:
        print(out.decode(), file=output_log)
      if err is not None:
        print(err.decode(), file=error_log)
    # Retrieving command return value
    res = self.client.api.exec_inspect(exec_handle['Id']).get('ExitCode')
    return res == 0  # If it is 0, it's all ok

  @staticmethod
  def _stop_container(container) -> None:
    """
    Stops a container.
    :param container: The container to stop
    """
    container.stop()

  class DockerImageNotFoundException(Exception):
    def __init__(self, message: str):
      super().__init__(message)

  def __check_image(self) -> None:
    if not self.client.images.list(filters={'reference': self.image}):
      raise self.DockerImageNotFoundException(f'Image not found: {self.image}')


class ContainerLogs:
  """
  Class allowing the usage of logs with the "with" keyword.
  """
  def __init__(self, out_logfile: str, err_logfile: str):
    """
    Initialize the container log object.
    :param out_logfile: Path to the file used to log container STDOUT. If None, host
                        stdout will be used
    :param err_logfile: Path to the file used to log container STDERR. If None, host
                        stdout will be used
    """
    self.__of = out_logfile
    self.__ef = err_logfile
    self.__out_logfile = None
    self.__err_logfile = None

  def open(self):
    self.__out_logfile = open(self.__of, 'w') if self.__of else sys.stdout
    self.__err_logfile = open(self.__ef, 'w') if self.__ef else sys.stderr

  @property
  def out_logfile(self) -> IO:
    return self.__out_logfile

  @property
  def err_logfile(self) -> IO:
    return self.__err_logfile

  def close(self):
    # Close the files if they were given
    if self.__of:
      self.__out_logfile.close()
    if self.__ef:
      self.__err_logfile.close()

  def __enter__(self):
    self.open()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

