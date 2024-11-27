import abc
from typing import Dict, Union, List

import docker
from docker.types import DeviceRequest, Mount


class BaseContainer(abc.ABC):

  def __init__(self, image: str, entrypoint: str, commands: Union[List[str], str] = None,
               volumes: Dict[str, str] = None, with_gpus: bool = False, detach: bool = False):
    if commands is None:
      commands = []
    self.image = image
    self.entrypoint = entrypoint
    self.commands = commands
    if volumes is None:
      volumes = {}
    self.volumes = volumes
    self.with_gpus = with_gpus
    self.auto_remove = True
    self.detach = detach
    self.client: docker.DockerClient | None = None

  def _create_container(self, client: docker.client.DockerClient):
    if client is None:
      client = self.client = docker.from_env()
    return client.containers.run(
      image=self.image,
      entrypoint=self.entrypoint,
      mounts=[Mount(target, source, type='bind') for source, target in self.volumes.items()],
      device_requests=[DeviceRequest(capabilities=[['gpu']])] if self.with_gpus else [],
      detach=self.detach,
      auto_remove=self.auto_remove,
      tty=True
    )

  def _run_container(self, client: docker.client.DockerClient = None):
    container = self._create_container(client)
    for cmd in self.commands:
      _, out = container.exec_run(cmd, stream=True)
      for line in out:
        print(line.decode().strip())
    self._stop_container(container)
    return True  # TODO: Fix this trying to check response of commands

  @staticmethod
  def _execute_command(container, command, file) -> bool:
    _, out = container.exec_run(command, stream=True)
    for line in out:
      print(line.decode().strip(), file=file)
    return True

  def _stop_container(self, container):
    container.stop()
    if self.client is not None:
      self.client.close()
