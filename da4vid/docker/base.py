import abc
from typing import Dict, Union, List

import docker
from docker.types import DeviceRequest, Mount


class BaseContainer(abc.ABC):

  def __init__(self, image: str, entrypoint: str, commands: Union[List[str], str] = [],
               volumes: Dict[str, str] = None, with_gpus: bool = False, detach: bool = False):
    self.image = image
    self.entrypoint = entrypoint
    self.commands = commands
    if volumes is None:
      volumes = {}
    self.volumes = volumes
    self.with_gpus = with_gpus
    self.auto_remove = True
    self.detach = detach

  def _run_container(self, client: docker.client.DockerClient = None):
    if client is None:
      client = docker.from_env()
    container = client.containers.run(
      image=self.image,
      entrypoint=self.entrypoint,
      mounts=[Mount(target, source, type='bind') for source, target in self.volumes.items()],
      device_requests=[DeviceRequest(capabilities=[['gpu']])] if self.with_gpus else [],
      detach=self.detach,
      auto_remove=self.auto_remove,
      tty=True
    )
    for cmd in self.commands:
      _, out = container.exec_run(cmd, stream=True)
      for line in out:
        print(line.decode().strip())
    container.stop()
    return True  # Fix this trying to check response of commands
