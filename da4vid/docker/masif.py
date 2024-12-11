import docker

from da4vid.docker.base import BaseContainer
from da4vid.gpus.cuda import CudaDeviceManager


class MasifContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/masif:latest'

  def __init__(self, client: docker.DockerClient, gpu_manager: CudaDeviceManager, image: str = DEFAULT_IMAGE):
    super().__init__(
      image=image,
      entrypoint='/bin/bash',
      volumes={
        
      },
      client=client,
      gpu_manager=gpu_manager
    )
