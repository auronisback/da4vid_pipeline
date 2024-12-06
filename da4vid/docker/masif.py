from da4vid.docker.base import BaseContainer


class MasifContainer(BaseContainer):
  def __init__(self):
    super().__init__(
      image='ameg/masif:latest',
      entrypoint='/bin/bash',
      volumes={
        
      },
      with_gpus=True,
      detach=True
    )
