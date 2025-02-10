import os
from typing import List

import docker

from da4vid.docker.base import BaseContainer, ContainerLogs
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.pipeline.steps import PipelineException


class CARBonAraContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/carbonara'

  SAMPLING_MAX = 'max'
  SAMPLING_SAMPLED = 'sampled'

  INPUT_DIR = '/data/inputs'
  OUTPUT_DIR = '/data/outputs'

  def __init__(self, input_dir: str, output_dir: str, client: docker.DockerClient,
               gpu_manager: CudaDeviceManager, num_sequences: int, imprint_ratio: float = .5,
               sampling_method: str = SAMPLING_SAMPLED, known_chains: List[str] | None = None,
               known_positions: List[int] | None = None, unknown_positions: List[int] | None = None,
               ignored_amino_acids: List[str] | None = None, ignore_het_atm: bool = False,
               ignore_water: bool = False, image: str = DEFAULT_IMAGE, out_logfile: str = None,
               err_logfile: str = None):
    super().__init__(
      image=image,
      entrypoint='/bin/bash',
      volumes={
        input_dir: self.INPUT_DIR,
        output_dir: self.OUTPUT_DIR
      },
      client=client,
      gpu_manager=gpu_manager
    )
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.num_sequences = num_sequences
    self.imprint_ratio = imprint_ratio
    self.sampling_method = sampling_method
    self.known_chains = known_chains
    self.known_positions = known_positions
    self.unknown_positions = unknown_positions
    self.ignored_amino_acids = ignored_amino_acids
    self.ignore_het_atm = ignore_het_atm
    self.ignore_water = ignore_water
    self.out_logfile = out_logfile
    self.err_logfile = err_logfile
    # Caching complex arguments
    self.__ignored_amino_acids_str = None
    self.__known_chains_str = None
    self.__known_positions_str = None
    self.__unknown_positions_str = None

  def run(self) -> bool:
    self.__ignored_amino_acids_str = '' if self.ignored_amino_acids is None else (
        '--ignored_amino_acids "' + ','.join(self.ignored_amino_acids) + '"')
    self.__known_chains_str = '' if self.known_chains is None else (
        '--known_chains "' + ','.join(self.known_chains) + '"')
    self.__known_positions_str = '' if self.known_positions is None else (
        '--known_positions "' + ','.join([str(p) for p in self.known_positions]) + '"')
    self.__unknown_positions_str = '' if self.unknown_positions is None else (
        '--unknown_positions "' + ','.join([str(p) for p in self.unknown_positions]) + '"')
    with ContainerLogs(self.out_logfile, self.err_logfile) as logs:
      res = True
      container, device = super()._create_container()
      for f in os.listdir(self.input_dir):
        if f.endswith('.pdb'):
          res &= super()._execute_command(container, self.__get_command_for_backbone(f),
                                          output_log=logs.out, error_log=logs.err)
          if not res:
            break
      super()._stop_container(container)
      return res

  def __get_command_for_backbone(self, backbone_basename: str) -> str:
    return (f'carbonara '
            f'--num_sequences {self.num_sequences} '
            f'--imprint_ratio {self.imprint_ratio} '
            f'--sampling_method {self.sampling_method} '
            f'{self.__known_chains_str} '
            f'{self.__known_positions_str} '
            f'{self.__unknown_positions_str} '
            f'{self.__ignored_amino_acids_str} '
            f'{f"--ignore_hetatm {self.ignore_het_atm} " if self.ignore_het_atm else ""}'
            f'{f"--ignore_water {self.ignore_water} " if self.ignore_water else ""}'
            f'{os.path.join(self.INPUT_DIR, backbone_basename)} {self.OUTPUT_DIR}')
