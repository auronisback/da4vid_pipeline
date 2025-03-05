import logging
import os
from typing import List

from da4vid.containers.base import BaseContainer
from da4vid.containers.executor import ContainerExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager


class CARBonAraContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/carbonara'

  SAMPLING_MAX = 'max'
  SAMPLING_SAMPLED = 'sampled'

  CONTAINER_INPUT_DIR = '/data/inputs'
  CONTAINER_OUTPUT_DIR = '/data/outputs'

  def __init__(self, builder: ContainerExecutorBuilder, gpu_manager: CudaDeviceManager,
               input_dir: str, output_dir: str, num_sequences: int, imprint_ratio: float = .5,
               sampling_method: str = SAMPLING_SAMPLED, known_chains: List[str] | None = None,
               known_positions: List[int] | None = None, unknown_positions: List[int] | None = None,
               ignored_amino_acids: List[str] | None = None, ignore_het_atm: bool = False,
               ignore_water: bool = False, out_logfile: str = None,
               err_logfile: str = None):
    super().__init__(
      builder=builder,
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
    self.builder.set_logs(self.out_logfile, self.err_logfile).set_volumes({
      self.input_dir: self.CONTAINER_INPUT_DIR,
      self.output_dir: self.CONTAINER_OUTPUT_DIR
    }).set_device(self.gpu_manager.next_device())
    self.__ignored_amino_acids_str = '' if self.ignored_amino_acids is None else (
        '--ignored_amino_acids "' + ','.join(self.ignored_amino_acids) + '"')
    self.__known_chains_str = '' if self.known_chains is None else (
        '--known_chains "' + ','.join(self.known_chains) + '"')
    self.__known_positions_str = '' if self.known_positions is None else (
        '--known_positions "' + ','.join([str(p) for p in self.known_positions]) + '"')
    self.__unknown_positions_str = '' if self.unknown_positions is None else (
        '--unknown_positions "' + ','.join([str(p) for p in self.unknown_positions]) + '"')
    with self.builder.build() as executor:
      logging.info(f'[HOST] Executing CARBonAra on {executor.device().name}')
      res = True
      for f in os.listdir(self.input_dir):
        if f.endswith('.pdb'):
          res = executor.execute(self.__get_command_for_backbone(f))
          if not res:
            break
      executor.execute(f'/usr/bin/chmod 777 --recursive {CARBonAraContainer.CONTAINER_OUTPUT_DIR}')
      return res

  def __get_command_for_backbone(self, backbone_basename: str) -> str:
    return (f'python /CARBonAra/carbonara.py '
            f'--num_sequences {self.num_sequences} '
            f'--imprint_ratio {self.imprint_ratio} '
            f'--sampling_method {self.sampling_method} '
            f'{self.__known_chains_str} '
            f'{self.__known_positions_str} '
            f'{self.__unknown_positions_str} '
            f'{self.__ignored_amino_acids_str} '
            f'{f"--ignore_hetatm {self.ignore_het_atm} " if self.ignore_het_atm else ""}'
            f'{f"--ignore_water {self.ignore_water} " if self.ignore_water else ""}'
            f'{os.path.join(self.CONTAINER_INPUT_DIR, backbone_basename)} {self.CONTAINER_OUTPUT_DIR}')
