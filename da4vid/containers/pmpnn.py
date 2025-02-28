import logging
import os
from typing import List

from da4vid.containers.base import BaseContainer
from da4vid.containers.executor import ContainerExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.model.proteins import Protein


class ProteinMPNNContainer(BaseContainer):
  """
  Manages the creation, configuration and execution of the
  ProteinMPNN docker container in the pipeline.
  """

  DEFAULT_IMAGE = 'da4vid/protein-mpnn:latest'

  # Used internally by the container
  CONTAINER_INPUT_DIR = '/home/ProteinMPNN/run/inputs'
  CONTAINER_OUTPUT_DIR = '/home/ProteinMPNN/run/outputs'

  # Internal scripts location
  __PARSE_CHAINS_SCRIPT = '/home/ProteinMPNN/helper_scripts/parse_multiple_chains.py'
  __ASSIGN_CHAINS_SCRIPT = '/home/ProteinMPNN/helper_scripts/assign_fixed_chains.py'
  __MAKE_DICT_SCRIPT = '/home/ProteinMPNN/helper_scripts/make_fixed_positions_dict.py'
  __PMPNN_SCRIPT = '/home/ProteinMPNN/protein_mpnn_run.py'

  # Internal directory to store jsonl assignment and dictionary files
  __JSONL_DIR = os.path.join(CONTAINER_OUTPUT_DIR, 'jsonl')

  def __init__(self, builder: ContainerExecutorBuilder, gpu_manager: CudaDeviceManager,
               input_dir: str, output_dir: str, seqs_per_target: int, batch_size: int = 1, sampling_temp: float = .1,
               backbone_noise: float = .0, use_soluble_model: bool = False, backbones: List[Protein] | None = None,
               out_logfile: str = None, err_logfile: str = None):
    super().__init__(builder=builder, gpu_manager=gpu_manager)
    # Checking data
    if seqs_per_target < 1:
      raise ValueError(f'Invalid seqs_per_target: {seqs_per_target}')
    if seqs_per_target % batch_size != 0:
      raise ValueError(f'Batch size ({batch_size}) should be a proper divisor of seqs_per_target ({seqs_per_target})')
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.seqs_per_target = seqs_per_target
    # Setting ProteinMPNN other
    self.sampling_temp = sampling_temp
    self.backbone_noise = backbone_noise
    self.soluble_model = use_soluble_model
    self.backbones = backbones
    self.batch_size = batch_size
    # Checking valid batch size
    self.out_logfile = out_logfile
    self.err_logfile = err_logfile
    # Default chains and positions
    self.__fixed_chains = {}

  def add_fixed_chain(self, chain: str, positions: List[int] = None):
    self.__fixed_chains[chain] = positions

  def run(self) -> bool:
    # Setting builder parameters
    self.builder.set_logs(self.out_logfile, self.err_logfile).set_volumes({
      self.input_dir: self.CONTAINER_INPUT_DIR,
      self.output_dir: self.CONTAINER_OUTPUT_DIR
    }).set_device(self.gpu_manager.next_device())
    commands = self.__create_commands()
    with self.builder.build() as executor:
      logging.info(f'[HOST] Executing ProteinMPNN on {executor.device().name}')
      res = True
      for cmd in commands:
        if not executor.execute(cmd):
          res = False
          break
      # Setting permissions on output directory
      executor.execute(f'/usr/bin/chmod 0777 --recursive {self.CONTAINER_OUTPUT_DIR}')
    return res

  def __create_commands(self) -> List[str]:
    parsed_chains_jsonl = f'{ProteinMPNNContainer.__JSONL_DIR}/parsed_chains.jsonl'
    # Constructing chains and positions strings
    chains = " ".join(self.__fixed_chains.keys())
    positions = ''
    for chain in self.__fixed_chains.keys():
      positions += ' '.join([str(p) for p in self.__fixed_chains[chain]]) + ', '
    positions = positions[:-2] if self.__fixed_chains else positions
    # Creating jsonl folder
    create_cmd = f'mkdir -p {ProteinMPNNContainer.__JSONL_DIR}'
    # Creating parse sequence command
    parse_cmd = (f'python {ProteinMPNNContainer.__PARSE_CHAINS_SCRIPT} '
                 f'--input_path={ProteinMPNNContainer.CONTAINER_INPUT_DIR} --output_path={parsed_chains_jsonl}')
    # Creating assign chains command
    assigned_chains_jsonl = f'{ProteinMPNNContainer.__JSONL_DIR}/assigned_chains.jsonl'
    assign_cmd = (f'python {ProteinMPNNContainer.__ASSIGN_CHAINS_SCRIPT} '
                  f'--input_path={parsed_chains_jsonl} --output_path={assigned_chains_jsonl} '
                  f'--chain_list="{chains}"')
    # Creating fixed dict command
    fixed_dict_jsonl = f'{ProteinMPNNContainer.__JSONL_DIR}/fixed_dict.jsonl'
    make_fixed_dict_cmd = (f'python {ProteinMPNNContainer.__MAKE_DICT_SCRIPT} '
                           f'--input_path={parsed_chains_jsonl} --output_path={fixed_dict_jsonl} '
                           f'--chain_list="{chains}" --position_list="{positions}"')
    # Creating Protein_MPNN running command (finally!)
    protein_mpnn_cmd = (f'python {ProteinMPNNContainer.__PMPNN_SCRIPT} '
                        f'--jsonl_path {parsed_chains_jsonl} '
                        f'--chain_id_jsonl {assigned_chains_jsonl} '
                        f'--fixed_positions_jsonl {fixed_dict_jsonl} '
                        f'--out_folder {self.CONTAINER_OUTPUT_DIR} '
                        f'--num_seq_per_target {self.seqs_per_target} '
                        f'--sampling_temp {self.sampling_temp} '
                        f'--backbone_noise {self.backbone_noise} '
                        f'{"--use_soluble_model " if self.soluble_model else ""}'
                        f'--batch_size {self.batch_size}')
    # Returning the commands
    return [create_cmd, parse_cmd, assign_cmd, make_fixed_dict_cmd, protein_mpnn_cmd]
