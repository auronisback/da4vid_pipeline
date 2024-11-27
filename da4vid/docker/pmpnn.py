import os
import shutil
from typing import List, Tuple

from docker import DockerClient

from da4vid.docker.base import BaseContainer
from da4vid.model.proteins import Protein


class ProteinMPNNContainer(BaseContainer):
  """
  Manages the creation, configuration and execution of the
  ProteinMPNN docker container in the pipeline.
  """

  # Used internally by the container
  INPUT_DIR = '/home/ProteinMPNN/run/inputs'
  OUTPUT_DIR = '/home/ProteinMPNN/run/outputs'

  # Internal scripts location
  __PARSE_CHAINS_SCRIPT = '/home/ProteinMPNN/helper_scripts/parse_multiple_chains.py'
  __ASSIGN_CHAINS_SCRIPT = '/home/ProteinMPNN/helper_scripts/assign_fixed_chains.py'
  __MAKE_DICT_SCRIPT = '/home/ProteinMPNN/helper_scripts/make_fixed_positions_dict.py'
  __PMPNN_SCRIPT = '/home/ProteinMPNN/protein_mpnn_run.py'

  # Internal directory to store jsonl assignment and dictionary files
  __JSONL_DIR = '/home/ProteinMPNN/run/jsonl'

  def __init__(self, input_dir: str, output_dir: str, seqs_per_target: int,
               sampling_temp: float = .1, backbone_noise: float = .0, backbones: List[Protein] | None = None):
    super().__init__(
      image='ameg/protein_mpnn:latest',
      entrypoint='/bin/bash',
      with_gpus=True,
      volumes={
        input_dir: ProteinMPNNContainer.INPUT_DIR,
        output_dir: ProteinMPNNContainer.OUTPUT_DIR
      },
      detach=True
    )
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.seqs_per_target = seqs_per_target
    # Setting ProteinMPNN other
    self.sampling_temp = sampling_temp
    self.backbone_noise = backbone_noise
    self.backbones = backbones  # TODO: make PMPNN uses only given backbones
    # Default chains and positions
    self.__fixed_chains = {}

  def add_fixed_chain(self, chain: str, positions: List[int] = None):
    self.__fixed_chains[chain] = positions

  def run(self, client: DockerClient = None):
    self.commands = self.__create_commands()
    return super()._run_container(client)

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
                 f'--input_path={ProteinMPNNContainer.INPUT_DIR} --output_path={parsed_chains_jsonl}')
    # Creating assign chains command
    assigned_chains_jsonl = f'{ProteinMPNNContainer.__JSONL_DIR}/assigned_chains.jsonl'
    assign_cmd = (f'python {ProteinMPNNContainer.__ASSIGN_CHAINS_SCRIPT} '
                  f'--input_path={parsed_chains_jsonl} --output_path={assigned_chains_jsonl} '
                  f'--chain_list "{chains}"')
    # Creating fixed dict command
    fixed_dict_jsonl = f'{ProteinMPNNContainer.__JSONL_DIR}/fixed_dict.jsonl'
    make_fixed_dict_cmd = (f'python {ProteinMPNNContainer.__MAKE_DICT_SCRIPT} '
                           f'--input_path={parsed_chains_jsonl} --output_path={fixed_dict_jsonl} '
                           f'--chain_list "{chains}" --position_list \'{positions}\'')
    # Creating Protein_MPNN running command (finally!)
    protein_mpnn_cmd = (f'python {ProteinMPNNContainer.__PMPNN_SCRIPT} '
                        f'--jsonl_path {parsed_chains_jsonl} '
                        f'--chain_id_jsonl {assigned_chains_jsonl} '
                        f'--fixed_positions_jsonl {fixed_dict_jsonl} '
                        f'--out_folder {self.OUTPUT_DIR} '
                        f'--num_seq_per_target {self.seqs_per_target} '
                        f'--sampling_temp {self.sampling_temp} '
                        f'--backbone_noise {self.backbone_noise}')
    # Returning the commands
    return [create_cmd, parse_cmd, assign_cmd, make_fixed_dict_cmd, protein_mpnn_cmd,
            f'/usr/bin/chmod 0777 --recursive {self.OUTPUT_DIR}']
