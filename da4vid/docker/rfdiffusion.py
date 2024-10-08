import os
from typing_extensions import Self

from docker.client import DockerClient

from da4vid.docker.base import BaseContainer
from da4vid.model import Protein, Chain


class RFdiffusionContigMap:
  """
  Utility class for defining contigs.
  """

  class __RandomLengthContig:
    def __init__(self, min_length: int, max_length: int):
      self.min_length = min_length
      self.max_length = max_length

  class __FixedSequenceContig:
    def __init__(self, chain: Chain, start: int, end: int):
      self.chain = chain
      self.start = start
      self.end = end

  class __ChainBreakContig:
    pass

  def __init__(self, protein: Protein = None, partial: bool = False):
    """
    Creates a new contigs map for the given protein.
    :param protein: The protein to which add contigs
    :param partial: The flag in order to check whether the diffusion will be partial or not
    """
    self.protein = protein
    self.contigs = []
    self.provide_seq = []
    self.partial = partial

  def add_random_length_sequence(self, min_length: int, max_length: int = -1) -> Self:
    """
    Adds a random contig length to the contigs
    :param min_length: Minimum length of residues to create
    :param max_length: Maximum length of residues to create, defaults to min_length
    :return: This instance
    """
    max_length = max_length if max_length != -1 else min_length
    if min_length > max_length:
      raise ValueError(f'min_length greater than max_length: {min_length} > {max_length}')
    self.contigs.append(self.__RandomLengthContig(min_length, max_length))
    return self

  def add_fixed_sequence(self, chain_name: str, start: int, end: int) -> Self:
    """
    Add a contig for fixed sequence in the protein.
    :param chain_name: Name of the chain
    :param start: Starting residue in given chain, zero-indexed
    :param end: Ending residue in given chain, zero-indexed
    :return: This instance
    :raise: AttributeError if the protein has not been set, as no
            chain on which choose fixed elements is present
    """
    if self.protein is None:
      raise AttributeError('protein on which read the fixed chain is not present')
    if start > end:
      raise ValueError(f'starting residue is greater than ending residue: {start} > {end}')
    chain = self.protein.get_chain(chain_name)
    n_resi = len(chain.residues)
    if start >= n_resi or end >= n_resi:
      raise ValueError(f'starting or ending greater than number of residues in the chain: {start}, {end} vs {n_resi}')
    self.contigs.append(self.__FixedSequenceContig(chain, start, end))
    return self

  def add_chain_break(self) -> Self:
    """
    Adds a chain break to contigs.
    :return: This instance
    """
    self.contigs.append(self.__ChainBreakContig())
    return self

  def add_provide_seq(self, start: int, end: int) -> Self:
    if start > end:
      raise ValueError(f'starting residue is greater than ending residue: {start} > {end}')
    self.partial = True  # Provide seq implies partial diffusion
    n_resi = self.protein.length()
    if start >= n_resi or end >= n_resi:
      raise ValueError(f'starting or ending greater than number of residues in the chain: {start}, {end} vs {n_resi}')
    self.provide_seq.append((start, end))
    return self

  def full_diffusion(self) -> Self:
    """
    Adds a contig to fully diffuse the entire protein.
    :return: This instance
    :raise: AttributeError if the protein has not been set
    """
    if self.protein is None:
      raise AttributeError('protein not set')
    n_resi = self.protein.length()
    self.contigs.append(self.__RandomLengthContig(n_resi, n_resi))
    return self

  def contigs_to_string(self) -> str:
    if len(self.contigs) == 0:
      return ''
    contigs = '['
    for contig in self.contigs:
      if isinstance(contig, self.__RandomLengthContig):
        contigs += f'{contig.min_length}-{contig.max_length}/'
      elif isinstance(contig, self.__FixedSequenceContig):
        contigs += f'{contig.chain.name}{contig.start}-{contig.end}/'
      else:  # __ChainBreakContig
        contigs += '0 '
    contigs = contigs[:-1] if contigs[-1] == '/' else contigs  # Removing last / if any
    return contigs + ']'

  def provide_seq_to_string(self) -> str:
    if len(self.provide_seq) == 0:
      return ''
    provide_seq = f'[{self.provide_seq[0][0]}-{self.provide_seq[0][1]}'
    for start, end in self.provide_seq[1:]:
      provide_seq += f',{start}-{end}'
    return provide_seq + ']'


class RFdiffusionPotentials:
  """
  Class defining the potentials used to guide the diffusion.
  """
  def __init__(self, guiding_scale: float = 1):
    """
    Creates a new guiding potential configuration with linear decay.
    :param guiding_scale: The guiding scale for potentials
    """
    self.guide_scale = guiding_scale
    self.guide_decay = 'linear'
    self.potentials = []

  def constant_decay(self) -> Self:
    self.guide_decay = 'constant'
    return self

  def linear_decay(self) -> Self:
    self.guide_decay = 'linear'
    return self

  def quadratic_decay(self) -> Self:
    self.guide_decay = 'quadratic'
    return self

  def cubic_decay(self) -> Self:
    self.guide_decay = 'cubic'
    return self

  def add_monomer_contacts(self, r_0: float, weight: float = 1) -> Self:
    """
    Add a monomer contacts potential to the diffusion.
    :param r_0: The minimum distance to tell whether there is a contact
                between two atoms or not
    :param weight: Potential's weight
    :return: This instance
    :raise: ValueError if r_0 or weight are negative
    """
    if r_0 <= 0:
      raise ValueError(f'r_0 cannot be negative: {r_0}')
    if weight <= 0:
      raise ValueError(f'weight cannot be negative')
    self.potentials.append({'type': 'monomer_contacts', 'r_0': r_0, 'weight': weight})
    return self

  def add_rog(self, min_dist: float, weight: float = 1) -> Self:
    """
    Adds a ROG potential on monomers.
    :param min_dist: The radius restraint
    :param weight: Potential's weight
    :return: This instance
    """
    if min_dist <= 0:
      raise ValueError(f'min_dist cannot be negative: {min_dist}')
    if weight <= 0:
      raise ValueError(f'weight cannot be negative')
    self.potentials.append({'type': 'monomer_ROG', 'min_dist': min_dist, 'weight': weight})
    return self

  def potentials_to_string(self):
    if len(self.potentials) == 0:
      return ''
    s = '['
    for potential in self.potentials:
      s += '"'
      for k, v in potential.items():
        s += f'{k}:{v},'
      s = s[:-1] + '",'
    return s[:-1] + ']'


class RFdiffusionContainer(BaseContainer):

  # Container local folders
  SCRIPT_LOCATION = '/app/RFdiffusion/scripts/run_inference.py'
  MODELS_FOLDER = '/app/RFdiffusion/models'
  INPUT_DIR = '/app/RFdiffusion/inputs'
  OUTPUT_DIR = '/app/RFdiffusion/outputs'

  def __init__(self, model_dir, input_dir, output_dir, num_designs: int = 3):
    super().__init__(
      image='ameg/rfdiffusion:latest',
      entrypoint='/bin/bash',
      with_gpus=True,
      volumes={
        model_dir: RFdiffusionContainer.MODELS_FOLDER,
        input_dir: RFdiffusionContainer.INPUT_DIR,
        output_dir: RFdiffusionContainer.OUTPUT_DIR
      },
      detach=True
    )
    self.model_dir = model_dir
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.num_designs = num_designs

  def run(self, input_pdb, contig_map: RFdiffusionContigMap, partial_T: int = 20,
          potentials: RFdiffusionPotentials = None,
          client: DockerClient = None) -> bool:
    self.commands = [self.__create_command(input_pdb, contig_map, potentials, partial_T)]
    return super()._run_container(client)

  def __create_command(self, input_pdb, contig_map: RFdiffusionContigMap,
                       potentials: RFdiffusionPotentials, partial_T: int) -> str:
    cmd = f'python {RFdiffusionContainer.SCRIPT_LOCATION}'
    pdb_name = os.path.basename(input_pdb).split('.')[0]
    pdb_path = f'{RFdiffusionContainer.INPUT_DIR}/{pdb_name}.pdb'
    output_prefix = f'{RFdiffusionContainer.OUTPUT_DIR}/{pdb_name}/{pdb_name}'
    args = {
      'inference.cautious': False,  # Overwrites previous diffusion in output folder
      'inference.input_pdb': pdb_path,
      'inference.output_prefix': output_prefix,
      'inference.model_directory_path': RFdiffusionContainer.MODELS_FOLDER,
      'inference.num_designs': self.num_designs,
      'contigmap.contigs': contig_map.contigs_to_string()
    }
    if contig_map.partial:  # Partial diffusion
      args['contigmap.provide_seq'] = contig_map.provide_seq_to_string()
      args['diffuser.partial_T'] = partial_T
    if potentials is not None:
      args['potentials.guiding_potentials'] = f"'{potentials.potentials_to_string()}'"
      args['potentials.guide_scale'] = potentials.guide_scale
      args['potentials.guide_decay'] = f'"{potentials.guide_decay}"'
    return ' '.join([cmd, *[f'{key}={value}' for key, value in args.items()]])
