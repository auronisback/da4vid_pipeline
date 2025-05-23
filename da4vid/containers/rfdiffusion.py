import logging
import os
import shutil

from typing_extensions import Self

from da4vid.containers.base import BaseContainer
from da4vid.containers.executor import ContainerExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.model.proteins import Protein, Chain, Epitope


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
    self.provide_seq.append((start - 1, end - 1))  # Provide seq starts from 0
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

  @staticmethod
  def partial_diffusion_around_epitope(protein: Protein, epitope: Epitope):
    """
    Creates the contig map for a partial diffusion around a given epitope of
    the given protein.
    :param protein: The protein on which extract contigs
    :param epitope: The epitope which will be fixed
    :return: The RFdiffusionContigMap for the partial diffusion around the epitope
    :raise ValueError: If the given epitope chain is not present in the protein
    """
    offset = 0
    for chain in protein.chains:
      if chain.name == epitope.chain:
        start = offset + epitope.start
        end = offset + epitope.end
        return RFdiffusionContigMap(protein).full_diffusion().add_provide_seq(start, end)
      offset += len(chain.residues)
    raise ValueError(f'Epitope chain not found in protein: {epitope.chain}')


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
  DEFAULT_IMAGE = 'da4vid/rfdiffusion:latest'

  # Container local folders
  SCRIPT_LOCATION = '/app/RFdiffusion/scripts/run_inference.py'
  MODELS_FOLDER = '/app/RFdiffusion/models'
  INPUT_DIR = '/app/RFdiffusion/inputs'
  OUTPUT_DIR = '/app/RFdiffusion/outputs'
  SCHEDULE_DIR = '/app/RFdiffusion/outputs/schedule'

  def __init__(self, builder: ContainerExecutorBuilder, gpu_manager: CudaDeviceManager,
               model_dir: str, output_dir: str, input_pdb: str, contig_map: RFdiffusionContigMap,
               input_dir: str = None, num_designs: int = 3, diffuser_T: int = 50, partial_T: int = 20,
               potentials: RFdiffusionPotentials = None, out_logfile: str = None, err_logfile: str = None):
    """
    Creates a new instance of this container, without starting it.
    :param builder: The container executor builder used to build the execution container
    :param gpu_manager: The CUDA device manager object to obtain GPUS
    :param model_dir: Directory where RFdiffusion model weights are stored
    :param output_dir: The output folder diffusions
    :param input_pdb: The PDB input to diffuse
    :param contig_map: The map with the contigs
    :param input_dir: The input directory used by the container
    :param num_designs: The number of output diffusions
    :param diffuser_T: Timesteps used for the diffusion. It should be > 15
    :param partial_T: Timesteps used if partial diffusion is specified in contig map
    :param potentials: The potential object used to bias diffusion
    :param out_logfile: The path to which container STDOUT will be written. If None, host STDOUT is used
    :param err_logfile: The path to which container STDERR will be written. If None, host STDERR is used
    """
    super().__init__(
      builder=builder,
      gpu_manager=gpu_manager
    )
    self.model_dir = model_dir
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.input_pdb = input_pdb
    self.contig_map = contig_map
    self.num_designs = num_designs
    self.diffuser_T = diffuser_T
    self.partial_T = partial_T
    self.potentials = potentials
    self.out_logfile = out_logfile
    self.err_logfile = err_logfile

  def run(self) -> bool:
    if self.input_dir is None:
      raise ValueError(f'Input folder not specified')
    # Moving input PDB to inputs folder if needed
    if self.input_dir != os.path.dirname(self.input_pdb):
      shutil.copy2(self.input_pdb, self.input_dir)
    # Setting the builder
    self.builder.set_logs(out_log_stream=self.out_logfile, err_log_stream=self.err_logfile)
    self.builder.set_device(self.gpu_manager.next_device()).set_volumes({
      f'{self.input_dir}': self.INPUT_DIR,
      f'{self.model_dir}': self.MODELS_FOLDER,
      f'{self.output_dir}': self.OUTPUT_DIR
    })
    with self.builder.build() as executor:
      logging.info(f'[HOST] executing RFdiffusion on {executor.device().name}')
      res = executor.execute(self.__create_command())
      # Executing chown in any case
      executor.execute(f'/usr/bin/chmod 0777 --recursive {self.OUTPUT_DIR}')
    return res

  def __create_command(self) -> str:
    cmd = f'python {RFdiffusionContainer.SCRIPT_LOCATION}'
    pdb_basename = os.path.basename(self.input_pdb)
    pdb_name = '.'.join(pdb_basename.split('.')[:-1])
    pdb_path = f'{RFdiffusionContainer.INPUT_DIR}/{pdb_basename}'
    output_prefix = f'{RFdiffusionContainer.OUTPUT_DIR}/{pdb_name}'
    args = {
      'inference.cautious': False,  # Overwrites previous diffusion in output folder
      'inference.input_pdb': pdb_path,
      'inference.output_prefix': output_prefix,
      'inference.model_directory_path': RFdiffusionContainer.MODELS_FOLDER,
      'inference.num_designs': self.num_designs,
      'diffuser.T': self.diffuser_T,
      'contigmap.contigs': self.contig_map.contigs_to_string(),
      'inference.schedule_directory_path': self.SCHEDULE_DIR
    }
    if self.contig_map.partial:  # Partial diffusion
      args['contigmap.provide_seq'] = self.contig_map.provide_seq_to_string()
      args['diffuser.partial_T'] = self.partial_T
    if self.potentials is not None:
      args['potentials.guiding_potentials'] = f"'{self.potentials.potentials_to_string()}'"
      args['potentials.guide_scale'] = f'{self.potentials.guide_scale}'
      args['potentials.guide_decay'] = f'"{self.potentials.guide_decay}"'
    cmd = ' '.join([cmd, *[f'{key}={value}' for key, value in args.items()]])
    logging.debug(f'[HOST] Executing command: {cmd}')
    return cmd
