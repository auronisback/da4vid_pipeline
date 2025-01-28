import os
import shutil
import sys
from typing import List

from tqdm import tqdm

from da4vid.docker.pmpnn import ProteinMPNNContainer
from da4vid.docker.rfdiffusion import RFdiffusionContainer, RFdiffusionPotentials, RFdiffusionContigMap
from da4vid.filters import cluster_by_ss, filter_by_rog
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io import read_pdb_folder, read_protein_mpnn_fasta
from da4vid.io.fasta_io import write_fasta
from da4vid.model.proteins import Protein, Epitope
from da4vid.model.samples import SampleSet, Sample, Sequence
from da4vid.pipeline.steps import PipelineStep, DockerStep, PipelineException


class RFdiffusionStep(DockerStep):
  """
  Abstracts a step of RFdiffusion in the pipeline.
  """

  class RFdiffusionConfig:
    """
    Class encapsulating the configuration for RFdiffusion.
    """

    def __init__(self, num_designs: int, partial_T: int = 23,
                 contacts_threshold: float = None, rog_potential: float = None):
      """
      Creates a new RFdiffusion configuration.
      :param num_designs: The number of generated backbones
      :param partial_T: Timesteps for partial diffusion. Defaults to 23
      :param contacts_threshold: The contact threshold for potentials. If none, no
                                 contact potential will be used
      :param rog_potential: The maximum RoG to condition generation. If none, no
                            RoG potential will be applied
      """
      self.num_designs = num_designs
      self.partial_T = partial_T
      self.contacts_threshold = contacts_threshold
      self.rog_potential = rog_potential

    def __str__(self):
      return (f'rfdiffusion:\n'
              f'- num_designs: {self.num_designs}\n - partial_T: {self.partial_T}\n '
              f'- contacts_threshold: {self.contacts_threshold}\n - rog_potential: {self.rog_potential}\n')

  def __init__(self, epitope: Epitope, model_dir: str,
               gpu_manager: CudaDeviceManager, config: RFdiffusionConfig, **kwargs):
    """
    Creates a generation step of RFdiffusion running in a container.
    :param epitope: The epitope around which diffuse the scaffold
    :param model_dir: The directory in which RFdiffusion weights are stored
    :param client: The docker client used to run RFdiffusion container
    :param gpu_manager: The CUDA device manager used to assign GPUs
    :param kwargs: Other parameters used to create the step, such as name and folder
    """
    super().__init__(**kwargs)
    self.epitope = epitope
    self.model_dir = model_dir
    self.gpu_manger = gpu_manager
    self.config = config
    self.input_dir = os.path.join(self.get_context_folder(), 'inputs')
    self.output_dir = os.path.join(self.get_context_folder(), 'outputs')

  def _execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes the RFdiffusion step and returns the list of filenames
    of generated backbones.
    :return: The list with the path to each generated protein
    """
    sample = sample_set.samples()[0]  # TODO: Only the 1st sample is used
    self.__check_params(sample.protein)
    # Creating input and output folders if it not exist
    self.__create_folders()
    # Running the container
    print('Starting RFdiffusion container with parameters:')
    print(f' - protein PDB file: {sample.filepath}')
    print(f' - epitope: {self.epitope}')
    print(f' - number of designs: {self.config.num_designs}')
    print(f' - partial T: {self.config.partial_T}')
    print(f' - contact threshold: {self.config.contacts_threshold}')
    print(f' - RoG potential: {self.config.rog_potential}')
    container = self.__create_container(sample)
    if not container.run():
      raise PipelineException('RFdiffusion step failed')
    return self.__create_sample_set()

  def _resume(self, sample_set: SampleSet) -> SampleSet:
    # Just creating the sample set from the outputs
    return self.__create_sample_set()

  @staticmethod
  def __check_params(protein):
    if not os.path.isfile(protein.filename):
      raise FileNotFoundError(f'input pdb not found: {protein.filename}')

  def __create_container(self, sample: Sample) -> RFdiffusionContainer:
    """
    Creates the RFdiffusion container given the sample backbone.
    :param sample: The sample used in the step
    :return: The initialized RFdiffusion container
    """
    potentials = None
    if self.config.contacts_threshold is not None or self.config.rog_potential is not None:
      potentials = RFdiffusionPotentials(guiding_scale=10).linear_decay()
      if self.config.contacts_threshold is not None:
        potentials.add_monomer_contacts(self.config.contacts_threshold)
      if self.config.rog_potential is not None:
        potentials.add_rog(self.config.rog_potential)
    shutil.copy2(sample.filepath, self.input_dir)
    return RFdiffusionContainer(
      model_dir=self.model_dir, input_dir=self.input_dir, output_dir=self.output_dir,
      input_pdb=sample.filepath, num_designs=self.config.num_designs, partial_T=self.config.partial_T,
      contig_map=RFdiffusionContigMap.partial_diffusion_around_epitope(sample.protein, self.epitope),
      potentials=potentials, client=self.client, gpu_manager=self.gpu_manger,
      out_logfile=self.out_logfile, err_logfile=self.err_logfile
    )

  def __create_folders(self) -> None:
    """
    Creates folders needed in the step.
    """
    os.makedirs(self.input_dir, exist_ok=True)
    os.makedirs(self.output_dir, exist_ok=True)

  def __create_sample_set(self) -> SampleSet:
    """
    Creates a new sample set with the outputs.
    :return: The sample set obtained after the RFdiffusion step
    """
    new_set = SampleSet()
    new_set.add_samples([Sample(name=b.name, filepath=b.filename, protein=b) for b
                         in read_pdb_folder(self.output_dir)])
    return new_set

  def input_folder(self) -> str:
    return self.input_dir

  def output_folder(self) -> str:
    return self.output_dir


class BackboneFilteringStep(PipelineStep):
  """
  Abstract the step of filtering backbones according to their structural data.
  """

  def __init__(self, ss_threshold: int, rog_cutoff: float, rog_percentage: bool = False,
               gpu_manager: CudaDeviceManager = None, **kwargs):
    """
    Creates a new step for filtering backbones in a sample set.
    :param ss_threshold: The threshold for number of Secondary Structures to be
                         inserted in backbone clusters
    :param rog_cutoff: The number of structures retained for each cluster
    :param rog_percentage: Flag determining whether the rog_cutoff is considered
                           absolute or in percentage of structures in each cluster
    :param gpu_manager: The CUDA device manager used to assign GPUs. If none, CPU
                        will be used instead
    :param kwargs: Other parameters used to create the step, such as name and folder
    """
    super().__init__(**kwargs)
    self.output_dir = os.path.join(self.get_context_folder(), 'outputs')
    self.ss_threshold = ss_threshold
    self.rog_cutoff = rog_cutoff
    self.rog_percentage = rog_percentage
    self.device = gpu_manager.next_device().name if gpu_manager else 'cpu'

  def _execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes the filtering and returns the filtered protein objects.
    :return: The sample set with the filtered proteins
    """
    print('Filtering generated backbones by SS and RoG')
    clustered_ss = cluster_by_ss([s.protein for s in sample_set.samples()],
                                 threshold=self.ss_threshold, device=self.device)
    print(f'Found {sum([len(v) for v in clustered_ss.values()])} proteins with SS number >= {self.ss_threshold}:')
    print('  SS: number ')
    for k in clustered_ss.keys():
      print(f'  {k}: {len(clustered_ss[k])}')
    # Retaining the 10 smallest proteins for each cluster by filtering via RoG (decreasing)
    filtered_proteins = []
    for k in tqdm(clustered_ss.keys(), file=sys.stdout):
      filtered_proteins += filter_by_rog(clustered_ss[k], cutoff=self.rog_cutoff,
                                         percentage=self.rog_percentage, device=self.device)
    print(f'Filtered {len(filtered_proteins)} proteins by RoG with '
          f'cutoff {self.rog_cutoff}{"%" if self.rog_percentage else ""}:')
    for p in filtered_proteins:
      print(f'  {p.name}: {p.get_prop("rog").item():.3f} A')
    # Copying the filtered proteins into the output location
    filtered_set = SampleSet()
    filtered_set.add_samples([sample_set.get_sample_by_name(p.name) for p in filtered_proteins])
    os.makedirs(self.output_dir, exist_ok=True)
    for sample in filtered_set.samples():
      filename = os.path.basename(sample.filepath)
      new_location = os.path.join(self.output_dir, filename)
      shutil.copy2(sample.filepath, new_location)
      sample.filepath = sample.protein.filename = new_location  # Updating protein location
    return filtered_set

  def _resume(self, sample_set: SampleSet) -> SampleSet:
    # In this case, we re-execute the filtering (a crucial point
    # if filtering options are changed), while first removing previous filtered samples
    if os.path.isdir(self.output_dir):
      shutil.rmtree(self.output_dir)
    return self._execute(sample_set)

  def input_folder(self) -> str:
    # This step does not use an input directory
    return ''

  def output_folder(self) -> str:
    return self.output_dir


class ProteinMPNNStep(DockerStep):
  """
  Abstracts a step of sequence sampling with ProteinMPNN.
  """

  class ProteinMPNNConfig:
    """
    Class storing configuration of a ProteinMPNN step.
    """

    def __init__(self, seqs_per_target: int, sampling_temp: float = 0.2,
                 backbone_noise: float = 0, batch_size: int = 32, use_soluble_model: bool = False):
      """
      Creates the configuration used in the related ProteinMPNN step.
      :param seqs_per_target: The number of sequences per each backbone
      :param sampling_temp: The temperature used by ProteinMPNN to sample. Defaults to 0.2
      :param backbone_noise: The noise which will be added to backbone atoms for
                             generating sequences. Defaults to 0 (noise not added)
      :param batch_size: The number of sequences in a single GPU batch. NOTE: this
                         value should be a divisor of seqs_per_target value, otherwise
                         the actual number of output sequences is the multiple of
                         batch size closest to seqs_per_target
      :param use_soluble_model: Flag checking whether use soluble model weights. Defaults to False
      :raise ValueError: If batch_size is greater than seqs_per_target, as in this
                         case no sequence will be generated
      """
      if batch_size > seqs_per_target:
        raise ValueError(f'Batch size greater than sequences per target: {batch_size} > {seqs_per_target}')
      self.seqs_per_target = seqs_per_target
      self.sampling_temp = sampling_temp
      self.backbone_noise = backbone_noise
      self.batch_size = batch_size
      self.use_soluble_model = use_soluble_model

    def __str__(self):
      return (f'protein_mpnn:\n'
              f' - seqs_per_target: {self.seqs_per_target}\n'
              f' - sampling_temp: {self.sampling_temp}\n'
              f' - backbone_noise: {self.backbone_noise}\n'
              f' - batch_size: {self.batch_size}\n'
              f' - use_soluble_model: {self.use_soluble_model}\n')

  def __init__(self, epitope: Epitope, gpu_manager: CudaDeviceManager, config: ProteinMPNNConfig, **kwargs):
    """
    Creates a ProteinMPNN step.
    :param epitope: The epitope which should be held fixed
    :param gpu_manager: The CUDA device manager for assigning GPU resources
    :param config: The ProteinMPNN configuration
    :param kwargs: Other arguments for creating the step, such as name and folder
    """
    super().__init__(**kwargs)
    self.input_dir = os.path.join(self.get_context_folder(), 'inputs')
    self.output_dir = os.path.join(self.get_context_folder(), 'outputs')
    self.epitope = epitope
    self.gpu_manager = gpu_manager
    self.config = config
    self.container = ProteinMPNNContainer(
      image=self.image,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=self.config.seqs_per_target,
      sampling_temp=self.config.sampling_temp,
      backbone_noise=self.config.backbone_noise,
      use_soluble_model=self.config.use_soluble_model,
      batch_size=self.config.batch_size,  # TODO: make a check in order to avoid losing sequences
      client=self.client,
      gpu_manager=self.gpu_manager
    )
    self.container.add_fixed_chain(
      self.epitope.chain, [r for r in range(self.epitope.start, self.epitope.end + 1)]
    )

  def _execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes the ProteinMPNN step.
    :param sample_set: The sample set on which create sequences
    :return: A sample set with the generated sequences subdivided by their
             original samples
    """
    print('Running ProteinMPNN on filtered backbones with parameters:')
    print(f' - input backbones: {self.input_dir}')
    print(f' - output folder: {self.output_dir}')
    print(f' - sequences per structure: {self.config.seqs_per_target}')
    print(f' - sampling temperature: {self.config.sampling_temp}')
    print(f' - backbone noise: {self.config.backbone_noise}')
    print(f' - batch size: {self.config.batch_size}')
    # Creating output and input dir if not existing
    os.makedirs(self.input_dir, exist_ok=True)
    os.makedirs(self.output_dir, exist_ok=True)
    # Copying input backbones into input folder
    for sample in sample_set.samples():
      sample_basename = os.path.basename(sample.filepath)
      shutil.copy2(sample.filepath, os.path.join(self.input_dir, sample_basename))
    # Running the client
    if not self.container.run():
      raise PipelineException('ProteinMPNN step failed')
    # Creating the output set of samples
    print('Loading sequences from FASTA')
    for sample in tqdm(sample_set.samples(), file=sys.stdout):
      fasta_filepath = '.'.join(os.path.basename(sample.filepath).split('.')[:-1]) + '.fa'
      proteins = self.__extract_proteins_from_fasta(os.path.join(
        self.__sequences_output_dir(), fasta_filepath))
      write_fasta(proteins, proteins[0].filename, overwrite=True)
      # Adding props to original protein
      for protein in proteins:
        protein.add_prop('protein_mpnn', proteins[0].get_prop('protein_mpnn'))
      sample.add_sequences(
        [Sequence(name=s.name, filepath=fasta_filepath, protein=s, sample=sample) for s in proteins])
    # Removing sequences directory after copy
    shutil.rmtree(self.__sequences_output_dir())
    return sample_set

  def _resume(self, sample_set: SampleSet) -> SampleSet:
    for sample in tqdm(sample_set.samples(), file=sys.stdout):
      fasta_filepath = '.'.join(os.path.basename(sample.filepath).split('.')[:-1]) + '.fa'
      proteins = self.__extract_proteins_from_fasta(os.path.join(
        self.output_dir, fasta_filepath))
      for protein in proteins:
        protein.add_prop('protein_mpnn', proteins[0].get_prop('protein_mpnn'))
        sample.add_sequences([Sequence(name=s.name, filepath=fasta_filepath, protein=s, sample=sample)
                              for s in proteins])
    return sample_set

  def __extract_proteins_from_fasta(self, fasta_path: str) -> List[Protein]:
    """
    Recovers the protein from FASTA sequences produced by this step.
    :param fasta_path: The path on which extract FASTA sequences
    :return: The list of generated proteins
    """
    proteins = read_protein_mpnn_fasta(fasta_path)
    if not proteins:
      return proteins
    # The first original protein will be the name of this sample
    sample_name = proteins[0].name
    # Excluding original protein sequence (useless to predict it back, especially if it was only backbone)
    proteins = proteins[1:]
    output_fasta = os.path.join(self.output_dir, f'{sample_name}.fa')
    for protein in proteins:
      protein.filename = output_fasta
    # Removing original sequences directory
    return proteins

  def __sequences_output_dir(self) -> str:
    """
    Gets the full path to output FASTAs.
    :return: The path to the folder where FASTAs are placed by the container
    """
    return os.path.join(self.output_dir, 'seqs')

  def input_folder(self) -> str:
    return self.input_dir

  def output_folder(self) -> str:
    return self.output_dir
