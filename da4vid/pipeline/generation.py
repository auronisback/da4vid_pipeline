import os
import shutil
import sys
from typing import Tuple, List

from docker import DockerClient
from tqdm import tqdm

from da4vid.docker.pmpnn import ProteinMPNNContainer
from da4vid.docker.rfdiffusion import RFdiffusionContainer, RFdiffusionPotentials, RFdiffusionContigMap
from da4vid.filters import cluster_by_ss, filter_by_rog
from da4vid.io import read_pdb_folder, read_protein_mpnn_fasta
from da4vid.io.fasta_io import write_fasta
from da4vid.model.proteins import Protein
from da4vid.model.samples import SampleSet, Sample, Sequence
from da4vid.pipeline.steps import PipelineStep


class GenerationStep(PipelineStep):

  def execute(self, sample_set: SampleSet) -> SampleSet:
    pass


class RFdiffusionStep(PipelineStep):

  def __init__(self, model_dir: str, output_dir: str,
               epitope: Tuple[int, int], num_designs: int, partial_T: int,
               contacts_threshold: float = None, rog_potential: float = None, client: DockerClient = None):
    """
    Creates a generation step of RFdiffusion to be ran in a container.
    :param model_dir: The directory in which RFdiffusion weights are stored
    :param output_dir: The directory in which outputs are stored
    :param epitope: A tuple with epitope starting and ending residues, starting from 1
    :param num_designs: The number of generated backbones
    :param partial_T: Timesteps for partial diffusion
    :param contacts_threshold: The contact threshold for potentials. If none, no
                               contact potential will be used
    :param rog_potential: The maximum RoG to condition generation. If none, no
                          RoG potential will be applied
    :param client: The docker client used to run RFdiffusion container. If none,
                   a new client will be created
    """
    self.model_dir = model_dir
    self.output_dir = output_dir
    self.epitope = epitope
    self.num_designs = num_designs
    self.partial_T = partial_T
    self.client = client
    self.contacts_threshold = contacts_threshold
    self.rog_potential = rog_potential

  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes the RFdiffusion step and returns the list of filenames
    of generated backbones.
    :return: The list with the path to each generated protein
    """
    sample = sample_set.samples()[0]  # TODO: Only the 1st sample is used
    self.__check_params(sample.protein)
    # Creating output folder if it not exists
    os.makedirs(self.output_dir, exist_ok=True)
    # Running the container
    print('Starting RFdiffusion container with parameters:')
    print(f' - protein PDB file: {sample.filepath}')
    print(f' - epitope interval: {self.epitope}')
    print(f' - number of designs: {self.num_designs}')
    container = self.__create_container(sample)
    if not container.run(self.client):
      raise RuntimeError('RFdiffusion step failed')
    return self.__create_sample_set()

  @staticmethod
  def __check_params(protein):
    if not os.path.isfile(protein.filename):
      raise FileNotFoundError(f'input pdb not found: {protein.filename}')

  def __create_container(self, sample: Sample) -> RFdiffusionContainer:
    potentials = None
    if self.contacts_threshold is not None or self.rog_potential is not None:
      potentials = RFdiffusionPotentials(guiding_scale=10).linear_decay()
      if self.contacts_threshold is not None:
        potentials.add_monomer_contacts(self.contacts_threshold)
      if self.rog_potential is not None:
        potentials.add_rog(self.rog_potential)
    input_dir = os.path.dirname(sample.filepath)
    return RFdiffusionContainer(
      model_dir=self.model_dir, input_dir=input_dir, output_dir=self.output_dir,
      input_pdb=sample.filepath, num_designs=self.num_designs, partial_T=self.partial_T,
      contig_map=RFdiffusionContigMap.partial_diffusion_around_epitope(sample.protein, self.epitope),
      potentials=potentials)

  def __create_folders(self):
    os.makedirs(self.output_dir, exist_ok=True)

  def __create_sample_set(self) -> SampleSet:
    new_set = SampleSet()
    new_set.add_samples([Sample(name=b.name, filepath=b.filename, protein=b) for b
                         in read_pdb_folder(self.output_dir)])
    return new_set


class BackboneFilteringStep(PipelineStep):
  def __init__(self, ss_threshold: int, rog_cutoff: float,
               rog_percentage: bool, output_dir: str):
    self.ss_threshold = ss_threshold
    self.rog_cutoff = rog_cutoff
    self.rog_percentage = rog_percentage
    self.output_dir = output_dir

  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes the filtering and returns the filtered protein objects.
    :return: The list of filtered proteins
    """
    print('Filtering generated backbones by SS and RoG')
    clustered_ss = cluster_by_ss([s.protein for s in sample_set.samples()], threshold=self.ss_threshold)
    print(f'Found {sum([len(v) for v in clustered_ss.values()])} proteins with SS number >= {self.ss_threshold}:')
    print('  SS: number ')
    for k in clustered_ss.keys():
      print(f'  {k}: {len(clustered_ss[k])}')
    # Retaining the 10 smallest proteins for each cluster by filtering via RoG (decreasing)
    filtered_proteins = []
    for k in tqdm(clustered_ss.keys(), file=sys.stdout):
      filtered_proteins += filter_by_rog(clustered_ss[k], cutoff=self.rog_cutoff, percentage=self.rog_percentage)
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


class ProteinMPNNStep(PipelineStep):

  def __init__(self, backbones: List[Protein], input_dir: str, chain: str, epitope: Tuple[int, int], output_dir: str,
               seqs_per_target: int, sampling_temp: float, backbone_noise: float, client: DockerClient = None):
    self.backbones = backbones
    self.input_dir = input_dir
    self.chain = chain
    self.epitope = epitope
    self.output_dir = output_dir
    self.seqs_per_target = seqs_per_target
    self.sampling_temp = sampling_temp
    self.backbone_noise = backbone_noise
    self.client = client
    self.container = ProteinMPNNContainer(
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=self.seqs_per_target,
      sampling_temp=self.sampling_temp,
      backbone_noise=self.backbone_noise
    )
    self.container.add_fixed_chain(self.chain, [r for r in range(self.epitope[0], self.epitope[1] + 1)])

  def execute(self, sample_set: SampleSet) -> SampleSet:
    print('Running ProteinMPNN on filtered backbones with parameters:')
    print(f'  - input backbones: {self.input_dir}')
    print(f'  - output folder: {self.output_dir}')
    print(f'  - sequences per structure: {self.seqs_per_target}')
    print(f'  - sampling temperature: {self.sampling_temp}')
    print(f'  - backbone noise: {self.backbone_noise}')
    # Creating output dir if not existing
    os.makedirs(self.output_dir, exist_ok=True)
    # Running the client
    if not self.container.run(self.client):
      raise RuntimeError('ProteinMPNN step failed')
    # Creating the output set of samples
    print('Loading sequences from FASTA')
    sample_set = SampleSet()
    sample_set.add_samples([Sample(name=b.name, filepath=b.filename, protein=b) for b in self.backbones])
    for sample in tqdm(sample_set.samples(), file=sys.stdout):
      fasta_filepath = '.'.join(os.path.basename(sample.filepath).split('.')[:-1]) + '.fa'
      sequences = self.__rename_sequences_and_extract_data(os.path.join(self.output_dir, 'seqs', fasta_filepath))
      # Adding props to original protein
      for protein in sequences:
        protein.add_prop('protein_mpnn', sequences[0].get_prop('protein_mpnn'))
      sample.add_sequences(
        [Sequence(name=s.name, filepath=fasta_filepath, protein=s, sample=sample) for s in sequences])
    return sample_set

  def __rename_sequences_and_extract_data(self, fasta_path: str) -> List[Protein]:
    proteins = read_protein_mpnn_fasta(fasta_path)
    if not proteins:
      return proteins
    # Excluding original protein sequence (useless to predict it back, especially if it was only backbone)
    sample_name = proteins[0].name
    proteins = proteins[1:]
    output_fasta = os.path.join(self.output_dir, f'{sample_name}.fa')
    for protein in proteins:
      protein.filename = output_fasta
    write_fasta(proteins, output_fasta, overwrite=True)
    return proteins
