import abc
import os.path
import shutil
import sys
from typing import Any, List, Tuple

import torch
from docker import DockerClient
from tqdm import tqdm

from da4vid.docker.omegafold import OmegaFoldContainer
from da4vid.docker.pmpnn import ProteinMPNNContainer
from da4vid.docker.rfdiffusion import RFdiffusionContainer, RFdiffusionContigMap, RFdiffusionPotentials
from da4vid.filters import cluster_by_ss, filter_by_rog, filter_by_plddt
from da4vid.io import read_from_pdb, read_protein_mpnn_fasta
from da4vid.metrics import evaluate_plddt, rog
from da4vid.model import Protein, Proteins


class PipelineStep(abc.ABC):
  @abc.abstractmethod
  def execute(self) -> Any:
    """
    Abstract method executing the concrete step.
    :return: Anything useful the concrete method wishes to return
    """
    return


class PipelineRun(PipelineStep):
  """
  Class composing one or more pipeline steps.
  """

  def __init__(self, steps: List[PipelineStep] = None):
    if steps is None:
      steps = []
    self.steps = steps

  def execute(self) -> List[Any]:
    """
    Executes all steps in the run.
    :return: A dictionary with the concatenation of all
             outputs of steps inside this run
    """
    outputs = []
    for step in self.steps:
      outputs.append(step.execute())
    return outputs


class RFdiffusionStep(PipelineStep):

  def __init__(self, model_dir: str, protein: Protein, epitope: Tuple[int, int],
               output_dir: str, num_designs: int, partial_T: int,
               contacts_threshold: float = None, rog_potential: float = None, client: DockerClient = None):
    self.protein = protein
    self.output_dir = output_dir
    self.epitope = epitope
    self.num_designs = num_designs
    self.partial_T = partial_T
    input_pdb = protein.filename
    input_dir = os.path.dirname(os.path.abspath(input_pdb))
    self.client = client
    potentials = None
    if contacts_threshold is not None or rog_potential is not None:
      potentials = RFdiffusionPotentials(guiding_scale=10).linear_decay()
      if contacts_threshold is not None:
        potentials.add_monomer_contacts(5)
      if rog_potential is not None:
        potentials.add_rog(12)
    self.container = RFdiffusionContainer(
      model_dir=model_dir, input_dir=input_dir, output_dir=output_dir, input_pdb=input_pdb,
      num_designs=num_designs, partial_T=partial_T,
      contig_map=RFdiffusionContigMap.partial_diffusion_around_epitope(protein, epitope),
      potentials=potentials)

  def execute(self) -> List[str]:
    """
    Executes the RFdiffusion step and returns the list of filenames
    of generated backbones.
    :return: The list with the path to each generated protein
    """
    self.__check_params()
    # Creating output folder if it not exists
    os.makedirs(self.output_dir, exist_ok=True)
    # Running the container
    print('Starting RFdiffusion container with parameters:')
    print(f' - protein PDB file: {self.protein.filename}')
    print(f' - epitope interval: {self.epitope}')
    print(f' - number of designs: {self.num_designs}')
    if not self.container.run(self.client):
      raise RuntimeError('RFdiffusion step failed')
    return [os.path.join(self.output_dir, pdb) for pdb
            in os.listdir(self.output_dir) if pdb.endswith('.pdb')]

  def __check_params(self):
    if not os.path.isfile(self.protein.filename):
      raise FileNotFoundError(f'input pdb not found: {self.protein.filename}')

  def __create_folders(self):
    os.makedirs(self.output_dir, exist_ok=True)


class BackboneFilteringStep(PipelineStep):
  def __init__(self, diffusions: List[str], ss_threshold: int, rog_cutoff: float,
               rog_percentage: bool, move_to: str):
    self.diffused = [read_from_pdb(d, b_fact_prop='epitope') for d in diffusions]
    self.ss_threshold = ss_threshold
    self.rog_cutoff = rog_cutoff
    self.rog_percentage = rog_percentage
    self.move_to = move_to

  def execute(self) -> List[Protein]:
    """
    Executes the filtering and returns the filtered protein objects.
    :return: The list of filtered proteins
    """
    print('Filtering generated backbones by SS and RoG')
    clustered_ss = cluster_by_ss(self.diffused, threshold=self.ss_threshold)
    print(f'Found {sum([len(v) for v in clustered_ss.values()])} proteins with SS number >= {self.ss_threshold}:')
    print('  SS: number ')
    for k in clustered_ss.keys():
      print(f'  {k}: {len(clustered_ss[k])}')
    # Retaining the 10 smallest proteins for each cluster by filtering via RoG (decreasing)
    filtered = []
    for k in tqdm(clustered_ss.keys(), file=sys.stdout):
      filtered += filter_by_rog(clustered_ss[k], cutoff=self.rog_cutoff, percentage=self.rog_percentage)
    print(f'Filtered {len(filtered)} proteins by RoG with '
          f'cutoff {self.rog_cutoff}{"%" if self.rog_percentage else ""}:')
    for p in filtered:
      print(f'  {p.name}: {p.props["rog"].item():.3f} A')
    # Moving filtered PDB files to PMPNN input directory
    os.makedirs(self.move_to, exist_ok=True)
    for protein in filtered:
      filename = os.path.basename(protein.filename)
      new_location = os.path.join(self.move_to, filename)
      shutil.copy2(protein.filename, new_location)
      protein.filename = new_location  # Updating protein location
    return filtered


class SampleSet:
  """
  Utility class to manage ProteinMPNN outputs.
  """

  def __init__(self):
    self.samples = {}

  def add_originals(self, originals: List[Protein]) -> None:
    """
    Adds all original proteins to this set.
    :param originals: The list of original proteins which needs
                      to be added, with empty set of samples
    """
    self.samples |= {protein.name: {'original': protein, 'samples': {}} for protein in originals}

  def add_original(self, original_protein: Protein,
                   samples: List[Protein] = None) -> None:
    """
    Adds a new original protein to the sample set
    :param original_protein: The protein which has to be added
    :param samples: The samples connected to the original protein. If
                    not given, no samples will be connected to the
                    original protein
    """
    self.samples[original_protein.name] = {
      'original': original_protein,
      'samples': {s.name: s for s in samples} if samples else {}
    }

  def add_sample_for_protein(self, original_protein: Protein | str,
                             samples: Protein | List[Protein]) -> None:
    """
    Adds one or more samples for an original protein.
    :param original_protein: The original protein object or its name
    :param samples: A single sample or a list of samples which need to be added
    """
    name = original_protein.name if isinstance(original_protein, Protein) else original_protein
    samples = [samples] if isinstance(samples, Protein) else samples
    if name not in self.samples.keys():
      # Adding the new original protein
      if not isinstance(original_protein, Protein):
        raise ValueError(f'unable to add sample: original protein unknown {name}')
      self.samples[name] = {
        'original': original_protein,
        'samples': {s.name: s for s in samples}}
    else:
      self.samples[name]['samples'] |= {s.name: s for s in samples}

  def get_originals(self) -> List[Protein]:
    """
    Gets all original proteins in the set.
    :return: A list with all original proteins
    """
    return [s['original'] for s in self.samples.values()]

  def get_original(self, name: str) -> Protein | None:
    return self.samples[name] if name in self.samples.keys() else None

  def get_samples_for(self, original: str | Protein) -> List[Protein] | None:
    """
    Gets all samples for an original protein.
    :param original: The original protein object or its name
    :return: The list of samples in the protein, or None if the specified original protein
             is not in the set
    """
    name = original if isinstance(original, str) else original.name
    if name in self.samples.keys():
      return list(self.samples[name]['samples'].values())
    return None

  def get_sample_by_name(self, original_name: str, sample_name: str) -> Tuple[Protein, Protein] | None:
    """
    Gets the original protein and its samples specified by their names.
    :param original_name: Name of the original protein
    :param sample_name: Name of the specific sample
    :return: A tuple with the original and sample proteins, or None if
             neither the original nor the sample are present
    """
    if original_name in self.samples.keys():
      orig = self.samples[original_name]['original']
      if sample_name in self.samples[original_name]['samples'].keys():
        return orig, self.samples[original_name]['samples'][sample_name]
    return None

  def samples_list(self) -> List[Protein]:
    return [s for p in self.samples.values() for s in p['samples'].values()]


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

  def execute(self) -> SampleSet:
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
    sample_set.add_originals(self.backbones)
    for protein in tqdm(self.backbones, file=sys.stdout):
      filename = ''.join(os.path.basename(protein.filename).split('.')[:-1]) + '.fa'
      sampled = read_protein_mpnn_fasta(os.path.join(self.output_dir, 'seqs', filename))
      # Adding props to original protein
      protein.props['protein_mpnn'] = sampled[0].props['protein_mpnn']
      sample_set.add_sample_for_protein(protein, sampled[1:])
    return sample_set


class OmegaFoldStep(PipelineStep):
  CONTAINER_TMP_OUTPUT = 'tmp'

  def __init__(self, sample_set: SampleSet, model_dir: str, input_dir: str, output_dir: str,
               num_recycles: int, model_weights: str, device: str, client: DockerClient = None):
    self.sample_set = sample_set
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.num_recycles = num_recycles
    self.model_weights = model_weights
    self.device = device
    self.client = client
    self.container = OmegaFoldContainer(
      model_dir=model_dir,
      input_dir=self.input_dir,
      output_dir=os.path.join(self.output_dir, OmegaFoldStep.CONTAINER_TMP_OUTPUT),
      num_recycles=self.num_recycles,
      model_weights=self.model_weights,
      device=self.device
    )

  def execute(self) -> SampleSet:
    """
    Executes OmegaFold on sequences to refold them.
    :return: A sample set with structure information obtained by OmegaFold prediction
    """
    # Creating output directories
    tmp_outputs = os.path.join(self.output_dir, OmegaFoldStep.CONTAINER_TMP_OUTPUT)
    os.makedirs(self.output_dir, exist_ok=True)
    os.makedirs(tmp_outputs, exist_ok=True)
    # Starting OmegaFold
    print('Running OmegaFold for structure prediction')
    print(f' - input folder: {self.input_dir}')
    print(f' - output folder: {self.output_dir}')
    print(f' - model weights: {self.model_weights}')
    print(f' - num_recycles: {self.num_recycles}')
    print(f' - running on: {self.device}')
    if not self.container.run(self.client):
      raise ValueError('OmegaFold step failed')
    # Renaming OmegaFold outputs and collecting paths
    paths = self.__perform_rename(tmp_outputs)
    # Merging data
    return self.__merge_data(paths)

  def __perform_rename(self, tmp_outputs: str) -> List[str]:
    paths = []
    for d in os.listdir(tmp_outputs):
      full_d = os.path.join(tmp_outputs, d)
      if os.path.isdir(full_d):
        orig_name = os.path.basename(full_d)
        os.makedirs(os.path.join(self.output_dir, orig_name), exist_ok=True)
        for f in os.listdir(full_d):
          if f.endswith('.pdb') and f.split(',')[0].strip() != orig_name:
            src_name = os.path.join(full_d, f)
            sample_num = f.split(',')[1].split('=')[1].strip()
            dest_name = os.path.join(self.output_dir, orig_name, f'{orig_name}_{sample_num}.pdb')
            shutil.copy2(src_name, dest_name)
            paths.append(dest_name)
    return paths

  def __merge_data(self, paths: List[str]) -> SampleSet:
    print('Retrieving Omegafold predictions')
    for f in tqdm(paths, file=sys.stdout):
      if f.endswith('.pdb'):
        folded = read_from_pdb(f, b_fact_prop='plddt')
        bn = os.path.basename(f)
        orig_name = '_'.join(bn.split('_')[:-1])
        sample_name = bn[:-4]
        original_protein, sample = self.sample_set.get_sample_by_name(orig_name, sample_name)
        Proteins.merge_sequence_with_structure(sample, folded)
    return self.sample_set


class SequenceFilteringStep(PipelineStep):
  def __init__(self, sample_set: SampleSet, plddt_threshold: float, rog_cutoff: float = None, device: str = 'cpu'):
    """
    Initializes the sequence filtering step, evaluating the average pLDDT
    of folding predictions for all samples in each original backbone protein
    and filtering those proteins under a certain threshold. Filtered proteins will be
    also cut off according to their RoG, in ascending order.
    :param sample_set: The set of sampled proteins from original backbones
    :param plddt_threshold: The threshold on samples average pLDDT values
    :param rog_cutoff: The cutoff for RoG filtering. If None, no cutoff will be applied
    :param device: The device on which execute calculations
    """
    self.sample_set = sample_set
    self.plddt_threshold = plddt_threshold
    self.rog_cutoff = rog_cutoff
    self.device = device

  def execute(self) -> SampleSet:
    """
    Executes the filtering step.
    :return: The SampleSet object with filtered samples for each original backbone
    """
    filtered_set = SampleSet()
    print(f'Filtering samples with mean pLDDT >= {self.plddt_threshold}')
    for original in tqdm(self.sample_set.get_originals(), file=sys.stdout):
      samples = self.sample_set.get_samples_for(original)
      mean_plddt = torch.mean(evaluate_plddt(samples, device=self.device))
      # Adding the original backbone and the samples beyond threshold to the filtered set
      if mean_plddt >= self.plddt_threshold:
        filtered_samples = [s for s in samples if s.props['plddt'] >= self.plddt_threshold]
        # Sorting and filtering by RoG
        if self.rog_cutoff is not None:
          rog(filtered_samples)  # Evaluating RoG
          filtered_samples.sort(key=lambda s: s.props['rog'])
          print([(p.name, p.props['rog']) for p in filtered_samples])
          filtered_samples = filtered_samples[:self.rog_cutoff]
          print([(p.name, p.props['rog']) for p in filtered_samples])
        filtered_set.add_original(original, filtered_samples)
    return filtered_set
