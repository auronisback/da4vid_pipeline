import os
import sys
from typing import List

import torch
from docker import DockerClient
from tqdm import tqdm

from da4vid.docker.omegafold import OmegaFoldContainer
from da4vid.io import read_from_pdb
from da4vid.metrics import evaluate_plddt, rog
from da4vid.model.proteins import Proteins
from da4vid.model.samples import SampleSet, Fold
from da4vid.pipeline.steps import PipelineStep


class ValidationStep(PipelineStep):

  def execute(self, sample_set: SampleSet) -> SampleSet:
    pass


class OmegaFoldStep(PipelineStep):

  def __init__(self, model_dir: str, input_dir: str, output_dir: str,
               num_recycles: int, model_weights: str, device: str, client: DockerClient = None,
               max_parallel: int = 1):
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.num_recycles = num_recycles
    self.model_weights = model_weights
    self.device = device
    self.client = client
    self.container = OmegaFoldContainer(
      model_dir=model_dir,
      input_dir=self.input_dir,
      output_dir=os.path.join(self.output_dir),
      num_recycles=self.num_recycles,
      model_weights=self.model_weights,
      device=self.device,
      max_parallel=max_parallel
    )

  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes OmegaFold on sequences to refold them.
    :return: A sample set with structure information obtained by OmegaFold prediction
    """
    # Creating output directory
    os.makedirs(self.output_dir, exist_ok=True)
    # Starting OmegaFold
    print('Running OmegaFold for structure prediction')
    print(f' - input folder: {self.input_dir}')
    print(f' - output folder: {self.output_dir}')
    print(f' - model weights: {self.model_weights}')
    print(f' - num_recycles: {self.num_recycles}')
    print(f' - running on: {self.device}')
    if not self.container.run(self.client):
      raise ValueError('OmegaFold step failed')
    # Merging data
    self.__merge_data(sample_set)
    return sample_set

  def __merge_data(self, sample_set: SampleSet) -> SampleSet:
    print('Retrieving Omegafold predictions')
    pdb_files = []
    for d in os.listdir(self.output_dir):
      full_path = os.path.join(self.output_dir, d)
      if os.path.isdir(full_path):
        pdb_files += [os.path.join(full_path, f) for f in os.listdir(full_path) if f.endswith('.pdb')]
    for f in tqdm(pdb_files, file=sys.stdout):
      folded = read_from_pdb(f, b_fact_prop='plddt')
      bn = os.path.basename(f)
      orig_name = '_'.join(bn.split('_')[:-1])
      sequence_name = bn[:-4]
      sample = sample_set.get_sample_by_name(orig_name)
      sequence = sample.get_sequence_by_name(sequence_name)
      plddt = evaluate_plddt(folded, plddt_prop='omegafold.plddt', device=self.device)
      Proteins.merge_sequence_with_structure(sequence.protein, folded)
      sequence.add_folds(
        Fold(
          sequence=sequence,
          filepath=folded.filename,
          model='omegafold',
          protein=folded,
          metrics={'plddt': plddt}
        )
      )
    return sample_set


class SequenceFilteringStep(PipelineStep):
  def __init__(self, model: str, plddt_threshold: float, avg_cutoff: float, rog_cutoff: float = None,
               max_samples: int = None, device: str = 'cpu'):
    """
    Initializes the sequence filtering step, evaluating the average pLDDT
    of folding predictions for all samples in each original backbone protein
    and filtering those proteins under a certain threshold. Filtered proteins will be
    also cut off according to their RoG, in ascending order.
    :param model: The model used to retrieve pLDDT predictions
    :param plddt_threshold: The threshold on samples average pLDDT values
    :param avg_cutoff: The number of samples used to evaluate average pLDDT
    :param rog_cutoff: The cutoff for RoG filtering. If None, no cutoff will be applied
    :param max_samples: The maximum number of samples to retain for each original protein. If None, no
                        limit to the number of samples will be used. Defaults to None
    :param device: The device on which execute calculations
    """
    self.model = model
    self.plddt_threshold = plddt_threshold
    self.avg_cutoff = avg_cutoff
    self.rog_cutoff = rog_cutoff
    self.max_samples = max_samples
    self.device = device

  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes the filtering step.
    :return: The SampleSet object with filtered samples for each original backbone
    """
    filtered_set = SampleSet()
    print(f'Filtering samples with mean pLDDT >= {self.plddt_threshold}')
    for sample in tqdm(sample_set.samples(), file=sys.stdout):
      folds = sample.get_folds_for_model(self.model)
      filtered_folds = self.__filter_by_plddt(folds)
      # Sorting and filtering by RoG
      if filtered_folds and self.rog_cutoff is not None:
        filtered_folds = self.__filter_by_rog(filtered_folds)
      if filtered_folds:
        filtered_set.add_samples(filtered_folds)
    print(f'Filtered {len(sample_set.samples())} backbones for '
          f'a total of {len(filtered_set.samples())} samples')
    return filtered_set

  def __filter_by_plddt(self, folds: List[Fold]) -> List[Fold]:
    if not folds:
      return []
    plddts = evaluate_plddt([f.protein for f in folds], f'{self.model}.plddt')
    mean_plddt = torch.mean(plddts)
    if mean_plddt >= self.plddt_threshold:
      # Adding pLDDT metric to fold
      for plddt, fold in zip(plddts, folds):
        fold.metrics.add_metric('plddt', plddt)
      folds.sort(key=lambda f: f.metrics.get_metric('plddt'), reverse=True)
      return folds[:self.avg_cutoff]
    return []

  def __filter_by_rog(self, folds: List[Fold]) -> List[Fold]:
    rog([fold.protein for fold in folds])  # Evaluating RoG
    for fold in folds:
      fold.metrics.add_metric('rog', fold.protein.get_prop('rog'))
    folds.sort(key=lambda f: f.metrics.get_metric('rog'))
    folds = folds[:self.rog_cutoff]
    return folds
