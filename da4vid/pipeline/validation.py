import os
import shutil
import sys
from typing import List

import docker.client
import torch
from docker import DockerClient
from tqdm import tqdm

from da4vid.docker.colabfold import ColabFoldContainer
from da4vid.docker.omegafold import OmegaFoldContainer
from da4vid.io import read_from_pdb
from da4vid.io.fasta_io import write_fasta
from da4vid.metrics import evaluate_plddt, rog
from da4vid.model.proteins import Proteins
from da4vid.model.samples import SampleSet, Fold, Sample
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


class ColabFoldStep(PipelineStep):

  def __init__(self, model_dir: str, input_dir: str, output_dir: str, model_name: str,
               num_recycles: int = 3, zip_outputs: bool = False, num_models: int = 5,
               msa_host_url: str = ColabFoldContainer.__COLABFOLD_API_URL, max_parallel: int = 1,
               client: docker.client.DockerClient = None):
    self.model_dir = model_dir
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.num_recycles = num_recycles
    self.zip_outputs = zip_outputs
    self.model_name = model_name
    self.num_models = num_models
    self.msa_host_url = msa_host_url
    self.max_parallel = max_parallel
    self.client = client

  def execute(self, sample_set: SampleSet) -> SampleSet:
    tmp_input_folder = self.__create_tmp_input_folder()
    self.__create_input_fastas(sample_set, tmp_input_folder)
    container = self.__create_container(tmp_input_folder)
    if not container.run(self.client):
      raise RuntimeError('ColabFold container failed')
    new_set = self.__create_new_sample_set(sample_set)
    self.__remove_tmp_input_folder(tmp_input_folder)
    return new_set

  def __create_container(self, input_dir: str) -> ColabFoldContainer:
    return ColabFoldContainer(
      model_dir=self.model_dir,
      input_dir=input_dir,
      output_dir=self.output_dir,
      num_recycle=self.num_recycles,
      model_name=self.model_name,
      zip_outputs=self.zip_outputs,
      num_models=self.num_models,
      msa_host_url=self.msa_host_url,
      max_parallel=self.max_parallel
    )

  def __create_tmp_input_folder(self) -> str:
    tmp_input_folder = os.path.join(self.input_dir, '_tmp')
    os.makedirs(tmp_input_folder, exist_ok=True)
    return tmp_input_folder

  @staticmethod
  def __create_input_fastas(sample_set: SampleSet, tmp_input_folder: str):
    for sample in sample_set.samples():
      out_fasta = os.path.join(tmp_input_folder, f'{sample.name}.fa')
      write_fasta([seq.protein for seq in sample.sequences()], out_fasta, overwrite=True)

  @staticmethod
  def __remove_tmp_input_folder(tmp_input_folder: str):
    shutil.rmtree(tmp_input_folder)

  def __create_new_sample_set(self, old_sample_set: SampleSet) -> SampleSet:
    # new_sample_set = SampleSet()
    # for sample in old_sample_set.samples():
    #   for sequence in sample.sequences():
    #     seq_
    return SampleSet()
