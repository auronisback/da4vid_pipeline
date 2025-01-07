import os
import shutil
import sys
from typing import List

import torch
from tqdm import tqdm

from da4vid.docker.colabfold import ColabFoldContainer
from da4vid.docker.omegafold import OmegaFoldContainer
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io import read_from_pdb
from da4vid.io.fasta_io import write_fasta
from da4vid.metrics import evaluate_plddt, rog
from da4vid.model.proteins import Proteins
from da4vid.model.samples import SampleSet, Fold
from da4vid.pipeline.steps import PipelineStep, DockerStep


class OmegaFoldStep(DockerStep):
  """
  Class implementing an OmegaFold prediction step in the pipeline.
  """
  class OmegaFoldConfig:
    def __init__(self, num_recycles: int = 5, model_weights: str = '2'):
      """
      Creates a configuration for the OmegaFold step.
      :param num_recycles: The number of recycles for the model. Defaults to 5
      :param model_weights: The version of weights used by the model. Defaults to the
                            most recent model, i.e. "2"
      """
      self.num_recycles = num_recycles
      self.model_weights = model_weights

    def __str__(self):
      return (f'omegafold:\n '
              f' - num_recycles: {self.num_recycles}\n'
              f' - model_weights: {self.model_weights}\n')

  def __init__(self, model_dir: str, gpu_manager: CudaDeviceManager,
               config: OmegaFoldConfig, max_parallel: int = 1, **kwargs):
    super().__init__(**kwargs)
    self.input_dir = os.path.join(self.get_context_folder(), 'inputs')
    self.output_dir = os.path.join(self.get_context_folder(), 'outputs')
    self.gpu_manager = gpu_manager
    self.max_parallel = max_parallel
    self.config = config
    self.container = OmegaFoldContainer(
      model_dir=model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=self.config.num_recycles,
      model_weights=self.config.model_weights,
      client=self.client,
      gpu_manager=self.gpu_manager,
      max_parallel=self.max_parallel,
      out_logfile=self.out_logfile,
      err_logfile=self.err_logfile
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
    print(f' - input folder: {self.output_dir}')
    print(f' - model weights: {self.config.model_weights}')
    print(f' - num_recycles: {self.config.num_recycles}')
    if not self.container.run():
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
      plddt = evaluate_plddt(folded, plddt_prop='omegafold.plddt', device=self.gpu_manager.next_device().name)
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
  def __init__(self, model: str, plddt_threshold: float, average_cutoff: float = None, rog_cutoff: float = None,
               max_samples: int = None, gpu_manager: CudaDeviceManager = None, **kwargs):
    """
    Initializes the sequence filtering step, evaluating the average pLDDT
    of folding predictions for all samples in each original backbone protein
    and filtering those proteins under a certain threshold. Filtered proteins will be
    also cut off according to their RoG, in ascending order.
    :param model: The model used to retrieve pLDDT predictions
    :param plddt_threshold: The threshold on samples average pLDDT values
    :param average_cutoff: The number of samples used to evaluate average pLDDT. If None, all
                       samples for the backbone will be used
    :param rog_cutoff: The cutoff for RoG filtering. If None, no cutoff will be applied
    :param max_samples: The maximum number of samples to retain for each original protein. If None, no
                        limit to the number of samples will be used. Defaults to None
    :param gpu_manager: The CUDA device manager used to assign GPUs. If not given, CPU will be used
                       instead
    """
    super().__init__(**kwargs)
    self.model = model
    self.plddt_threshold = plddt_threshold
    self.average_cutoff = average_cutoff
    self.rog_cutoff = rog_cutoff
    self.max_samples = max_samples
    self.device = gpu_manager.next_device().name if gpu_manager else 'cpu'

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
    plddts = evaluate_plddt([f.protein for f in folds], f'{self.model}.plddt', self.device)
    mean_plddt = torch.mean(plddts)
    if mean_plddt >= self.plddt_threshold:
      # Adding pLDDT metric to fold
      for plddt, fold in zip(plddts, folds):
        fold.metrics.add_metric('plddt', plddt)
      folds.sort(key=lambda f: f.metrics.get_metric('plddt'), reverse=True)
      return folds[:self.average_cutoff if self.average_cutoff else len(folds)]
    return []

  def __filter_by_rog(self, folds: List[Fold]) -> List[Fold]:
    rog([fold.protein for fold in folds], device=self.device)  # Evaluating RoG
    for fold in folds:
      fold.metrics.add_metric('rog', fold.protein.get_prop('rog'))
    folds.sort(key=lambda f: f.metrics.get_metric('rog'))
    folds = folds[:self.rog_cutoff]
    return folds


class ColabFoldStep(DockerStep):
  class ColabFoldConfig:
    def __init__(self, num_recycles: int = 3, model_name: str = ColabFoldContainer.MODEL_NAMES[0],
                 num_models: int = 5, msa_host_url: str = ColabFoldContainer.COLABFOLD_API_URL,
                 zip_outputs: bool = False):
      """
      Creates a new configuration for the subsequent colabfold step.
      :param num_recycles: The number of recycles for predictions. Defaults to 3
      :param model_name: The name of the used model. Defaults to the first one defined in the container
      :param num_models: The number of models used for the ranking. Defaults to 5
      :param msa_host_url: The URL for the MSA server. Defaults to the standard URL of alphafold
      :param zip_outputs: Whether the inputs should be compressed. Defaults to False (no compression)
      """
      self.num_recycles = num_recycles
      self.model_name = model_name
      self.num_models = num_models
      self.msa_host_url = msa_host_url
      self.zip_outputs = zip_outputs

    def __str__(self):
      return (f'colabfold:\n'
              f' - num_recycles: {self.num_recycles}\n'
              f' - model_name: {self.model_name}\n'
              f' - num_models: {self.num_models}\n'
              f' - msa_host_url: {self.msa_host_url}\n'
              f' - zip_outputs: {self.zip_outputs}\n')

  def __init__(self, model_dir: str, gpu_manager: CudaDeviceManager,
               config: ColabFoldConfig, max_parallel: int = 1, **kwargs):
    super().__init__(**kwargs)
    self.input_dir = os.path.join(self.get_context_folder(), 'inputs')
    self.output_dir = os.path.join(self.get_context_folder(), 'outputs')
    self.model_dir = model_dir
    self.config = config
    self.gpu_manager = gpu_manager
    self.max_parallel = max_parallel

  def execute(self, sample_set: SampleSet) -> SampleSet:
    tmp_input_folder = self.__create_tmp_input_folder()
    self.__create_input_fastas(sample_set, tmp_input_folder)
    container = self.__create_container(tmp_input_folder)
    if not container.run():
      raise RuntimeError('ColabFold container failed')
    new_set = self.__create_new_sample_set(sample_set)
    self.__remove_tmp_input_folder(tmp_input_folder)
    return new_set

  def __create_container(self, input_dir: str) -> ColabFoldContainer:
    return ColabFoldContainer(
      image=self.image,
      model_dir=self.model_dir,
      input_dir=input_dir,
      output_dir=self.output_dir,
      num_recycle=self.config.num_recycles,
      model_name=self.config.model_name,
      zip_outputs=self.config.zip_outputs,
      num_models=self.config.num_models,
      msa_host_url=self.config.msa_host_url,
      max_parallel=self.max_parallel,
      client=self.client,
      gpu_manager=self.gpu_manager
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
