import os

import yaml


class RunConfig:
  """
  Configuration for a single run.
  """

  def __init__(self, name: str | int, root_folder: str, cutoff: int = None, percentage: bool = False):
    """
    Initializes the run configuration, specifying its name.
    :param name: The full name identifier of the run, if it is a string, or the
                 index of the run
    """
    self.name = name if isinstance(name, str) else f'run{name}'
    self.root_folder = root_folder
    self.cutoff = cutoff
    self.percentage = percentage
    self.rfdiffusion_config = None
    self.backbone_filtering_config = None
    self.proteinmpnn_config = None
    self.omegafold_config = None
    self.sequence_filtering_config = None

  class __RFdiffusionConfig:
    """
    Class encapsulating the configuration for RFdiffusion.
    """

    def __init__(self, run, model_dir: str, num_designs: int, partial_T: int,
                 contacts_threshold: float, rog_potential: float):
      self.run = run
      self.model_dir = model_dir
      self.num_designs = num_designs
      self.partial_T = partial_T
      self.contacts_threshold = contacts_threshold
      self.rog_potential = rog_potential

    def output_folder(self) -> str:
      return os.path.join(self.run.root_folder, 'rfdiffusion', 'outputs')

    def __str__(self):
      return (f'rfdiffusion:\n - model_dir: {self.model_dir}\n - num_designs: {self.num_designs}\n'
              f' - partial_T: {self.partial_T}\n - contacts_threshold: {self.contacts_threshold}\n'
              f' - rog_potential: {self.rog_potential}\n')

  def add_rfdiffusion_configuration(self, model_dir: str, num_designs: int, partial_T: int, contacts_threshold: float,
                                    rog_potential: float):
    self.rfdiffusion_config = RunConfig.__RFdiffusionConfig(run=self, model_dir=model_dir, num_designs=num_designs,
                                                            partial_T=partial_T, contacts_threshold=contacts_threshold,
                                                            rog_potential=rog_potential)

  class __BackboneFilteringConfig:
    def __init__(self, run, ss_threshold: float, rog_cutoff: float, rog_percentage: bool):
      self.run = run
      self.ss_threshold = ss_threshold
      self.rog_cutoff = rog_cutoff
      self.rog_percentage = rog_percentage

    def output_folder(self) -> str:
      return os.path.join(self.run.root_folder, 'filtered_backbones')

    def __str__(self):
      return (f'backbone_filtering:\n - ss_threshold: {self.ss_threshold}\n - rog_cutoff: {self.rog_cutoff}\n'
              f' - rog_percentage: {self.rog_percentage}\n')

  def add_backbone_filtering_configuration(self, ss_threshold: int, rog_cutoff: float, rog_percentage: bool = False):
    self.backbone_filtering_config = RunConfig.__BackboneFilteringConfig(
      run=self,
      ss_threshold=ss_threshold,
      rog_cutoff=rog_cutoff,
      rog_percentage=rog_percentage
    )

  class __ProteinMPNNConfig:
    def __init__(self, run, seqs_per_target: int, sampling_temp: float, backbone_noise: float):
      self.run = run
      self.seqs_per_target = seqs_per_target
      self.sampling_temp = sampling_temp
      self.backbone_noise = backbone_noise

    def output_folder(self) -> str:
      return os.path.join(self.run.root_folder, 'protein_mpnn', 'outputs')

    def complete_output_folder(self) -> str:
      """
      Gets the full path to output FASTAs.
      :return: The path to the folder where FASTAs are
      """
      return os.path.join(self.output_folder(), 'seqs')

    def __str__(self):
      return (f'protein_mpnn:\n - seqs_per_target: {self.seqs_per_target}\n - sampling_temp: {self.sampling_temp}\n'
              f' - backbone_noise: {self.backbone_noise}\n')

  def add_proteinmpnn_configuration(self, seqs_per_target: int, sampling_temp: float, backbone_noise: float):
    self.proteinmpnn_config = RunConfig.__ProteinMPNNConfig(
      run=self,
      seqs_per_target=seqs_per_target,
      sampling_temp=sampling_temp,
      backbone_noise=backbone_noise
    )

  class __OmegaFoldConfig:
    def __init__(self, run, model_dir: str, num_recycles: int, model_weights: str):
      self.run = run
      self.model_dir = model_dir
      self.num_recycles = num_recycles
      self.model_weights = model_weights

    def output_folder(self) -> str:
      return os.path.join(self.run.root_folder, 'omegafold', 'outputs')

    def __str__(self):
      return (f'omegafold:\n - model_dir: {self.model_dir}\n - num_recycles: {self.num_recycles}\n'
              f' - model_weights: {self.model_weights}\n')

  def add_omegafold_configuration(self, model_dir: str,
                                  num_recycles: int, model_weights: str):
    self.omegafold_config = RunConfig.__OmegaFoldConfig(
      run=self,
      model_dir=model_dir,
      num_recycles=num_recycles,
      model_weights=model_weights
    )

  class __SequenceFilteringConfig:
    def __init__(self, plddt_threshold: float, rog_cutoff: float):
      self.plddt_threshold = plddt_threshold
      self.rog_cutoff = rog_cutoff

    def __str__(self):
      return (f'sequence_filtering:\n - plddt_threshold: {self.plddt_threshold}\n'
              f' - rog_threshold: {self.rog_cutoff}\n')

  def add_sequence_filtering_configuration(self, plddt_threshold: float, rog_cutoff: float = None):
    self.sequence_filtering_config = RunConfig.__SequenceFilteringConfig(
      plddt_threshold=plddt_threshold,
      rog_cutoff=rog_cutoff
    )

  def get_rfdiffusion_configuration(self) -> __RFdiffusionConfig:
    return self.rfdiffusion_config

  def get_backbone_filtering_configuration(self) -> __BackboneFilteringConfig:
    return self.backbone_filtering_config

  def get_proteinmpnn_configuration(self) -> __ProteinMPNNConfig:
    return self.proteinmpnn_config

  def get_omegafold_configuration(self) -> __OmegaFoldConfig:
    return self.omegafold_config

  def get_sequence_filtering_configuration(self) -> __SequenceFilteringConfig:
    return self.sequence_filtering_config

  def output_folder(self) -> str:
    return os.path.join(self.root_folder, 'outputs')

  def __str__(self):
    return (f'run: {self.name}:\n - root_folder: {self.root_folder}\n'
            f'{self.rfdiffusion_config if self.rfdiffusion_config else ""}'
            f'{self.backbone_filtering_config if self.backbone_filtering_config else ""}'
            f'{self.proteinmpnn_config}'
            f'{self.omegafold_config}'
            f'{self.sequence_filtering_config}')


class PipelineConfig:
  def __init__(self):
    self.runs = []

  def get_run(self, iteration: int) -> RunConfig:
    """
    runs are 1-indexed
    :param iteration:
    :return:
    """
    return self.runs[iteration - 1]

  def add_run_configuration(self, config: RunConfig, idx: int = None):
    if idx is None:
      self.runs.append(config)
    else:
      self.runs.insert(idx, config)

  def __str__(self):
    s = ''
    for r in self.runs:
      s += str(r)
    return s

  @staticmethod
  def load_from_yaml(cfg_file: str):
    with open(cfg_file) as f:
      data = yaml.safe_load(f)
      config = PipelineConfig()
      for el in data:
        run_name = list(el.keys())[0]
        run_el = el[run_name]
        run = RunConfig(run_name, run_el.get('root', None), run_el.get('cutoff', None), run_el.get('percentage', None))
        if 'rfdiffusion' in run_el.keys():
          run.add_rfdiffusion_configuration(**run_el['rfdiffusion'])
        if 'backbone_filtering' in run_el.keys():
          run.add_backbone_filtering_configuration(**run_el['backbone_filtering'])
        if 'proteinmpnn' in run_el.keys():
          run.add_proteinmpnn_configuration(**run_el['proteinmpnn'])
        if 'omegafold' in run_el.keys():
          run.add_omegafold_configuration(**run_el['omegafold'])
        if 'sequence_filtering' in run_el.keys():
          run.add_sequence_filtering_configuration(**run_el['sequence_filtering'])
        config.add_run_configuration(run)
      return config
