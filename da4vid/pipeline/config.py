import os
from typing import List, Dict, Any

import docker
import dotenv
import yaml

from da4vid.docker.colabfold import ColabFoldContainer
from da4vid.docker.masif import MasifContainer
from da4vid.docker.omegafold import OmegaFoldContainer
from da4vid.docker.pmpnn import ProteinMPNNContainer
from da4vid.docker.rfdiffusion import RFdiffusionContainer
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io import read_from_pdb
from da4vid.model.proteins import Protein, Epitope
from da4vid.model.samples import Sample
from da4vid.pipeline.generation import RFdiffusionStep, BackboneFilteringStep, ProteinMPNNStep
from da4vid.pipeline.steps import CompositeStep, PipelineStep, PipelineRootStep
from da4vid.pipeline.validation import OmegaFoldStep, SequenceFilteringStep, ColabFoldStep


class StaticConfig:
  """
  Stores the static configuration of the pipeline, such as folders in which models
  are stored, image names and gpu managers.
  """

  instance = None

  def __init__(self, client: docker.DockerClient, gpu_manager: CudaDeviceManager,
               rfdiffusion_models_dir: str, omegafold_models_dir: str, colabfold_models_dir: str,
               rfdiffusion_image: str, protein_mpnn_image: str, omegafold_image: str, colabfold_image: str,
               masif_image: str, omegafold_max_parallel: int, colabfold_max_parallel: int):
    """
    Creates a new instance of static configuration of the pipeline. This should not be invoked
    directly, but an instance should be created using the <em>load_from_yaml</em> static method.
    :param client: The client used to query Docker APIs
    :param gpu_manager: The CUDA device manager used to assign GPU resources to containers
    :param rfdiffusion_models_dir: Directory where RFdiffusion models are locally stored
    :param omegafold_models_dir: Directory where Omegafold models are locally stored
    :param colabfold_models_dir: Directory where Colabfold models are locally stored
    :param rfdiffusion_image: Tag of the RFdiffusion docker image
    :param protein_mpnn_image: Tag of the ProteinMPNN docker image
    :param omegafold_image: Tag of the OmegaFold docker image
    :param colabfold_image: Tag of the Colabfold docker image
    :param masif_image: Tag of the MaSIF docker image
    :param omegafold_max_parallel: Number of parallel instances for OmegaFold containers
    :param colabfold_max_parallel: Number of parallel instances for ColabFold containers
    :raise Da4vidConfigurationError: If images are not available, or models folder are invalid
    """
    self.client = client
    self.gpu_manager = gpu_manager
    self.rfdiffusion_models_dir = rfdiffusion_models_dir
    self.omegafold_models_dir = omegafold_models_dir
    self.colabfold_models_dir = colabfold_models_dir
    self.rfdiffusion_image = rfdiffusion_image
    self.protein_mpnn_image = protein_mpnn_image
    self.omegafold_image = omegafold_image
    self.colabfold_image = colabfold_image
    self.masif_image = masif_image
    self.omegafold_max_parallel = omegafold_max_parallel
    self.colabfold_max_parallel = colabfold_max_parallel
    self.__check_parameters()

  def __check_parameters(self):
    errors = []
    # Checking images
    for image in [self.rfdiffusion_image, self.protein_mpnn_image, self.omegafold_image,
                  self.colabfold_image, self.masif_image]:
      if not self.client.images.list(filters={'reference': image}):
        errors.append(f'Image not found: {image}')
    # Checking directories
    if self.rfdiffusion_models_dir is None or not os.path.isdir(self.rfdiffusion_models_dir):
      errors.append(f'Invalid RFdiffusion model dir: {self.rfdiffusion_models_dir}')
    if self.omegafold_models_dir is None or not os.path.isdir(self.omegafold_models_dir):
      errors.append(f'Invalid OmegaFold model dir: {self.omegafold_models_dir}')
    if self.colabfold_models_dir is None or not os.path.isdir(self.colabfold_models_dir):
      errors.append(f'Invalid Colabfold models dir: {self.colabfold_models_dir}')
    # Checking parallel containers numbers are positive
    if self.omegafold_max_parallel < 1:
      errors.append(f'Invalid number of OmegaFold parallel containers: {self.omegafold_max_parallel}')
    if self.colabfold_max_parallel < 1:
      errors.append(f'Invalid number of ColabFold parallel containers: {self.colabfold_max_parallel}')
    if errors:
      raise self.Da4vidConfigurationError(errors)

  class Da4vidConfigurationError(Exception):
    def __init__(self, errors: List[str]):
      super().__init__()
      self.errors = errors

    def __str__(self):
      return '; '.join(self.errors)

  @staticmethod
  def get(env_file: str = None):
    """
    Gets the pipeline static configuration. If it has not been already
    inited, it will be created from the <em>.env</em> file.
    :param env_file: If given, the configuration will be loaded from
                        the specified dotenv file
    :return: The static configuration of the pipeline.
    """
    if not StaticConfig.instance:
      dotenv.load_dotenv(env_file)
      StaticConfig.instance = StaticConfig(
        client=docker.from_env(),
        gpu_manager=CudaDeviceManager(),
        rfdiffusion_image=os.environ.get('RFDIFFUSION_IMAGE', RFdiffusionContainer.DEFAULT_IMAGE),
        rfdiffusion_models_dir=os.environ.get('RFDIFFUSION_MODEL_FOLDER', None),
        protein_mpnn_image=os.environ.get('PROTEIN_MPNN_FOLDER', ProteinMPNNContainer.DEFAULT_IMAGE),
        omegafold_image=os.environ.get('OMEGAFOLD_IMAGE', OmegaFoldContainer.DEFAULT_IMAGE),
        omegafold_models_dir=os.environ.get('OMEGAFOLD_MODEL_FOLDER', None),
        colabfold_image=os.environ.get('COLABFOLD_IMAGE', ColabFoldContainer.DEFAULT_IMAGE),
        colabfold_models_dir=os.environ.get('COLABFOLD_MODEL_FOLDER', None),
        masif_image=os.environ.get('MASIF_IMAGE', MasifContainer.DEFAULT_IMAGE),
        omegafold_max_parallel=int(os.environ.get('OMEGAFOLD_MAX_PARALLEL', 1)),
        colabfold_max_parallel=int(os.environ.get('COLABFOLD_MAX_PARALLEL', 1))
      )
    return StaticConfig.instance

  def __str__(self):
    return (f'StaticConfig:'
            f' - RFdiffusion image: {self.rfdiffusion_image}\n'
            f' - RFdiffusion models: {self.rfdiffusion_models_dir}\n'
            f' - ProteinMPNN image: {self.protein_mpnn_image}\n'
            f' - OmegaFold image: {self.omegafold_image}\n'
            f' - OmegaFold models: {self.omegafold_models_dir}\n'
            f' - ColabFold image: {self.colabfold_image}\n'
            f' - ColabFold models: {self.colabfold_models_dir}\n'
            f' - MaSIF image: {self.masif_image}\n')


class PipelineCreator:

  def __init__(self, static_config_env_file: str = None):
    self.static_config = StaticConfig.get(static_config_env_file)

  class PipelineCreationError(Exception):
    def __init__(self, message: str):
      super().__init__(message)

  def from_yml(self, yml_config: str) -> PipelineRootStep:
    """
    Creates the pipeline from the pipeline configuration file in YAML format.
    :param yml_config: The path to the configuration file
    :return: A single step or a composite step including all steps defined in the pipeline
    """
    with open(yml_config) as f:
      data = yaml.safe_load(f)
      return self.__process_root_element(data)

  def __process_root_element(self, el: Dict[str, Any]) -> PipelineRootStep:
    el_name = list(el.keys())[0]
    root_el = el[el_name]
    # Checks
    if 'folder' not in root_el.keys():
      raise self.PipelineCreationError('Root folder not specified')
    if 'antigen' not in root_el.keys():
      raise self.PipelineCreationError('Antigen path is not specified')
    if 'epitope' not in root_el.keys():
      raise self.PipelineCreationError('Epitope has not been specified')
    antigen = read_from_pdb(os.path.abspath(root_el['antigen']))
    epitope = self.__create_epitope(antigen, root_el['epitope'])
    root_step = PipelineRootStep(
      name=el_name,
      antigen=Sample(name=antigen.name, filepath=antigen.filename, protein=antigen),
      epitope=epitope,
      folder=os.path.abspath(root_el['folder'])
    )
    root_step.add_step([self.__process_element(root_step, root_step, step_el)
                        for step_el in root_el['steps']])
    self.__check_different_context_folder(root_step)
    return root_step

  @staticmethod
  def __create_epitope(antigen: Protein, epitope_str: str) -> Epitope:
    # TODO: refactor epitope to accept not contiguous epitopes
    chain = epitope_str[0]
    start, end = epitope_str[1:].split('-')
    return Epitope(chain, int(start), int(end), antigen)

  def __process_element(self, root: PipelineRootStep, parent: CompositeStep,
                        el: Dict[str, Any]) -> PipelineStep:
    el_name = list(el.keys())[0]
    el = el[el_name]
    match el_name:
      case 'rfdiffusion':
        return RFdiffusionStep(
          name=el.get('name', 'rfdiffusion'),
          parent=parent,
          epitope=root.epitope,
          model_dir=self.static_config.rfdiffusion_models_dir,
          client=self.static_config.client,
          image=self.static_config.rfdiffusion_image,
          gpu_manager=self.static_config.gpu_manager,
          config=RFdiffusionStep.RFdiffusionConfig(**{k: v for k, v in el.items() if k != 'name'})
        )
      case 'backbone_filtering':
        return BackboneFilteringStep(
          name=el.get('name', 'backbone_filtering'),
          parent=parent,
          **{k: v for k, v in el.items() if k != 'name'}
        )
      case 'proteinmpnn':
        return ProteinMPNNStep(
          name=el.get('name', 'proteinmpnn'),
          parent=parent,
          epitope=root.epitope,
          client=self.static_config.client,
          image=self.static_config.protein_mpnn_image,
          gpu_manager=self.static_config.gpu_manager,
          config=ProteinMPNNStep.ProteinMPNNConfig(**{k: v for k, v in el.items() if k != 'name'})
        )
        pass
      case 'omegafold':
        return OmegaFoldStep(
          name=el.get('name', 'omegafold'),
          parent=parent,
          client=self.static_config.client,
          image=self.static_config.omegafold_image,
          model_dir=self.static_config.omegafold_models_dir,
          gpu_manager=self.static_config.gpu_manager,
          max_parallel=self.static_config.omegafold_max_parallel,
          config=OmegaFoldStep.OmegaFoldConfig(**{k: v for k, v in el.items() if k != 'name'})
        )
      case 'sequence_filtering':
        return SequenceFilteringStep(
          name=el.get('name', 'sequence_filtering'),
          parent=parent,
          gpu_manager=self.static_config.gpu_manager,
          **{k: v for k, v in el.items() if k != 'name'}
        )
      case 'colabfold':
        return ColabFoldStep(
          name=el.get('name', 'colabfold'),
          parent=parent,
          client=self.static_config.client,
          image=self.static_config.colabfold_image,
          gpu_manager=self.static_config.gpu_manager,
          model_dir=self.static_config.colabfold_models_dir,
          max_parallel=self.static_config.colabfold_max_parallel,
          config=ColabFoldStep.ColabFoldConfig(**{k: v for k, v in el.items() if k != 'name'})
        )
      case _:  # Any other case is hopefully a composite step
        comp_step = CompositeStep(name=el_name, parent=parent, folder=el.get('folder', None))
        comp_step.add_step([self.__process_element(root, comp_step, step_el) for step_el in el['steps']])
        # Checking if all children have different context folder
        self.__check_different_context_folder(comp_step)
        return comp_step

  @staticmethod
  def __check_different_context_folder(comp_step: CompositeStep):
    seen = set()
    for step in comp_step.steps:
      ctx_folder = step.get_context_folder()
      if ctx_folder not in seen:
        seen.add(ctx_folder)
      else:
        raise PipelineCreator.PipelineCreationError(f'Two steps have the same context folder: {ctx_folder}')
