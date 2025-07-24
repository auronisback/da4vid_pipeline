import logging
import os
from typing import List, Dict, Any

import docker
import dotenv
import spython.main
import yaml

from da4vid.containers.carbonara import CARBonAraContainer
from da4vid.containers.colabfold import ColabFoldContainer
from da4vid.containers.docker import DockerExecutorBuilder
from da4vid.containers.masif import MasifContainer
from da4vid.containers.omegafold import OmegaFoldContainer
from da4vid.containers.pesto import PestoContainer
from da4vid.containers.pmpnn import ProteinMPNNContainer
from da4vid.containers.rfdiffusion import RFdiffusionContainer
from da4vid.containers.singularity import SingularityExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io import read_from_pdb
from da4vid.model.proteins import Protein, Epitope
from da4vid.model.samples import Sample
from da4vid.pipeline.generation import RFdiffusionStep, BackboneFilteringStep, ProteinMPNNStep, CARBonAraStep
from da4vid.pipeline.interaction import MasifStep, PestoStep, InteractionWindowEvaluationStep
from da4vid.pipeline.steps import CompositeStep, PipelineStep, PipelineRootStep, FoldCollectionStep, ContainerizedStep
from da4vid.pipeline.validation import OmegaFoldStep, SequenceFilteringStep, ColabFoldStep


class Da4vidConfigurationError(Exception):
  def __init__(self, errors: List[str]):
    super().__init__()
    self.errors = errors

  def __str__(self):
    return '; '.join(self.errors)


class StaticConfig:
  """
  Stores the static configuration of the pipeline, such as folders in which models
  are stored, image names and gpu managers.
  """

  class DockerStaticConfig:
    """
    Defines the static configuration when Docker backend is used to execute containers.
    """

    def __init__(self, client: docker.DockerClient, rfdiffusion_image: str, protein_mpnn_image: str,
                 omegafold_image: str, colabfold_image: str, masif_image: str, carbonara_image: str,
                 pesto_image: str):
      """
      Creates the docker configuration object that stores the static configuration of the pipeline.
      :param client: The client used to query Docker APIs
      :param rfdiffusion_image: Tag of the RFdiffusion docker image
      :param protein_mpnn_image: Tag of the ProteinMPNN docker image
      :param omegafold_image: Tag of the OmegaFold docker image
      :param colabfold_image: Tag of the Colabfold docker image
      :param masif_image: Tag of the MaSIF docker image
      :param carbonara_image: Tag of the CARBonAra docker image
      :param pesto_image: Tag of the PeSTo docker image
      :raise Da4vidConfigurationError: If specified container images are not available on the machine
      """
      self.client = client
      self.rfdiffusion_image = rfdiffusion_image
      self.protein_mpnn_image = protein_mpnn_image
      self.omegafold_image = omegafold_image
      self.colabfold_image = colabfold_image
      self.masif_image = masif_image
      self.carbonara_image = carbonara_image
      self.pesto_image = pesto_image
      # Dictionary for caching images
      self.__dict = {
        'rfdiffusion': self.rfdiffusion_image,
        'protein_mpnn': self.protein_mpnn_image,
        'omegafold': self.omegafold_image,
        'colabfold': self.colabfold_image,
        'masif': self.masif_image,
        'carbonara': self.carbonara_image,
        'pesto': self.pesto_image
      }
      self.__check_images()

    def get_image_name(self, img_type: str) -> str:
      return self.__dict.get(img_type, None)

    def __check_images(self) -> None:
      errors = [f'Image not found: {image}' for image in [self.rfdiffusion_image, self.protein_mpnn_image,
                                                          self.omegafold_image, self.colabfold_image, self.masif_image]
                if self.client.images.list(name=image) is None]
      if errors:
        raise Da4vidConfigurationError(errors)

    def __str__(self, indent: int = 0):
      return (f'{" " * indent}Docker Configuration:\n'
              f'{" " * indent} - RFdiffusion image: {self.rfdiffusion_image}\n'
              f'{" " * indent} - Protein MPNN image: {self.protein_mpnn_image}\n'
              f'{" " * indent} - OmegaFold image: {self.omegafold_image}\n'
              f'{" " * indent} - Colabfold image: {self.colabfold_image}\n'
              f'{" " * indent} - Masif image: {self.masif_image}\n'
              f'{" " * indent} - CARBonAra image: {self.carbonara_image}')

  class SingularityStaticConfig:
    """
    Defines the static configuration when Singularity backend is used to execute containers.
    """

    def __init__(self, rfdiffusion_sif: str, protein_mpnn_sif: str, omegafold_sif: str,
                 colabfold_sif: str, masif_sif: str, carbonara_sif: str, pesto_sif: str):
      self.rfdiffusion_sif = rfdiffusion_sif
      self.protein_mpnn_sif = protein_mpnn_sif
      self.omegafold_sif = omegafold_sif
      self.colabfold_sif = colabfold_sif
      self.masif_sif = masif_sif
      self.carbonara_sif = carbonara_sif
      self.pesto_sif = pesto_sif
      # Caching dict
      self.__dict = {
        'rfdiffusion': self.rfdiffusion_sif,
        'protein_mpnn': self.protein_mpnn_sif,
        'omegafold': self.omegafold_sif,
        'colabfold': self.colabfold_sif,
        'masif': self.masif_sif,
        'carbonara': self.carbonara_sif,
        'pesto': self.pesto_sif
      }
      self.__check_sif_paths()

    def get_sif_path_from_type(self, img_type: str) -> str:
      return self.__dict.get(img_type, None)

    def __check_sif_paths(self) -> None:
      errors = [f'SIF file not found: {sif_path}' for sif_path in self.__dict.values()
                if not os.path.isfile(sif_path)]
      if errors:
        raise Da4vidConfigurationError(errors)

    def __str__(self, indent: int = 0):
      return (f'{" " * indent}Singularity Configuration:\n'
              f'{" " * indent} - RFdiffusion SIF: {self.rfdiffusion_sif}\n'
              f'{" " * indent} - Protein MPNN SIF: {self.protein_mpnn_sif}\n'
              f'{" " * indent} - OmegaFold SIF: {self.omegafold_sif}\n'
              f'{" " * indent} - Colabfold SIF: {self.colabfold_sif}\n'
              f'{" " * indent} - Masif SIF: {self.masif_sif}\n'
              f'{" " * indent} - CARBonAra SIF: {self.carbonara_sif}')

  instance = None

  def __init__(self, gpu_manager: CudaDeviceManager,
               rfdiffusion_models_dir: str, omegafold_models_dir: str, colabfold_models_dir: str,
               omegafold_max_parallel: int, colabfold_max_parallel: int,
               docker_config: DockerStaticConfig = None, singularity_config: SingularityStaticConfig = None):
    """
    Creates a new instance of static configuration of the pipeline. This should not be invoked
    directly, but an instance should be created using the <em>load_from_yaml</em> static method.
    :param gpu_manager: The CUDA device manager used to assign GPU resources to containers
    :param rfdiffusion_models_dir: Directory where RFdiffusion models are locally stored
    :param omegafold_models_dir: Directory where Omegafold models are locally stored
    :param colabfold_models_dir: Directory where Colabfold models are locally stored
    :param omegafold_max_parallel: Number of parallel instances for OmegaFold containers
    :param colabfold_max_parallel: Number of parallel instances for ColabFold containers
    :param docker_config: Configuration if docker has been chosen as container backend
    :param singularity_config: Configuration if singularity has been chosen as container backend
    :raise Da4vidConfigurationError: If container backend configuration is invalid, or it is not
                                     possible to locate folders with models parameter
    """
    if not docker_config and not singularity_config:
      raise Da4vidConfigurationError(['Container backend not given: one among "docker" and "singularity" needed'])
    self.docker_configuration = docker_config
    self.singularity_config = singularity_config
    self.gpu_manager = gpu_manager
    self.rfdiffusion_models_dir = rfdiffusion_models_dir
    self.omegafold_models_dir = omegafold_models_dir
    self.colabfold_models_dir = colabfold_models_dir
    self.omegafold_max_parallel = omegafold_max_parallel
    self.colabfold_max_parallel = colabfold_max_parallel
    self.__check_parameters()

  def backend(self) -> str:
    """
    Gets the container backend for the pipeline.
    :return: 'docker' or 'singularity', according to the chosen container execution backend
    """
    return 'docker' if self.docker_configuration else 'singularity'

  def __check_parameters(self):
    errors = []
    # Checking model folders
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
      raise Da4vidConfigurationError(errors)

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
      docker_config = None
      singularity_config = None
      backend = os.environ.get('CONTAINER_BACKEND', 'docker')  # Defaults to docker
      if backend == 'docker':
        docker_config = StaticConfig.DockerStaticConfig(
          client=docker.from_env(),
          rfdiffusion_image=os.environ.get('RFDIFFUSION_IMAGE', RFdiffusionContainer.DEFAULT_IMAGE),
          protein_mpnn_image=os.environ.get('PROTEIN_MPNN_IMAGE', ProteinMPNNContainer.DEFAULT_IMAGE),
          omegafold_image=os.environ.get('OMEGAFOLD_IMAGE', OmegaFoldContainer.DEFAULT_IMAGE),
          colabfold_image=os.environ.get('COLABFOLD_IMAGE', ColabFoldContainer.DEFAULT_IMAGE),
          masif_image=os.environ.get('MASIF_IMAGE', MasifContainer.DEFAULT_IMAGE),
          carbonara_image=os.environ.get('CARBONARA_IMAGE', CARBonAraContainer.DEFAULT_IMAGE),
          pesto_image=os.environ.get('PESTO_IMAGE', PestoContainer.DEFAULT_IMAGE)
        )
      elif backend == 'singularity':
        singularity_config = StaticConfig.SingularityStaticConfig(
          rfdiffusion_sif=os.environ.get('RFDIFFUSION_SIF'),
          protein_mpnn_sif=os.environ.get('PROTEIN_MPNN_SIF'),
          omegafold_sif=os.environ.get('OMEGAFOLD_SIF'),
          colabfold_sif=os.environ.get('COLABFOLD_SIF'),
          masif_sif=os.environ.get('MASIF_SIF'),
          carbonara_sif=os.environ.get('CARBONARA_SIF'),
          pesto_sif=os.environ.get('PESTO_SIF')
        )  # Todo
      else:
        raise Da4vidConfigurationError([f'Unknown backend: {backend}'])
      StaticConfig.instance = StaticConfig(
        gpu_manager=CudaDeviceManager(),
        rfdiffusion_models_dir=os.environ.get('RFDIFFUSION_MODEL_FOLDER', None),
        omegafold_models_dir=os.environ.get('OMEGAFOLD_MODEL_FOLDER', None),
        colabfold_models_dir=os.environ.get('COLABFOLD_MODEL_FOLDER', None),
        omegafold_max_parallel=int(os.environ.get('OMEGAFOLD_MAX_PARALLEL', 1)),
        colabfold_max_parallel=int(os.environ.get('COLABFOLD_MAX_PARALLEL', 1)),
        docker_config=docker_config,
        singularity_config=singularity_config,
      )
    return StaticConfig.instance

  def __str__(self):
    s = f'StaticConfig:\n'
    s += f'  Backend: {self.backend()}\n'
    if self.backend() == 'docker':
      s += f'{self.docker_configuration.__str__(2)}\n'
    else:
      s += f'{self.singularity_config.__str__(2)}\n'
    return s + (f' - RFdiffusion models: {self.rfdiffusion_models_dir}\n'
                f' - OmegaFold models: {self.omegafold_models_dir}\n'
                f' - ColabFold models: {self.colabfold_models_dir}\n')


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
      pipeline = self.__process_root_element(data, os.path.dirname(yml_config))
      self.__validate_pipeline(pipeline)
      return pipeline

  def __get_executor_builder_for_step(self, step_type: str):
    if self.static_config.backend() == 'docker':
      return DockerExecutorBuilder().set_client(docker.from_env()).set_image(
        self.static_config.docker_configuration.get_image_name(step_type.lower()))
    else:
      return SingularityExecutorBuilder().set_client(spython.main.get_client()).set_sif_path(
        self.static_config.singularity_config.get_sif_path_from_type(step_type.lower())
      )

  def __process_root_element(self, el: Dict[str, Any], yml_path: str) -> PipelineRootStep:
    el_name = list(el.keys())[0]
    root_el = el[el_name]
    # Checks
    if 'folder' not in root_el.keys():
      raise self.PipelineCreationError('Root folder not specified')
    if 'antigen' not in root_el.keys():
      raise self.PipelineCreationError('Antigen path is not specified')
    if 'epitope' not in root_el.keys():
      raise self.PipelineCreationError('Epitope has not been specified')
    folder = root_el['folder']
    folder = folder if os.path.isabs(folder) else os.path.abspath(os.path.join(yml_path, folder))
    ag_path = root_el['antigen']
    ag_path = ag_path if os.path.isabs(ag_path) else os.path.abspath(os.path.join(yml_path, ag_path))
    antigen = read_from_pdb(os.path.abspath(ag_path))
    epitope = self.__create_epitope(antigen, root_el['epitope'])
    root_step = PipelineRootStep(
      name=el_name,
      antigen=Sample(name=antigen.name, filepath=antigen.filename, protein=antigen),
      epitope=epitope,
      folder=folder
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
          builder=self.__get_executor_builder_for_step('rfdiffusion').preserve_quotes_in_cmds(['"']),
          name=el.get('name', 'rfdiffusion'),
          parent=parent,
          epitope=root.epitope,
          model_dir=self.static_config.rfdiffusion_models_dir,
          gpu_manager=self.static_config.gpu_manager,
          config=RFdiffusionStep.RFdiffusionConfig(**{k: v for k, v in el.items() if k != 'name'}),

        )
      case 'backbone_filtering':
        return BackboneFilteringStep(
          name=el.get('name', 'backbone_filtering'),
          parent=parent,
          **{k: v for k, v in el.items() if k != 'name'}
        )
      case 'proteinmpnn':
        return ProteinMPNNStep(
          builder=self.__get_executor_builder_for_step('protein_mpnn'),
          name=el.get('name', 'proteinmpnn'),
          parent=parent,
          epitope=root.epitope,
          gpu_manager=self.static_config.gpu_manager,
          config=ProteinMPNNStep.ProteinMPNNConfig(**{k: v for k, v in el.items() if k != 'name'})
        )
      case 'omegafold':
        return OmegaFoldStep(
          builder=self.__get_executor_builder_for_step('omegafold'),
          name=el.get('name', 'omegafold'),
          parent=parent,
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
          builder=self.__get_executor_builder_for_step('colabfold'),
          name=el.get('name', 'colabfold'),
          parent=parent,
          gpu_manager=self.static_config.gpu_manager,
          model_dir=self.static_config.colabfold_models_dir,
          max_parallel=self.static_config.colabfold_max_parallel,
          config=ColabFoldStep.ColabFoldConfig(**{k: v for k, v in el.items() if k != 'name'})
        )
      case 'masif':
        return MasifStep(
          builder=self.__get_executor_builder_for_step('masif'),
          name=el.get('name', 'masif'),
          parent=parent,
          gpu_manager=self.static_config.gpu_manager
        )
      case 'pesto':
        return PestoStep(
          builder=self.__get_executor_builder_for_step('pesto'),
          name=el.get('name', 'pesto'),
          parent=parent,
          gpu_manager=self.static_config.gpu_manager
        )
      case 'fold_collection':
        return FoldCollectionStep(
          name=el.get('name', 'fold_collection'),
          model=el['model'],
          parent=parent
        )
      case 'carbonara':
        return CARBonAraStep(
          builder=self.__get_executor_builder_for_step('carbonara'),
          name=el.get('name', 'CARBonAra'),
          parent=parent,
          gpu_manager=self.static_config.gpu_manager,
          epitope=root.epitope,
          config=CARBonAraStep.CARBonAraConfig(
            num_sequences=el['num_sequences'],
            imprint_ratio=el.get('imprint_ratio', .5),
            sampling_method=el.get('sampling_method', CARBonAraContainer.SAMPLING_SAMPLED),
            ignored_amino_acids=None if 'ignored_amino_acids' not in el else
            [aa.strip() for aa in el['ignored_amino_acids'].split(' ')],
            ignore_water=bool(el.get('ignore_water', False)),
            ignore_het_atm=bool(el.get('ignore_het_atm', False))
          )
        )
      case 'interaction_window':
        return InteractionWindowEvaluationStep(
          name=el_name, parent=parent, folder=el.get('folder', None), gpu_manager=self.static_config.gpu_manager,
          epitope=root.epitope,
          offset=el.get('offset', 3),
          interaction_key=el['interaction_key']
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

  def __validate_pipeline(self, pipeline: PipelineRootStep) -> None:
    errors = self.__validate_step(pipeline)
    if errors:
      for err in errors:
        logging.error(err)
      raise Exception('Pipeline is not configured correctly')

  def __validate_step(self, step: PipelineStep) -> List[str]:
    errors = []
    if isinstance(step, CompositeStep):
      for sub_step in step.steps:
        errors += self.__validate_step(sub_step)
    if isinstance(step, ContainerizedStep):
      if isinstance(step.builder, SingularityExecutorBuilder):
        if not os.path.exists(step.builder.sif_path):
          errors.append(f'Container {step.full_name()} does not exist: {step.builder.sif_path}')
    return errors
