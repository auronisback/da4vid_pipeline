import os
import sys
from typing import List, Dict, Any, IO

import docker
import dotenv
import yaml

from da4vid.docker.carbonara import CARBonAraContainer
from da4vid.docker.colabfold import ColabFoldContainer
from da4vid.docker.masif import MasifContainer
from da4vid.docker.omegafold import OmegaFoldContainer
from da4vid.docker.pmpnn import ProteinMPNNContainer
from da4vid.docker.rfdiffusion import RFdiffusionContainer
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io import read_from_pdb
from da4vid.model.proteins import Protein, Epitope
from da4vid.model.samples import Sample
from da4vid.pipeline.generation import RFdiffusionStep, BackboneFilteringStep, ProteinMPNNStep, CARBonAraStep
from da4vid.pipeline.interaction import MasifStep
from da4vid.pipeline.steps import CompositeStep, PipelineStep, PipelineRootStep, FoldCollectionStep
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
               masif_image: str, omegafold_max_parallel: int, colabfold_max_parallel: int, carbonara_image: str):
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
    :param carbonara_image: Tag of the CARBonAra docker image
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
    self.carbonara_image = carbonara_image
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
        colabfold_max_parallel=int(os.environ.get('COLABFOLD_MAX_PARALLEL', 1)),
        carbonara_image=os.environ.get('CARBONARA_IMAGE', CARBonAraContainer.DEFAULT_IMAGE)
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
      return self.__process_root_element(data, os.path.dirname(yml_config))

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
      case 'masif':
        return MasifStep(
          name=el.get('name', 'masif'),
          parent=parent,
          client=self.static_config.client,
          image=self.static_config.masif_image,
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
          name=el.get('name', 'CARBonAra'),
          parent=parent,
          image=self.static_config.carbonara_image,
          client=self.static_config.client,
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


class PipelinePrinter:
  ELBOW = "└──"
  PIPE = "│  "
  TEE = "├──"
  BLANK = "   "

  def __init__(self, file: IO = sys.stdout):
    self.file = file

  def print(self, pipeline: PipelineRootStep) -> None:
    self.__print_root_step(pipeline)
    for i, step in enumerate(pipeline.steps):
      self.__print_step(step, header='', last=i == len(pipeline.steps) - 1)

  def __print_root_step(self, pipeline: PipelineRootStep) -> None:
    print(f'Pipeline: {pipeline.name}', file=self.file)
    print(f'{self.PIPE}  +  Folder: {pipeline.folder}', file=self.file)
    ag_prot = pipeline.antigen.protein
    epi = pipeline.epitope
    print(f'{self.PIPE}  +  Antigen: {ag_prot.sequence()}', file=self.file)
    epi_seq = ''
    for chain in ag_prot.chains:
      if chain.name == epi.chain:
        epi_seq += (('-' * epi.start
                     + chain.sequence()[epi.start:epi.end + 1])
                    + ('-' * (len(chain.sequence()) - epi.end - 1)))
    print(f'{self.PIPE}  +  Epitope: {epi_seq}', file=self.file)

  def __print_step(self, step: PipelineStep, header: str, last: bool) -> None:
    if isinstance(step, CompositeStep):
      self.__print_composite_step(step, header, last)
    elif isinstance(step, RFdiffusionStep):
      self.__print_rfdiffusion_step(step, header, last)
    elif isinstance(step, BackboneFilteringStep):
      self.__print_backbone_filtering_step(step, header, last)
    elif isinstance(step, ProteinMPNNStep):
      self.__print_protein_mpnn_step(step, header, last)
    elif isinstance(step, OmegaFoldStep):
      self.__print_omegafold_step(step, header, last)
    elif isinstance(step, SequenceFilteringStep):
      self.__print_sequence_filtering_step(step, header, last)
    elif isinstance(step, ColabFoldStep):
      self.__print_colabfold_step(step, header, last)
    elif isinstance(step, MasifStep):
      self.__print_masif_step(step, header, last)
    elif isinstance(step, FoldCollectionStep):
      self.__print_fold_collection_step(step, header, last)
    elif isinstance(step, CARBonAraStep):
      self.__print_carbonara_step(step, header, last)
    else:
      print(header, file=self.file)

  def __print_composite_step(self, composite: CompositeStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}Composite: {composite.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}{self.PIPE}  +  Folder: {composite.get_context_folder()}',
          file=self.file)
    for i, step in enumerate(composite.steps):
      self.__print_step(step, header + (self.BLANK if last else self.PIPE), i == len(composite.steps) - 1)

  def __print_rfdiffusion_step(self, rfdiff_step: RFdiffusionStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}RFdiffusion: {rfdiff_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Docker Image: {rfdiff_step.image}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {rfdiff_step.get_context_folder()}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Number of Designs: {rfdiff_step.config.num_designs}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Partial T: {rfdiff_step.config.partial_T}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  RoG Potential R_0: {rfdiff_step.config.rog_potential}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Contact threshold: {rfdiff_step.config.contacts_threshold}',
          file=self.file)

  def __print_backbone_filtering_step(self, bbf_step: BackboneFilteringStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}Backbone Filtering: {bbf_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {bbf_step.get_context_folder()}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Secondary Structure Threshold: {bbf_step.ss_threshold}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  RoG Cutoff: '
          f'{bbf_step.rog_cutoff}{"%" if bbf_step.rog_percentage else ""}',
          file=self.file)

  def __print_protein_mpnn_step(self, pnn_step: ProteinMPNNStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}Protein MPNN: {pnn_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {pnn_step.get_context_folder()}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Docker Image: {pnn_step.image}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Sequences per target: {pnn_step.config.seqs_per_target}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Sampling Temperature: {pnn_step.config.sampling_temp}',
          file=self.file),
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Backbone Noise: {pnn_step.config.backbone_noise}',
          file=self.file),
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Batch Size: {pnn_step.config.batch_size}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Soluble Model: {pnn_step.config.use_soluble_model}',
          file=self.file)

  def __print_omegafold_step(self, of_step: OmegaFoldStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}OmegaFold: {of_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {of_step.get_context_folder()}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Docker Image: {of_step.image}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Parallel Instances: {of_step.max_parallel}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Model: {of_step.config.model_weights}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Number of Recycles: {of_step.config.num_recycles}',
          file=self.file)

  def __print_sequence_filtering_step(self, sf_step: SequenceFilteringStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}Sequence Filtering: {sf_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {sf_step.get_context_folder()}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Work on Model: {sf_step.model}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  pLDDT Threshold: {sf_step.plddt_threshold}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Number of Samples for pLDDT avg: {sf_step.average_cutoff}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  RoG cutoff: {sf_step.rog_cutoff}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Max Number of Samples: {sf_step.max_samples}',
          file=self.file)

  def __print_colabfold_step(self, cf_step: ColabFoldStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}ColabFold: {cf_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {cf_step.get_context_folder()}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Docker Image: {cf_step.image}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Parallel Instances: {cf_step.max_parallel}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Model: {cf_step.config.model_name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Number of Recycles: {cf_step.config.num_recycles}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Compress Outputs: '
          f'{"Yes" if cf_step.config.zip_outputs else "No"}', file=self.file)

  def __print_masif_step(self, masif_step: MasifStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}MaSIF: {masif_step.name}', file=self.file)

  def __print_fold_collection_step(self, fc_step: FoldCollectionStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}FoldCollection', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Model: {fc_step.model}', file=self.file)

  def __print_carbonara_step(self, cb_step: CARBonAraStep, header: str, last: bool):
    print(f'{header + (self.ELBOW if last else self.TEE)}CARBonAra: {cb_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {cb_step.get_context_folder()}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Docker Image: {cb_step.image}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Number of Sequences: {cb_step.config.num_sequences}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Imprint Ratio: {cb_step.config.imprint_ratio}',
          file=self.file),
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Ignored Amino-Acids: {cb_step.config.ignored_amino_acids}',
          file=self.file),
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Ignore Hetero-Atoms: {cb_step.config.ignore_het_atm}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Ignore Water: {cb_step.config.ignore_water}',
          file=self.file)
