import sys

from da4vid.pipeline.config import StaticConfig
from da4vid.pipeline.generation import RFdiffusionStep, BackboneFilteringStep, ProteinMPNNStep, CARBonAraStep
from da4vid.pipeline.interaction import MasifStep, PestoStep, InteractionWindowEvaluationStep
from da4vid.pipeline.steps import PipelineRootStep, PipelineStep, CompositeStep, FoldCollectionStep
from da4vid.pipeline.validation import OmegaFoldStep, SequenceFilteringStep, ColabFoldStep


class PipelinePrinter:
  ELBOW = "└──"
  PIPE = "│  "
  TEE = "├──"
  BLANK = "   "

  def __init__(self, file=sys.stdout):
    self.file = file

  def print(self, pipeline: PipelineRootStep) -> None:
    print(f'backend: {StaticConfig.get().backend()}')
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
        epi_seq += (('-' * (epi.start - 1)
                     + chain.sequence()[epi.start-1:epi.end])
                    + ('-' * (len(chain.sequence()) - epi.end)))
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
    elif isinstance(step, PestoStep):
      self.__print_pesto_step(step, header, last)
    elif isinstance(step, InteractionWindowEvaluationStep):
      self.__print_interaction_window_step(step, header, last)
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
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Max Number of folds per sample: '
          f'{sf_step.max_folds_per_sample if sf_step.max_folds_per_sample else "no limit"}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Max Number of Samples: '
          f'{sf_step.max_samples if sf_step.max_samples else "no limit"}', file=self.file)

  def __print_colabfold_step(self, cf_step: ColabFoldStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}ColabFold: {cf_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {cf_step.get_context_folder()}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Parallel Instances: {cf_step.max_parallel}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Model: {cf_step.config.model_name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Number of Recycles: {cf_step.config.num_recycles}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Compress Outputs: '
          f'{"Yes" if cf_step.config.zip_outputs else "No"}', file=self.file)

  def __print_masif_step(self, masif_step: MasifStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}MaSIF: {masif_step.name}', file=self.file)

  def __print_pesto_step(self, pesto_step: PestoStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}PeSTo: {pesto_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {pesto_step.get_context_folder()}', file=self.file)

  def __print_fold_collection_step(self, fc_step: FoldCollectionStep, header: str, last: bool) -> None:
    print(f'{header + (self.ELBOW if last else self.TEE)}FoldCollection', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Model: {fc_step.model}', file=self.file)

  def __print_carbonara_step(self, cb_step: CARBonAraStep, header: str, last: bool):
    print(f'{header + (self.ELBOW if last else self.TEE)}CARBonAra: {cb_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {cb_step.get_context_folder()}', file=self.file)
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

  def __print_interaction_window_step(self, iw_step: InteractionWindowEvaluationStep, header: str, last: bool):
    print(f'{header + (self.ELBOW if last else self.TEE)}CARBonAra: {iw_step.name}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Folder: {iw_step.get_context_folder()}', file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Offset: {iw_step.offset}',
          file=self.file)
    print(f'{header}{self.BLANK if last else self.PIPE}  +  Interaction Key: {iw_step.interaction_key}',
          file=self.file)
