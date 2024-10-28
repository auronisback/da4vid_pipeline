import logging
import os
import shutil

import docker
from termcolor import colored

from da4vid.io.pdb_io import read_from_pdb
from da4vid.pipeline.config import PipelineConfig
from da4vid.pipeline.steps import RFdiffusionStep, BackboneFilteringStep, ProteinMPNNStep, OmegaFoldStep, \
  SequenceFilteringStep

# Logging properties
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[{asctime} {levelname}] {message}',
                    style='{',
                    datefmt="%Y-%m-%d %H:%M:%S")

CFG_FILE = 'pipeline.cfg.yml'

client = docker.from_env()

# Input protein properties
protein_name = 'demo_input'
protein_file = '/home/user/da4vid/pipeline_test/demo_input.pdb'
epitope = (27, 35)
epitope_chain = 'A'

# Printing epitope data in order to start
protein = read_from_pdb(protein_file)
sequence = protein.sequence()
print(f'Selected sequence and {colored("epitope", "red")}:')
print(sequence[:epitope[0]]
      + colored(sequence[epitope[0]:epitope[1] + 1], 'red', attrs=['bold'])
      + sequence[epitope[1] + 1:])

# Retrieving pipeline configuration
config = PipelineConfig.load_from_yaml(CFG_FILE)
logger.info(str(config))

# Running RFdiffusion
run1_conf = config.get_run(1)
rfd_config = run1_conf.get_rfdiffusion_configuration()
rfdiff_step = RFdiffusionStep(
  model_dir=rfd_config.model_dir,
  protein=protein,
  epitope=epitope,
  output_dir=rfd_config.output_folder(),
  num_designs=rfd_config.num_designs,
  partial_T=rfd_config.partial_T,
  contacts_threshold=rfd_config.contacts_threshold,
  rog_potential=rfd_config.rog_potential,
  client=client
)
diffused_pdbs = rfdiff_step.execute()

# First round of filtering on backbones
bf_config = run1_conf.get_backbone_filtering_configuration()
ss_rog_filter_step = BackboneFilteringStep(
  diffusions=diffused_pdbs,
  ss_threshold=bf_config.ss_threshold,
  rog_cutoff=bf_config.rog_cutoff,
  rog_percentage=bf_config.rog_percentage,
  move_to=bf_config.output_folder()
)
filtered = ss_rog_filter_step.execute()

# Protein MPNN
pmpnn_config = run1_conf.get_proteinmpnn_configuration()
pmpnn_step = ProteinMPNNStep(
  backbones=filtered,
  chain=epitope_chain,
  epitope=epitope,
  input_dir=bf_config.output_folder(),
  output_dir=pmpnn_config.output_folder(),
  seqs_per_target=pmpnn_config.seqs_per_target,
  sampling_temp=pmpnn_config.sampling_temp,
  backbone_noise=pmpnn_config.backbone_noise,
  client=client
)
pmpnn_set = pmpnn_step.execute()

for original in pmpnn_set.get_originals():
  print(original.name)
  for sample in pmpnn_set.get_samples_for(original):
    print(f'  {sample.name}')

of_config = run1_conf.get_omegafold_configuration()
omegafold_step = OmegaFoldStep(
  sample_set=pmpnn_set,
  model_dir=of_config.model_dir,
  input_dir=pmpnn_config.complete_output_folder(),
  output_dir=of_config.output_folder(),
  num_recycles=of_config.num_recycles,
  model_weights=of_config.model_weights,
  device='cuda',
  client=client
)
of_set = omegafold_step.execute()

seq_filter_config = run1_conf.get_sequence_filtering_configuration()
sequence_filtering_step = SequenceFilteringStep(
  sample_set=of_set,
  plddt_threshold=seq_filter_config.plddt_threshold,
  rog_cutoff=seq_filter_config.rog_cutoff,
  device='cuda'
)
filtered = sequence_filtering_step.execute()

# Moving filtered to run output folder
os.makedirs(run1_conf.output_folder(), exist_ok=True)
for sample in filtered.samples_list():
  sample_dest = os.path.join(run1_conf.output_folder(), os.path.basename(sample.filename))
  shutil.copy2(sample.filename, sample_dest)

exit()

# Copying filtered values to the output folder of first run
run1_outputs = run1_conf['output_folder']
os.makedirs(run1_outputs, exist_ok=True)
for protein in plddt_filtered:
  protein_basename = os.path.basename(protein.filename)
  shutil.copy2(protein.filename, f'{run1_outputs}/{protein_basename}')

# Second sequence design run

run2_conf = {
  'proteinmpnn': {
    'input_dir': run1_conf['output_folder'],
    'output_dir': '/home/user/da4vid/pipeline_test/run2/protein_mpnn/outputs',
    'seqs_per_target': 20,
    'sampling_temp': .2,
    'backbone_noise': .0
  },
  'omegafold': {
    'model_dir': '/home/user/.cache/omegafold_ckpt',
    'input_dir': '/home/user/da4vid/pipeline_test/run2/omegafold/inputs',
    'output_dir': '/home/user/da4vid/pipeline_test/run2/omegafold/outputs',
    'num_recycles': 5,
    'model_weights': "2",
    'device': 'cuda:0',
  },
  'sequence_filtering': {
    'plddt_threshold': 85
  },
  'output_dir': '/home/user/da4vid/pipeline_test/run2/outputs'
}

# Re-Running ProteinMPNN
os.makedirs(run2_conf['proteinmpnn']['input_dir'], exist_ok=True)
os.makedirs(run2_conf['proteinmpnn']['output_dir'], exist_ok=True)
print('Running ProteinMPNN on previous inputs with parameters:')
print(f'  - sequences per structure: {run2_conf["proteinmpnn"]["seqs_per_target"]}')
print(f'  - sampling temperature: {run2_conf["proteinmpnn"]["sampling_temp"]}')
print(f'  - backbone noise: {run2_conf["proteinmpnn"]["backbone_noise"]}')

pmpnn = ProteinMPNNContainer(**run2_conf['proteinmpnn'])
pmpnn.add_fixed_chain('A', [p for p in range(epitope[0], epitope[1] + 1)])
pmpnn.run(client=client)

# Copying PMPNN outputs to OmegaFold directory
os.makedirs(run2_conf['omegafold']['input_dir'], exist_ok=True)
os.makedirs(run2_conf['omegafold']['output_dir'], exist_ok=True)
for f in os.listdir(f'{run2_conf["proteinmpnn"]["output_dir"]}/seqs'):
  if f.endswith('.fa'):
    shutil.copy2(f'{run2_conf["proteinmpnn"]["output_dir"]}/seqs/{f}',
                 f'{run2_conf["omegafold"]["input_dir"]}/{f}')

print('Running OmegaFold for structure prediction')
print(f' - model weights: {run2_conf["omegafold"]["model_weights"]}')
print(f' - num_recycles: {run2_conf["omegafold"]["num_recycles"]}')

omegafold = OmegaFoldContainer(**run2_conf['omegafold'])
omegafold.run(client=client)

# Retrieving second run predictions
print('Retrieving Omegafold predictions')
run2_omegafold_renamed = '/home/user/da4vid/pipeline_test/run2/omegafold/renamed'
for d in os.listdir(run2_conf['omegafold']['output_dir']):
  # Prediction of sequences in input FASTAs are saved in a folder
  if os.path.isdir(os.path.join(run2_omegafold_renamed, d)) and d in sequenced.keys():
    orig_name = d
    orig_protein = sequenced[orig_name]['original']
    samples = read_pdb_folder(os.path.join(run2_omegafold_renamed, d), b_fact_prop='plddt')
    # Adding atom and coordinates to FASTAs
    for s in samples:
      seq = Proteins.merge_sequence_with_structure(sequenced[orig_name]['sampled'][s.name], s)

# Evaluating RMSD w.r.t. the original backbone
for s in tqdm(sequenced.values()):
  orig_protein = s['original']
  rmsd_vals, _, _ = evaluate_rmsd(orig_protein, list(s['sampled'].values()))
  for rmsd_val, p in zip(rmsd_vals, s['sampled'].values()):
    p.props['rmsd'] = rmsd_val

# Filtering by pLDDT values
plddt_threshold = run2_conf['sequence_filtering']['plddt_threshold']
plddt_filtered = filter_by_plddt([p for s in sequenced.values() for p in s['sampled'].values()],
                                 threshold=plddt_threshold)
plddt_filtered.sort(key=lambda s: s.props['rmsd'])
print(f'Filtered {len(plddt_filtered)} designs by pLDDT value >= {plddt_threshold}:')
# print([(p.name, p.filename, p.props['plddt'], p.props['rmsd']) for p in plddt_filtered])
