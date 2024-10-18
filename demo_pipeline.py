import os
import shutil
from termcolor import colored

import docker

from da4vid.docker.omegafold import OmegaFoldContainer
from da4vid.docker.pmpnn import ProteinMPNNContainer
from da4vid.docker.rfdiffusion import RFdiffusionContainer, RFdiffusionContigMap, RFdiffusionPotentials
from da4vid.filters import filter_by_rog, cluster_by_ss, filter_by_plddt
from da4vid.io import read_protein_mpnn_fasta
from da4vid.io.pdb_io import read_from_pdb, read_pdb_folder
from da4vid.metrics import evaluate_rmsd
from da4vid.model import Proteins

client = docker.from_env()

# Input protein properties
protein_name = 'demo_input'
protein_file = '/home/user/da4vid/pipeline_demo/demo_input.pdb'
epitope = (27, 35)

# Printing epitope data in order to start
protein = read_from_pdb(protein_file)
sequence = protein.sequence()
print(f'Selected sequence and {colored("epitope", "red")}:')
print(sequence[:epitope[0]]
      + colored(sequence[epitope[0]:epitope[1] + 1], 'red', attrs=['bold'])
      + sequence[epitope[1] + 1:])

# First run
run1_conf = {
  'rfdiffusion': {
    'model_dir': '/home/user/rfdiffusion_models',
    'input_dir': '/home/user/da4vid/pipeline_demo/run1/rfdiffusion/inputs',
    'output_dir': '/home/user/da4vid/pipeline_demo/run1/rfdiffusion/outputs',
    'num_designs': 2000,
    'partial_T': 23
  },
  'backbone_filtering': {
    'ss_threshold': 5,
    'rog_cutoff': 10,
    'rog_percentage': False
  },
  'proteinmpnn': {
    'input_dir': '/home/user/da4vid/pipeline_demo/run1/protein_mpnn/inputs',
    'output_dir': '/home/user/da4vid/pipeline_demo/run1/protein_mpnn/outputs',
    'seqs_per_target': 2000,
    'sampling_temp': .5,
    'backbone_noise': .20
  },
  'omegafold': {
    'model_dir': '/home/user/.cache/omegafold_ckpt',
    'input_dir': '/home/user/da4vid/pipeline_demo/run1/omegafold/inputs',
    'output_dir': '/home/user/da4vid/pipeline_demo/run1/omegafold/outputs',
    'num_recycles': 5,
    'model_weights': "2",
    'device': 'cuda:0',
  },
  'sequence_filtering': {
    'plddt_threshold': 70.
  },
  'output_folder': '/home/user/da4vid/pipeline_demo/run1/outputs'
}

# Running RFdiffusion
os.makedirs(run1_conf['rfdiffusion']['input_dir'], exist_ok=True)
os.makedirs(run1_conf['rfdiffusion']['output_dir'], exist_ok=True)
rfd_input_protein = f"{run1_conf['rfdiffusion']['input_dir']}/{os.path.basename(protein_file)}"
shutil.copy2(protein_file, rfd_input_protein)

print('Starting RFdiffusion container with parameters:')
print(f' - protein PDB file: {rfd_input_protein}')
print(f' - epitope interval: {epitope}')
print(f' - number of designs: {run1_conf["rfdiffusion"]["num_designs"]}')

rfdiff = RFdiffusionContainer(**run1_conf['rfdiffusion'])

contig_map = RFdiffusionContigMap(protein).full_diffusion().add_provide_seq(*epitope)
potentials = RFdiffusionPotentials(guiding_scale=10).add_monomer_contacts(5).add_rog(12).linear_decay()
# rfdiff.run(input_pdb=protein_name, contig_map=contig_map, potentials=potentials, client=client)

# First round of filtering on backbones
print('Filtering generated backbones by SS and RoG')

ss_threshold = run1_conf['backbone_filtering']['ss_threshold']
rog_cutoff = run1_conf['backbone_filtering']['rog_cutoff']
rog_percentage = run1_conf['backbone_filtering']['rog_percentage']

diffused = read_pdb_folder(f'{run1_conf["rfdiffusion"]["output_dir"]}/{protein_name}', b_fact_prop='perplexity')
clustered_ss = cluster_by_ss(diffused, threshold=ss_threshold)
print(f'Found {sum([len(v) for v in clustered_ss.values()])} proteins with SS number >= {ss_threshold}:')
print('  SS: number ')
for k in clustered_ss.keys():
  print(f'  {k}: {len(clustered_ss[k])}')
# Retaining the 10 smallest proteins for each cluster by filtering via RoG (decreasing)
backbones = []
for k in clustered_ss.keys():
  backbones += filter_by_rog(clustered_ss[k], cutoff=rog_cutoff, percentage=rog_percentage)
print(f'Filtered {len(backbones)} proteins by RoG with cutoff {rog_cutoff}{"%" if rog_percentage else ""}:')
for p in backbones:
  print(f'  {p.name}: {p.props["rog"].item():.3f} A')

# Moving filtered PDB files to PMPNN input directory
os.makedirs(run1_conf['proteinmpnn']['input_dir'], exist_ok=True)
for protein in backbones:
  filename = os.path.basename(protein.filename)
  new_location = os.path.join(run1_conf['proteinmpnn']['input_dir'], filename)
  shutil.copy2(protein.filename, new_location)
  protein.filename = new_location  # Updating protein location

# Running ProteinMPNN
print('Running ProteinMPNN on filtered backbones with parameters:')
print(f'  - sequences per structure: {run1_conf["proteinmpnn"]["seqs_per_target"]}')
print(f'  - sampling temperature: {run1_conf["proteinmpnn"]["sampling_temp"]}')
print(f'  - backbone noise: {run1_conf["proteinmpnn"]["backbone_noise"]}')
os.makedirs(run1_conf['proteinmpnn']['output_dir'], exist_ok=True)
pmpnn = ProteinMPNNContainer(**run1_conf['proteinmpnn'])
pmpnn.add_fixed_chain('A', [p for p in range(epitope[0], epitope[1] + 1)])
# pmpnn.run(client)

# Loading new proteins from FASTAs
sequenced = {}
for protein in backbones:
  filename = ''.join(os.path.basename(protein.filename).split('.')[:-1]) + '.fa'
  sampled = read_protein_mpnn_fasta(f'{run1_conf["proteinmpnn"]["output_dir"]}/seqs/{filename}')
  # Adding props to original protein
  protein.props['protein_mpnn'] = sampled[0].props['protein_mpnn']
  sequenced[protein.name] = {
    'original': protein,
    'sampled': {s.name: s for s in sampled[1:]}
  }

  # Copying PMPNN outputs to OmegaFold directory

os.makedirs(run1_conf['omegafold']['input_dir'], exist_ok=True)
os.makedirs(run1_conf['omegafold']['output_dir'], exist_ok=True)
for f in os.listdir(f'{run1_conf["proteinmpnn"]["output_dir"]}/seqs'):
  if f.endswith('.fa'):
    shutil.copy2(f'{run1_conf["proteinmpnn"]["output_dir"]}/seqs/{f}',
                 f'{run1_conf["omegafold"]["input_dir"]}/{f}')

print('Running OmegaFold for structure prediction')
print(f' - model weights: {run1_conf["omegafold"]["model_weights"]}')
print(f' - num_recycles: {run1_conf["omegafold"]["num_recycles"]}')

omegafold = OmegaFoldContainer(**run1_conf['omegafold'])
#omegafold.run(client=client)

# Renaming OmegaFold outputs in another directory
run1_omegafold_renamed = '/home/user/da4vid/pipeline_demo/run1/omegafold/renamed'
os.makedirs(run1_omegafold_renamed, exist_ok=True)
for d in os.listdir(run1_conf['omegafold']['output_dir']):
  full_d = os.path.join(run1_conf['omegafold']['output_dir'], d)
  if os.path.isdir(full_d):
    orig_name = os.path.basename(d)
    dest_folder = os.path.join(run1_omegafold_renamed, orig_name)
    os.makedirs(dest_folder, exist_ok=True)
    for f in os.listdir(full_d):
      if f.endswith('.pdb') and f.split(',')[0].strip() != orig_name:
        src_name = os.path.join(full_d, f)
        sample_num = f.split(',')[1].split('=')[1].strip()
        dest_name = os.path.join(run1_omegafold_renamed, orig_name, f'{orig_name}_{sample_num}.pdb')
        shutil.copy2(src_name, dest_name)

# Retrieving first run predictions
print('Retrieving Omegafold predictions')
for d in os.listdir(run1_conf['omegafold']['output_dir']):
  # Prediction of sequences in input FASTAs are saved in a folder
  if os.path.isdir(os.path.join(run1_omegafold_renamed, d)) and d in sequenced.keys():
    orig_name = d
    orig_protein = sequenced[orig_name]['original']
    samples = read_pdb_folder(os.path.join(run1_omegafold_renamed, d), b_fact_prop='plddt')
    # Adding atom and coordinates to FASTAs
    for s in samples:
      seq = Proteins.merge_sequence_with_structure(sequenced[orig_name]['sampled'][s.name], s)

# Evaluating RMSD w.r.t. the original backbone
for s in sequenced.values():
  orig_protein = s['original']
  rmsd_vals, _, _ = evaluate_rmsd(orig_protein, list(s['sampled'].values()))
  for rmsd_val, p in zip(rmsd_vals, s['sampled'].values()):
    p.props['rmsd'] = rmsd_val

# Filtering by pLDDT values
plddt_threshold = run1_conf['sequence_filtering']['plddt_threshold']
plddt_filtered = filter_by_plddt([p for s in sequenced.values() for p in s['sampled'].values()],
                                 threshold=plddt_threshold)
plddt_filtered.sort(key=lambda s: s.props['rmsd'])
print(f'Filtered {len(plddt_filtered)} designs by pLDDT value >= {plddt_threshold}:')
print([(p.name, p.filename, p.props['plddt'], p.props['rmsd']) for p in plddt_filtered])

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
    'output_dir': '/home/user/da4vid/pipeline_demo/run2/protein_mpnn/outputs',
    'seqs_per_target': 20,
    'sampling_temp': .2,
    'backbone_noise': .0
  },
  'omegafold': {
    'model_dir': '/home/user/.cache/omegafold_ckpt',
    'input_dir': '/home/user/da4vid/pipeline_demo/run2/omegafold/inputs',
    'output_dir': '/home/user/da4vid/pipeline_demo/run2/omegafold/outputs',
    'num_recycles': 5,
    'model_weights': "2",
    'device': 'cuda:0',
  },
  'sequence_filtering': {
    'plddt_threshold': 85
  },
  'output_dir': '/home/user/da4vid/pipeline_demo/run2/outputs'
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
run2_omegafold_renamed = '/home/user/da4vid/pipeline_demo/run2/omegafold/renamed'
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
for s in sequenced.values():
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
print([(p.name, p.filename, p.props['plddt'], p.props['rmsd']) for p in plddt_filtered])
