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
epitope = (26, 34)

# Printing epitope data in order to start
protein = read_from_pdb(protein_file)
sequence = protein.sequence()
print(f'Selected sequence and {colored("epitope", "red")}:')
print(sequence[:epitope[0]]
      + colored(sequence[epitope[0]:epitope[1]+1], 'red', attrs=['bold'])
      + sequence[epitope[1]+1:])


# Running RFdiffusion
rfd_model_dir = '/home/user/rfdiffusion_models'
rfd_input_dir = '/home/user/da4vid/pipeline_demo/run1/rfdiffusion/inputs'
rfd_output_dir = '/home/user/da4vid/pipeline_demo/run1/rfdiffusion/outputs'

rfd_num_designs = 10
rfd_timesteps = 23

os.makedirs(rfd_input_dir, exist_ok=True)
os.makedirs(rfd_output_dir, exist_ok=True)
rfd_input_protein = f'{rfd_input_dir}/{os.path.basename(protein_file)}'
shutil.copy2(protein_file, rfd_input_protein)

print('Starting RFdiffusion container with parameters:')
print(f' - protein PDB file: {rfd_input_protein}')
print(f' - epitope interval: {epitope}')
print(f' - number of designs: {rfd_num_designs}')
os.makedirs(rfd_output_dir, exist_ok=True)  # Creating output mount point if not existing

rfdiff = RFdiffusionContainer(
  rfd_model_dir,
  rfd_input_dir,
  rfd_output_dir,
  num_designs=rfd_num_designs
)
contig_map = RFdiffusionContigMap(protein).full_diffusion().add_provide_seq(*epitope)
potentials = RFdiffusionPotentials(guiding_scale=10).add_monomer_contacts(5).add_rog(12).linear_decay()
rfdiff.run(input_pdb=protein_name, contig_map=contig_map, potentials=potentials, partial_T=rfd_timesteps, client=client)


# First round of filtering on backbones
print('Filtering generated backbones by SS and RoG')

ss_threshold = 5
rog_cutoff = 10
rog_percentage = False

diffused = read_pdb_folder(f'{rfd_output_dir}/{protein_name}', b_fact_prop='perplexity')
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


# PMPNN configuration
pmpnn_input_dir = '/home/user/da4vid/pipeline_demo/run1/protein_mpnn/inputs'
pmpnn_output_dir = '/home/user/da4vid/pipeline_demo/run1/protein_mpnn/outputs'

seqs_per_target = 20
sampling_temp = .5
backbone_noise = .20

# Moving filtered PDB files to PMPNN input directory
os.makedirs(pmpnn_input_dir, exist_ok=True)
for protein in backbones:
  filename = os.path.basename(protein.filename)
  new_location = os.path.join(pmpnn_input_dir, filename)
  shutil.copy2(protein.filename, new_location)
  protein.filename = new_location  # Updating protein location

# Running ProteinMPNN
print('Running ProteinMPNN on filtered backbones with parameters:')
print(f'  - sequences per structure: {seqs_per_target}')
print(f'  - sampling temperature: {sampling_temp}')
print(f'  - backbone noise: {backbone_noise}')
os.makedirs(pmpnn_output_dir, exist_ok=True)
pmpnn = ProteinMPNNContainer(input_dir=pmpnn_input_dir, output_dir=pmpnn_output_dir,
                             seqs_per_target=seqs_per_target, sampling_temp=sampling_temp,
                             backbone_noise=backbone_noise)
pmpnn.run(client)

# Loading new proteins from FASTAs
sequenced = {}
for protein in backbones:
  filename = ''.join(os.path.basename(protein.filename).split('.')[:-1]) + '.fa'
  sampled = read_protein_mpnn_fasta(f'{pmpnn_output_dir}/seqs/{filename}')
  # Adding props to original protein
  protein.props['protein_mpnn'] = sampled[0].props['protein_mpnn']
  sequenced[protein.name] = {
    'original': protein,
    'sampled': {s.name: s for s in sampled[1:]}
  }

# Copying fasta outputs into omegafold input folder
print('Running OmegaFold for structure prediction')

omegafold_models = '/home/user/.cache/omegafold_ckpt'
omegafold_inputs = '/home/user/da4vid/pipeline_demo/run1/omegafold/inputs'
omegafold_outputs = '/home/user/da4vid/pipeline_demo/run1/omegafold/outputs'

# Copying PMPNN outputs to OmegaFold directory
os.makedirs(omegafold_inputs, exist_ok=True)
os.makedirs(omegafold_outputs, exist_ok=True)
for f in os.listdir(f'{pmpnn_output_dir}/seqs'):
  if f.endswith('.fa'):
    shutil.copy2(f'{pmpnn_output_dir}/seqs/{f}',
                 f'{omegafold_inputs}/{f}')

omegafold_recycles = 5
omegafold_running_model = "2"
omegafold_device = 'cuda:0'

omegafold = OmegaFoldContainer(
  model_dir=omegafold_models,
  input_dir=omegafold_inputs,
  output_dir=omegafold_outputs,
  running_model=omegafold_running_model
)
omegafold.run(num_cycle=omegafold_recycles, device=omegafold_device, client=client)

# Renaming OmegaFold outputs
run1_outputs = '/home/user/da4vid/pipeline_demo/run1/outputs'
os.makedirs(run1_outputs, exist_ok=True)

for d in os.listdir(omegafold_outputs):
  full_d = os.path.join(omegafold_outputs, d)
  if os.path.isdir(full_d):
    orig_name = os.path.basename(d)
    dest_folder = os.path.join(run1_outputs, orig_name)
    os.makedirs(dest_folder, exist_ok=True)
    for f in os.listdir(full_d):
      if f.endswith('.pdb') and f.split(',')[0].strip() != orig_name:
        src_name = os.path.join(full_d, f)
        sample_num = f.split(',')[1].split('=')[1].strip()
        dest_name = os.path.join(run1_outputs, orig_name, f'{orig_name}_{sample_num}.pdb')
        shutil.copy2(src_name, dest_name)


# Retrieving first run predictions
print('Retrieving Omegafold predictions')
for d in os.listdir(omegafold_outputs):
  # Prediction of sequences in input FASTAs are saved in a folder
  if os.path.isdir(os.path.join(run1_outputs, d)) and d in sequenced.keys():
    orig_name = d
    orig_protein = sequenced[orig_name]['original']
    samples = read_pdb_folder(os.path.join(run1_outputs, d), b_fact_prop='plddt')
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
plddt_filtered = filter_by_plddt([p for s in sequenced.values() for p in s['sampled'].values()], threshold=70)
plddt_filtered.sort(key=lambda s: s.props['rmsd'])
print(f'Filtered {len(plddt_filtered)} designs by pLDDT value >= 70:')
print([(p.name, p.props['plddt'], p.props['rmsd']) for p in plddt_filtered])
