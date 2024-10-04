import os

import docker
import shutil

from da4vid.docker.pmpnn import ProteinMPNNContainer
from da4vid.docker.rfdiffusion import RFdiffusionContainer, RFdiffusionContigMap, RFdiffusionPotentials
from da4vid.io import read_protein_mpnn_fasta
from da4vid.io.pdb_io import read_from_pdb, read_pdb_folder
from da4vid.filters import filter_by_ss, filter_by_rog, cluster_by_ss

client = docker.from_env()
protein_name = 'demo_input'
protein_file = 'demo_input.pdb'
epitope = (25, 33)

# Running RFdiffusion

rfd_model_dir = '/home/user/rfdiffusion_models'
rfd_input_dir = '/home/user/da4vid/pipeline_demo/rfdiffusion/inputs'
rfd_output_dir = '/home/user/da4vid/pipeline_demo/rfdiffusion/outputs'

pmpnn_input_dir = '/home/user/da4vid/pipeline_demo/protein_mpnn/inputs'
pmpnn_output_dir = '/home/user/da4vid/pipeline_demo/protein_mpnn/outputs'
num_designs = 10

print('Starting RFdiffusion container with parameters:')
print(f' - protein PDB file: {rfd_input_dir}/{protein_file}')
print(f' - epitope interval: {epitope}')
print(f' - number of designs: {num_designs}')
os.makedirs(rfd_output_dir, exist_ok=True)  # Creating output mount point if not existing

rfdiff = RFdiffusionContainer(
  rfd_model_dir,
  rfd_input_dir,
  rfd_output_dir,
  num_designs=num_designs,
)
protein = read_from_pdb(f'/home/user/da4vid/pipeline_demo/rfdiffusion/inputs/{protein_file}')
contig_map = RFdiffusionContigMap(protein).full_diffusion().add_provide_seq(*epitope)
potentials = RFdiffusionPotentials(guiding_scale=10).add_monomer_contacts(5).add_rog(12).linear_decay()
rfdiff.run(input_pdb=protein_name, contig_map=contig_map, potentials=potentials, client=client)

# Filtering

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

# Moving filtered PDB files to pmpnn input directory
os.makedirs(pmpnn_input_dir, exist_ok=True)
for protein in backbones:
  filename = os.path.basename(protein.filename)
  new_location = os.path.join(pmpnn_input_dir, filename)
  shutil.copy2(protein.filename, new_location)
  protein.filename = new_location  # Updating protein location

# Running ProteinMPNN
seqs_per_target = 20
sampling_temp = .5
backbone_noise = .20

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
sequenced = []
for protein in backbones:
  filename = ''.join(os.path.basename(protein.filename).split('.')[:-1]) + '.fa'
  sampled = read_protein_mpnn_fasta(f'{pmpnn_output_dir}/seqs/{filename}')
  # Adding props to original protein
  protein.props['protein_mpnn'] = sampled[0].props['protein_mpnn']
  sequenced.append({
    'original': protein,
    'sampled': sampled[1:]
  })
for p in sequenced:
  print(p['original'].props, [s.props for s in p['sampled']])
