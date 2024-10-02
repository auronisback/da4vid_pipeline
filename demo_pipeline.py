import os

import docker
import shutil

from da4vid.docker.pmpnn import ProteinMPNNContainer
from da4vid.docker.rfdiffusion import RFdiffusionContainer, RFdiffusionContigMap, RFdiffusionPotentials
from da4vid.io import read_protein_mpnn_fasta
from da4vid.io.pdb_io import read_from_pdb, read_pdb_folder
from da4vid.filters import filter_by_ss, filter_by_rog

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

os.makedirs(rfd_output_dir, exist_ok=True)  # Creating output mount point if not existing

rfdiff = RFdiffusionContainer(
  rfd_model_dir,
  rfd_input_dir,
  rfd_output_dir,
  num_designs=10,
)
protein = read_from_pdb(f'/home/user/da4vid/pipeline_demo/rfdiffusion/inputs/{protein_file}')
contig_map = RFdiffusionContigMap(protein).full_diffusion().add_provide_seq(*epitope)
potentials = RFdiffusionPotentials(guiding_scale=10).add_monomer_contacts(5).add_rog(12).linear_decay()
rfdiff.run(input_pdb=protein_name, contig_map=contig_map, potentials=potentials, client=client)

# Filtering

ss_threshold = 5
rog_cutoff = 50
rog_percentage = True

diffused = read_pdb_folder(f'{rfd_output_dir}/{protein_name}', b_fact_prop='perplexity')
filtered_ss = filter_by_ss(diffused, threshold=ss_threshold)
print(f'Filtered {len(filtered_ss)} by SS with threshold {ss_threshold}:')
for p in filtered_ss:
  print(f'  {p.name}: {p.props["ss"]}')

filtered_rog = filter_by_rog(filtered_ss, cutoff=rog_cutoff, percentage=rog_percentage)
print(f'Filtered {len(filtered_rog)} by RoG with cutoff {rog_cutoff}{"%" if rog_percentage else ""}:')
for p in filtered_rog:
  print(f'  {p.name}: {p.props["rog"].item():.3f} A')

# Moving filtered PDB files to pmpnn input directory
os.makedirs(pmpnn_input_dir, exist_ok=True)
for protein in filtered_rog:
  filename = os.path.basename(protein.filename)
  new_location = os.path.join(pmpnn_input_dir, filename)
  shutil.copy2(protein.filename, new_location)
  protein.filename = new_location  # Updating protein location

# Running ProteinMPNN
os.makedirs(pmpnn_output_dir, exist_ok=True)
pmpnn = ProteinMPNNContainer(input_dir=pmpnn_input_dir, output_dir=pmpnn_output_dir,
                             seqs_per_target=20, sampling_temp=.5, backbone_noise=.20)
pmpnn.run(client)

# Loading new proteins from FASTAs
sequenced = []
for protein in filtered_rog:
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
