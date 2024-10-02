import docker

from da4vid.docker.rfdiffusion import RFdiffusionContainer, RFdiffusionContigMap, RFdiffusionPotentials
from da4vid.utils.io import read_from_pdb

client = docker.from_env()
protein_name = 'demo_input.pdb'

rfdiff = RFdiffusionContainer(
  '/home/user/rfdiffusion_models',
  '/home/user/da4vid/pipeline_demo/rfdiffusion/inputs',
  '/home/user/da4vid/pipeline_demo/rfdiffusion/outputs',
  num_designs=1
)
protein = read_from_pdb(f'/home/user/da4vid/pipeline_demo/rfdiffusion/inputs/{protein_name}')
contig_map = RFdiffusionContigMap(protein).full_diffusion().add_provide_seq(25, 33)
potentials = RFdiffusionPotentials(guiding_scale=10).add_monomer_contacts(5).add_rog(12).linear_decay()
rfdiff.run(input_pdb=protein_name, contig_map=contig_map, potentials=potentials, client=client)
