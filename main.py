import sys

import docker
from docker.types import DeviceRequest, Mount

from da4vid.docker.rfdiffusion import RFdiffusionContainer, RFdiffusionContigMap, RFdiffusionPotentials
from da4vid.filters import filter_by_rog, filter_by_ss
from da4vid.utils.io import read_pdb_folder, read_from_pdb

client = docker.from_env()
# container = client.containers.run('ameg/rfdiffusion', [
#     'python /app/RFdiffusion/scripts/run_inference.py'
#   ],
#   device_requests=[
#     DeviceRequest(capabilities=[['gpu']]),
#   ],
#   mounts=[
#     Mount('/app/RFdiffusion/models', '/home/user/rfdiffusion_models', type='bind'),
#     Mount('/app/RFdiffusion/inputs', '/home/user/RFdiffusion/inputs', type='bind'),
#     Mount('/app/RFdiffusion/outputs', '/home/user/RFdiffusion/outputs', type='bind')
#   ],
#   detach=True,
#   entrypoint='/bin/bash -c',
#   remove=True
# )
#
# rfdiff = (RFdiffusionContainer(
#   '/home/user/rfdiffusion_models',
#   '/home/user/RFdiffusion/inputs',
#   '/home/user/RFdiffusion/outputs',
#   num_designs=10
# ))
# status = rfdiff.run(input_pdb='example.pdb', contigs='[15-20/A25-34/15-20]', client=client)
# if not status:
#   print(f'Error: {status}', file=sys.stderr)
#
# diffusion_folder = '/home/user/RFdiffusion/outputs/example'
# proteins = read_pdb_folder(diffusion_folder)
# rog_filtered = filter_by_rog(proteins, cutoff=50, percentage=True, device='cuda:0')
# print([(p.name, p.props['rog'], p.coords().shape) for p in proteins])
# ss_filtered = filter_by_ss(rog_filtered, threshold=3)
# print([(p.name, p.props['rog'], p.props['ss']) for p in ss_filtered])

rfdiff = RFdiffusionContainer(
  '/home/user/rfdiffusion_models',
  '/home/user/RFdiffusion/inputs',
  '/home/user/RFdiffusion/outputs',
  num_designs=1
)
protein = read_from_pdb('/home/user/RFdiffusion/inputs/example.pdb')
# contig_map = RFdiffusionContigMap(protein).add_random_length_sequence(15, 20)\
#   .add_fixed_sequence('A', 25, 34).add_random_length_sequence(15, 20)
contig_map = RFdiffusionContigMap(protein).full_diffusion().add_provide_seq(14, 19)
potentials = RFdiffusionPotentials().add_monomer_contacts(0.5).add_rog(12)
rfdiff.run(input_pdb='example.pdb', contig_map=contig_map, potentials=potentials)
