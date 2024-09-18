import sys

import docker
from docker.types import DeviceRequest, Mount

from da4vid.docker.rfdiffusion import RFdiffusionContainer

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

rfdiff = (RFdiffusionContainer(
  '/home/user/rfdiffusion_models',
  '/home/user/RFdiffusion/inputs',
  '/home/user/RFdiffusion/outputs',
  num_designs=1
))
status = rfdiff.run(input_pdb='example.pdb', contigs='[15-20/A25-34/15-20]', client=client)
if not status:
  print(f'Error: {status}', file=sys.stderr)
