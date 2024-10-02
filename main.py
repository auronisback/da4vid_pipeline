import docker
from docker.models.resource import Model

from da4vid.docker.pmpnn import ProteinMPNNContainer

input_dir = '/home/user/da4vid/pipeline_demo/protein_mpnn/inputs'
output_dir = '/home/user/da4vid/pipeline_demo/protein_mpnn/outputs'

pmpnn = ProteinMPNNContainer(
  input_dir=input_dir,
  output_dir=output_dir,
  seqs_per_target=10,
  sampling_temp=0.5,
  backbone_noise=0.5
)
pmpnn.add_fixed_chain('A', list(range(25, 34)))
pmpnn.run()

# client = docker.from_env()
# cont = client.containers.run(
#   image='alpine:latest',
#   command='/bin/sh',
#   detach=True,
#   tty=True
# )
# _, out = cont.exec_run('echo 1', stream=True)
# for line in out:
#   print(line)
# cont.exec_run('echo 2')
# cont.stop()
# cont.remove()

