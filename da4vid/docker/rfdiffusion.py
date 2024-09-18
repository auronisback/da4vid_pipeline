import os
from docker.client import DockerClient

from da4vid.docker.base import BaseContainer


class RFdiffusionContainer(BaseContainer):

  # Container local f
  SCRIPT_LOCATION = '/app/RFdiffusion/scripts/run_inference.py'
  MODELS_FOLDER = '/app/RFdiffusion/models'
  INPUT_DIR = '/app/RFdiffusion/inputs'
  OUTPUT_DIR = '/app/RFdiffusion/outputs'

  def __init__(self, model_dir, input_dir, output_dir, num_designs: int = 3):
    super().__init__(
      image='ameg/rfdiffusion',
      entrypoint='/bin/bash -c',
      with_gpus=True,
      volumes={
        model_dir: RFdiffusionContainer.MODELS_FOLDER,
        input_dir: RFdiffusionContainer.INPUT_DIR,
        output_dir: RFdiffusionContainer.OUTPUT_DIR
      },
      detach=True
    )
    self.model_dir = model_dir
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.num_designs = num_designs

  def run(self, input_pdb, contigs, client: DockerClient = None) -> bool:
    self.commands = [self.__create_command(input_pdb, contigs)]
    rfdiff = super()._run_container(client)
    for line in rfdiff.logs(stream=True):
      print(line.decode().strip())
    results = rfdiff.wait()
    return results['StatusCode'] == 0

  def __create_command(self, input_pdb, contigs) -> str:
    cmd = f'python {RFdiffusionContainer.SCRIPT_LOCATION}'
    pdb_name = os.path.basename(input_pdb).split('.')[0]
    pdb_path = f'{RFdiffusionContainer.INPUT_DIR}/{pdb_name}.pdb'
    output_prefix = f'{RFdiffusionContainer.OUTPUT_DIR}/{pdb_name}/{pdb_name}'
    args = {
      'inference.input_pdb': pdb_path,
      'inference.output_prefix': output_prefix,
      'inference.model_directory_path': RFdiffusionContainer.MODELS_FOLDER,
      'inference.num_designs': self.num_designs,
      'contigmap.contigs': contigs
    }
    return ' '.join([cmd, *[f'{key}={value}' for key, value in args.items()]])
