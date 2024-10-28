from typing import List, Union

from da4vid.docker.base import BaseContainer


class ColabFoldContainer(BaseContainer):
  MODELS_FOLDER = '/localcolabfold/colabfold/weights'
  INPUT_DIR = '/localcolabfold/colabfold/inputs'
  OUTPUT_DIR = '/localcolabfold/colabfold/outputs'

  COLABFOLD_API_URL = 'https://api.colabfold.com'
  MODEL_NAMES = ['auto', 'alphafold2', 'alphafold2_ptm,alphafold2_multimer_v1', 'alphafold2_multimer_v2',
                 'alphafold2_multimer_v3', 'deepfold_v1']

  def __init__(self, model_dir: str, input_dir: str, output_dir: str,
               num_recycle: int = 5, zip_outputs: bool = False,
               model_name: str = MODEL_NAMES[0], num_models: int = 5,
               msa_host_url: str = COLABFOLD_API_URL):
    super().__init__(
      image='ameg/colabfold:latest',
      entrypoint='/bin/bash',
      with_gpus=True,
      volumes={
        model_dir: ColabFoldContainer.MODELS_FOLDER,
        input_dir: ColabFoldContainer.INPUT_DIR,
        output_dir: ColabFoldContainer.OUTPUT_DIR
      },
      detach=True
    )
    self.num_recycle = num_recycle
    self.zip_outputs = zip_outputs
    # Checking valid model
    if model_name not in ColabFoldContainer.MODEL_NAMES:
      raise ValueError(f'given model "{model_name}" is invalid '
                       f'(choices: {", ".join(ColabFoldContainer.MODEL_NAMES)})')
    self.model_name = model_name
    self.num_models = num_models
    # Initializing list of MSA endpoint URLs
    self.msa_host_url = msa_host_url

  def run(self):
    pass
