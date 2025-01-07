import os.path
import shutil
import unittest
import warnings

import docker
import dotenv

from da4vid.docker.omegafold import OmegaFoldContainer
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io.sample_io import sample_set_from_fasta_folders
from da4vid.pipeline.validation import OmegaFoldStep
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class OmegaFoldStepTest(unittest.TestCase):

  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources = os.path.join(RESOURCES_ROOT, 'steps_test', 'omegafold_test')
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['OMEGAFOLD_MODEL_FOLDER']
    self.output_folder = os.path.join(self.resources, 'outputs')
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()

  def tearDown(self):
    shutil.rmtree(self.output_folder)
    self.client.close()

  def test_omegafold_on_single_sample(self):
    input_folder = os.path.join(self.resources, 'inputs')
    sample_set = sample_set_from_fasta_folders(self.resources, input_folder, from_pmpnn=False)
    config = OmegaFoldStep.OmegaFoldConfig(
      num_recycles=1,
      model_weights='2',
    )
    sample_set = OmegaFoldStep(
      name='OF_STEP',
      folder=self.resources,
      image=OmegaFoldContainer.DEFAULT_IMAGE,
      model_dir=self.model_weights,
      config=config,
      client=self.client,
      gpu_manager=self.gpu_manager
    ).execute(sample_set)
    pdb_list = [d for d in os.listdir(self.output_folder)
                if os.path.isdir(os.path.join(self.output_folder, d))]
    self.assertEqual(1, len(pdb_list),
                     f'Invalid number of predicted folders: {len(pdb_list)} (exp 1)')
    self.assertEqual(1, len(sample_set.samples()))
    sample = sample_set.samples()[0]
    sequences = sample.sequences()
    self.assertEqual(5, len(sequences), f'Invalid number of samples: {len(sequences)} (exp 5)')
    for seq in sequences:
      self.assertEqual(1, len(seq.folds()))
      self.assertIsNotNone(seq.get_fold_for_model('omegafold'))
