import os.path
import shutil
import unittest

import docker
import dotenv

from da4vid.io.sample_io import sample_set_from_fasta_folders
from da4vid.pipeline.validation import OmegaFoldStep
from test.cfg import RESOURCES_ROOT, TEST_GPU, DOTENV_FILE


class OmegaFoldStepTest(unittest.TestCase):

  def setUp(self):
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['OMEGAFOLD_MODEL_FOLDER']
    self.output_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'omegafold_test', 'outputs')
    self.client = docker.from_env()

  def tearDown(self):
    shutil.rmtree(self.output_folder)
    self.client.close()

  def test_omegafold_on_single_sample(self):
    of_test_root = os.path.join(RESOURCES_ROOT, 'steps_test', 'omegafold_test')
    input_folder = os.path.join(of_test_root, 'inputs')
    sample_set = sample_set_from_fasta_folders(of_test_root, input_folder, from_pmpnn=False)
    sample_set = OmegaFoldStep(
      input_dir=input_folder,
      output_dir=self.output_folder,
      model_dir=self.model_weights,
      num_recycles=1,
      model_weights='2',
      device=TEST_GPU,
      client=self.client
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
