import os.path
import shutil
import unittest
import warnings

import docker
import dotenv

from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io.sample_io import sample_set_from_fasta_folders
from da4vid.pipeline.config import StaticConfig
from da4vid.pipeline.validation import ColabFoldStep
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class ColabFoldStepTest(unittest.TestCase):

  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.image = StaticConfig.get(DOTENV_FILE).colabfold_image
    self.model_weights = StaticConfig.get(DOTENV_FILE).colabfold_models_dir
    self.resource_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'colabfold_test')

  def tearDown(self):
    self.client.close()

  def test_colabfold_predictions_presents_in_sample_set_with_af2(self):
    step_folder = os.path.join(self.resource_folder, 'shallow_test')
    backbone_folder = os.path.join(self.resource_folder, 'inputs', 'shallow', 'backbones')
    fasta_folder = os.path.join(self.resource_folder, 'inputs', 'shallow', 'fastas')
    sample_set = sample_set_from_fasta_folders(backbone_folder, fasta_folder, from_pmpnn=False)
    step = ColabFoldStep(
      name='colab_demo',
      folder=step_folder,
      client=self.client,
      model_dir=self.model_weights,
      gpu_manager=self.gpu_manager,
      image=self.image,
      config=ColabFoldStep.ColabFoldConfig(model_name='alphafold2', num_models=3),
    )
    try:
      res_set = step.execute(sample_set)
      for sample in res_set.samples():
        for sequence in sample.sequences():
          self.assertIsNotNone(sequence.get_fold_for_model('alphafold2'))
      folded_set = res_set.folded_sample_set('alphafold2')
      folded_samples = folded_set.samples()
      expected_samples = ['sample_1000_1', 'sample_1000_2',
                          'sample_1001_1', 'sample_1001_2', 'sample_1001_3']
      self.assertEqual(5, len(folded_samples))
      self.assertIn(folded_samples[0].name, expected_samples)
      self.assertIn(folded_samples[1].name, expected_samples)
      self.assertIn(folded_samples[2].name, expected_samples)
      self.assertIn(folded_samples[3].name, expected_samples)
      self.assertIn(folded_samples[4].name, expected_samples)
    finally:
      shutil.rmtree(step_folder, ignore_errors=True)
