import os.path
import shutil
import unittest
import warnings

import docker
import dotenv

from da4vid.docker.base import BaseContainer
from da4vid.docker.omegafold import OmegaFoldContainer
from da4vid.gpus.cuda import CudaDeviceManager
from test.cfg import RESOURCES_ROOT, DOTENV_FILE
from test.test_da4vid.docker.helpers import duplicate_image, remove_duplicate_image


class OmegaFoldContainerTest(unittest.TestCase):
  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources_path = os.path.join(RESOURCES_ROOT, 'docker_test', 'omegafold_test')
    self.input_dir = os.path.join(self.resources_path, 'inputs')
    self.output_dir = os.path.join(self.resources_path, 'outputs')
    self.model_dir = dotenv.dotenv_values(DOTENV_FILE)['OMEGAFOLD_MODEL_FOLDER']
    os.makedirs(self.output_dir, exist_ok=True)
    self.gpu_manager = CudaDeviceManager()
    self.client = docker.from_env()
    duplicate_image(self.client, 'da4vid/omegafold', 'of_duplicate')

  def tearDown(self):
    shutil.rmtree(self.output_dir)
    remove_duplicate_image(self.client, 'of_duplicate')
    self.client.close()

  def test_should_raise_error_if_invalid_image(self):
    of = OmegaFoldContainer(
      image='invalid_image',
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
      client=self.client,
      gpu_manager=self.gpu_manager
    )
    with self.assertRaises(BaseContainer.DockerImageNotFoundException):
      of.run()

  def test_should_create_predictions_with_default_image(self):
    res = OmegaFoldContainer(
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
      client=self.client,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertTrue(res, 'OmegaFold container stopped with errors!')
    folds = [f for f in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, f))]
    self.assertIn('base_1', folds)
    self.assertIn('base_2', folds)
    base_1_folds = [f for f in os.listdir(os.path.join(self.output_dir, 'base_1')) if f.endswith('.pdb')]
    self.assertEqual(3, len(base_1_folds))
    self.assertIn('Base.pdb', base_1_folds)
    self.assertIn('Fold_1.pdb', base_1_folds)
    self.assertIn('Fold_2.pdb', base_1_folds)
    base_2_folds = [f for f in os.listdir(os.path.join(self.output_dir, 'base_2')) if f.endswith('.pdb')]
    self.assertEqual(2, len(base_2_folds))
    self.assertIn('Base.pdb', base_2_folds)
    self.assertIn('Fold_1.pdb', base_2_folds)

  def test_should_create_predictions_with_specified_image(self):
    res = OmegaFoldContainer(
      image='of_duplicate',
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
      client=self.client,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertTrue(res, 'OmegaFold container stopped with errors!')
    folds = [f for f in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, f))]
    self.assertIn('base_1', folds)
    self.assertIn('base_2', folds)
    base_1_folds = [f for f in os.listdir(os.path.join(self.output_dir, 'base_1')) if f.endswith('.pdb')]
    self.assertEqual(3, len(base_1_folds))
    self.assertIn('Base.pdb', base_1_folds)
    self.assertIn('Fold_1.pdb', base_1_folds)
    self.assertIn('Fold_2.pdb', base_1_folds)
    base_2_folds = [f for f in os.listdir(os.path.join(self.output_dir, 'base_2')) if f.endswith('.pdb')]
    self.assertEqual(2, len(base_2_folds))
    self.assertIn('Base.pdb', base_2_folds)
    self.assertIn('Fold_1.pdb', base_2_folds)

  def test_should_return_false_if_container_execution_fails(self):
    res = OmegaFoldContainer(
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      model_weights='invalid',  # Invalid model weights
      num_recycles=-2,  # Negative number of recycles
      client=self.client,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertFalse(res, 'OmegaFold container should have risen errors')

