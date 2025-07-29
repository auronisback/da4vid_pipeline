import os.path
import shutil
import unittest
import warnings

import docker
import dotenv
import spython.main

from da4vid.containers.docker import DockerExecutorBuilder, DockerExecutor
from da4vid.containers.omegafold import OmegaFoldContainer
from da4vid.containers.singularity import SingularityExecutorBuilder, SingularityExecutor
from da4vid.gpus.cuda import CudaDeviceManager
from test.cfg import RESOURCES_ROOT, DOTENV_FILE
from test.test_da4vid.containers.helpers import duplicate_image, remove_duplicate_image


class OmegaFoldDockerContainerTest(unittest.TestCase):
  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources_path = os.path.join(RESOURCES_ROOT, 'container_test', 'omegafold_test')
    self.input_dir = os.path.join(self.resources_path, 'inputs')
    self.output_dir = os.path.join(self.resources_path, 'outputs')
    self.model_dir = dotenv.dotenv_values(DOTENV_FILE)['OMEGAFOLD_MODEL_FOLDER']
    os.makedirs(self.output_dir, exist_ok=True)
    self.gpu_manager = CudaDeviceManager()
    self.client = docker.from_env()
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(OmegaFoldContainer.DEFAULT_IMAGE)
    duplicate_image(self.client, OmegaFoldContainer.DEFAULT_IMAGE, 'of_duplicate')

  def tearDown(self):
    shutil.rmtree(self.output_dir)
    remove_duplicate_image(self.client, 'of_duplicate')
    self.client.close()

  def test_should_raise_error_if_invalid_image(self):
    self.builder.set_image('invalid_image')
    of = OmegaFoldContainer(
      builder=self.builder,
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
      gpu_manager=self.gpu_manager
    )
    with self.assertRaises(DockerExecutor.DockerImageNotFoundException):
      of.run()

  def test_should_create_predictions_with_default_image(self):
    res = OmegaFoldContainer(
      builder=self.builder,
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
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

  def test_should_create_predictions_in_parallel(self):
    res = OmegaFoldContainer(
      builder=self.builder,
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
      gpu_manager=self.gpu_manager,
      max_parallel=2
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
    self.builder.set_image('of_duplicate')
    res = OmegaFoldContainer(
      builder=self.builder,
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
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
      builder=self.builder,
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      model=-1,  # Invalid model weights
      num_recycles=-2,  # Negative number of recycles
      gpu_manager=self.gpu_manager
    ).run()
    self.assertFalse(res, 'OmegaFold container should have risen errors')


class OmegaFoldSingularityContainerTest(unittest.TestCase):

  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources_path = os.path.join(RESOURCES_ROOT, 'container_test', 'omegafold_test')
    self.input_dir = os.path.join(self.resources_path, 'inputs')
    self.output_dir = os.path.join(self.resources_path, 'outputs')
    self.model_dir = dotenv.dotenv_values(DOTENV_FILE)['OMEGAFOLD_MODEL_FOLDER']
    os.makedirs(self.output_dir, exist_ok=True)
    self.gpu_manager = CudaDeviceManager()
    self.client = spython.main.get_client()
    self.sif_path = dotenv.dotenv_values(DOTENV_FILE)['OMEGAFOLD_SIF']
    self.builder = SingularityExecutorBuilder().set_client(self.client).set_sif_path(self.sif_path)

  def tearDown(self):
    shutil.rmtree(self.output_dir)

  def test_should_raise_error_if_invalid_sif(self):
    self.builder.set_sif_path('invalid_sif')
    of = OmegaFoldContainer(
      builder=self.builder,
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
      gpu_manager=self.gpu_manager
    )
    with self.assertRaises(SingularityExecutor.SifFileNotFoundException):
      of.run()

  def test_should_create_predictions(self):
    res = OmegaFoldContainer(
      builder=self.builder,
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
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

  def test_should_create_predictions_in_parallel(self):
    res = OmegaFoldContainer(
      builder=self.builder,
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      num_recycles=2,
      gpu_manager=self.gpu_manager,
      max_parallel=2
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
      builder=self.builder,
      model_dir=self.model_dir,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      model=-1,  # Invalid model weights
      num_recycles=-2,  # Negative number of recycles
      gpu_manager=self.gpu_manager
    ).run()
    self.assertFalse(res, 'OmegaFold container should have risen errors')
