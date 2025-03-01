import os.path
import shutil
import unittest
import warnings

import docker
import dotenv
import spython

from da4vid.containers.colabfold import ColabFoldContainer
from da4vid.containers.docker import DockerExecutorBuilder, DockerExecutor
from da4vid.containers.singularity import SingularityExecutorBuilder, SingularityExecutor
from da4vid.gpus.cuda import CudaDeviceManager
from test.cfg import RESOURCES_ROOT, DOTENV_FILE
from test.test_da4vid.containers.helpers import duplicate_image, remove_duplicate_image


class ColabFoldDockerContainerTest(unittest.TestCase):
  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.gpu_manager = CudaDeviceManager()
    self.client = docker.from_env()
    self.builder = DockerExecutorBuilder().set_image(ColabFoldContainer.DEFAULT_IMAGE).set_client(self.client)
    duplicate_image(self.client, 'da4vid/colabfold', 'colabfold_duplicate')
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['COLABFOLD_MODEL_FOLDER']
    self.resource_path = os.path.join(RESOURCES_ROOT, 'container_test', 'colabfold_test')
    os.makedirs(os.path.join(self.resource_path, 'single', 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(self.resource_path, 'multiple', 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(self.resource_path, 'multiple_parallel', 'outputs'), exist_ok=True)

  def tearDown(self):
    shutil.rmtree(os.path.join(self.resource_path, 'single', 'outputs'))
    shutil.rmtree(os.path.join(self.resource_path, 'multiple', 'outputs'))
    shutil.rmtree(os.path.join(self.resource_path, 'multiple_parallel', 'outputs'))
    remove_duplicate_image(self.client, 'colabfold_duplicate')
    self.client.close()

  def test_colabfold_should_raise_error_if_invalid_image(self):
    self.builder.set_image('invalid_image')
    input_dir = os.path.join(self.resource_path, 'single', 'inputs')
    output_dir = os.path.join(self.resource_path, 'single', 'outputs')
    cf = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      gpu_manager=self.gpu_manager
    )
    with self.assertRaises(DockerExecutor.DockerImageNotFoundException):
      cf.run()

  def test_colabfold_container_on_single_fasta(self):
    input_dir = os.path.join(self.resource_path, 'single', 'inputs')
    output_dir = os.path.join(self.resource_path, 'single', 'outputs')
    res = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertTrue(res, 'Container exited unsuccessfully')
    basenames = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    output_fastas = [os.path.join(output_dir, f) for f in basenames]
    self.assertEqual(1, len(output_fastas),
                     f'Invalid number of predicted fastas: {len(output_fastas)} (exp: 1)')
    predictions = [f for f in os.listdir(output_fastas[0]) if f.endswith('.pdb')]
    self.assertEqual(4, len(predictions),
                     f'Invalid number of predictions: {len(predictions)} (exp: 10)')

  def test_colabfold_should_create_prediction_on_single_fasta_with_specified_image(self):
    input_dir = os.path.join(self.resource_path, 'single', 'inputs')
    output_dir = os.path.join(self.resource_path, 'single', 'outputs')
    self.builder.set_image('colabfold_duplicate')
    res = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertTrue(res, 'Container exited unsuccessfully')
    basenames = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    output_fastas = [os.path.join(output_dir, f) for f in basenames]
    self.assertEqual(1, len(output_fastas),
                     f'Invalid number of predicted fastas: {len(output_fastas)} (exp: 1)')
    predictions = [f for f in os.listdir(output_fastas[0]) if f.endswith('.pdb')]
    self.assertEqual(4, len(predictions),
                     f'Invalid number of predictions: {len(predictions)} (exp: 10)')

  def test_colabfold_container_on_multiple_fasta(self):
    input_dir = os.path.join(self.resource_path, 'multiple', 'inputs')
    output_dir = os.path.join(self.resource_path, 'multiple', 'outputs')
    res = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      gpu_manager=self.gpu_manager
    ).run()
    expected = {'sample_1000': 4, 'sample_1001': 6, 'sample_1002': 2}
    self.assertTrue(res, 'Container exited unsuccessfully')
    basenames = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    output_fastas = [os.path.join(output_dir, f) for f in basenames]
    self.assertEqual(3, len(output_fastas),
                     f'Invalid number of predicted fastas: {len(output_fastas)} (exp: 3)')
    for basename, output_fasta in zip(basenames, output_fastas):
      predictions = [f for f in os.listdir(output_fasta) if f.endswith('.pdb')]
      self.assertEqual(expected[basename], len(predictions),
                       f'Invalid number of predictions: {len(predictions)} (exp: {expected[basename]})')

  def test_colabfold_container_on_multiple_fasta_and_parallel(self):
    input_dir = os.path.join(self.resource_path, 'multiple_parallel', 'inputs')
    output_dir = os.path.join(self.resource_path, 'multiple_parallel', 'outputs')
    res = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      max_parallel=4,
      gpu_manager=self.gpu_manager
    ).run()
    expected = {'sample_1000': 4, 'sample_1001': 6, 'sample_1002': 2,
                'sample_1003': 4, 'sample_1004': 2, 'sample_1005': 6}
    self.assertTrue(res, 'Container exited unsuccessfully')
    basenames = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    output_fastas = [os.path.join(output_dir, f) for f in basenames]
    self.assertEqual(6, len(output_fastas),
                     f'Invalid number of predicted fastas: {len(output_fastas)} (exp: 6)')
    for basename, output_fasta in zip(basenames, output_fastas):
      predictions = [f for f in os.listdir(output_fasta) if f.endswith('.pdb')]
      self.assertEqual(expected[basename], len(predictions),
                       f'Invalid number of predictions: {len(predictions)} (exp: {expected[basename]})')

  def test_should_return_false_if_container_execution_fails(self):
    input_dir = os.path.join(self.resource_path, 'single', 'inputs')
    output_dir = os.path.join(self.resource_path, 'single', 'outputs')
    res = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=-1,  # Invalid number of recycles
      gpu_manager=self.gpu_manager
    ).run()
    self.assertFalse(res, 'Colabfold container should have exited unsuccessfully')


class ColabFoldSingularityContainerTest(unittest.TestCase):

  def setUp(self):
    self.gpu_manager = CudaDeviceManager()
    self.client = spython.main.get_client()
    self.sif_path = dotenv.dotenv_values(DOTENV_FILE)['COLABFOLD_SIF']
    self.builder = SingularityExecutorBuilder().set_client(self.client).set_sif_path(self.sif_path)
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['COLABFOLD_MODEL_FOLDER']
    self.resource_path = os.path.join(RESOURCES_ROOT, 'container_test', 'colabfold_test')
    os.makedirs(os.path.join(self.resource_path, 'single', 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(self.resource_path, 'multiple', 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(self.resource_path, 'multiple_parallel', 'outputs'), exist_ok=True)

  def tearDown(self):
    shutil.rmtree(os.path.join(self.resource_path, 'single', 'outputs'))
    shutil.rmtree(os.path.join(self.resource_path, 'multiple', 'outputs'))
    shutil.rmtree(os.path.join(self.resource_path, 'multiple_parallel', 'outputs'))

  def test_colabfold_should_raise_error_if_invalid_image(self):
    self.builder.set_sif_path('invalid_sif')
    input_dir = os.path.join(self.resource_path, 'single', 'inputs')
    output_dir = os.path.join(self.resource_path, 'single', 'outputs')
    cf = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      gpu_manager=self.gpu_manager
    )
    with self.assertRaises(SingularityExecutor.SifFileNotFoundException):
      cf.run()

  def test_colabfold_container_on_single_fasta(self):
    input_dir = os.path.join(self.resource_path, 'single', 'inputs')
    output_dir = os.path.join(self.resource_path, 'single', 'outputs')
    res = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertTrue(res, 'Container exited unsuccessfully')
    basenames = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    output_fastas = [os.path.join(output_dir, f) for f in basenames]
    self.assertEqual(1, len(output_fastas),
                     f'Invalid number of predicted fastas: {len(output_fastas)} (exp: 1)')
    predictions = [f for f in os.listdir(output_fastas[0]) if f.endswith('.pdb')]
    self.assertEqual(4, len(predictions),
                     f'Invalid number of predictions: {len(predictions)} (exp: 10)')

  def test_colabfold_container_on_multiple_fasta(self):
    input_dir = os.path.join(self.resource_path, 'multiple', 'inputs')
    output_dir = os.path.join(self.resource_path, 'multiple', 'outputs')
    res = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      gpu_manager=self.gpu_manager
    ).run()
    expected = {'sample_1000': 4, 'sample_1001': 6, 'sample_1002': 2}
    self.assertTrue(res, 'Container exited unsuccessfully')
    basenames = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    output_fastas = [os.path.join(output_dir, f) for f in basenames]
    self.assertEqual(3, len(output_fastas),
                     f'Invalid number of predicted fastas: {len(output_fastas)} (exp: 3)')
    for basename, output_fasta in zip(basenames, output_fastas):
      predictions = [f for f in os.listdir(output_fasta) if f.endswith('.pdb')]
      self.assertEqual(expected[basename], len(predictions),
                       f'Invalid number of predictions: {len(predictions)} (exp: {expected[basename]})')

  def test_colabfold_container_on_multiple_fasta_and_parallel(self):
    input_dir = os.path.join(self.resource_path, 'multiple_parallel', 'inputs')
    output_dir = os.path.join(self.resource_path, 'multiple_parallel', 'outputs')
    res = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      max_parallel=4,
      gpu_manager=self.gpu_manager
    ).run()
    expected = {'sample_1000': 4, 'sample_1001': 6, 'sample_1002': 2,
                'sample_1003': 4, 'sample_1004': 2, 'sample_1005': 6}
    self.assertTrue(res, 'Container exited unsuccessfully')
    basenames = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    output_fastas = [os.path.join(output_dir, f) for f in basenames]
    self.assertEqual(6, len(output_fastas),
                     f'Invalid number of predicted fastas: {len(output_fastas)} (exp: 6)')
    for basename, output_fasta in zip(basenames, output_fastas):
      predictions = [f for f in os.listdir(output_fasta) if f.endswith('.pdb')]
      self.assertEqual(expected[basename], len(predictions),
                       f'Invalid number of predictions: {len(predictions)} (exp: {expected[basename]})')

  def test_should_return_false_if_container_execution_fails(self):
    input_dir = os.path.join(self.resource_path, 'single', 'inputs')
    output_dir = os.path.join(self.resource_path, 'single', 'outputs')
    res = ColabFoldContainer(
      builder=self.builder,
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=-1,  # Invalid number of recycles
      gpu_manager=self.gpu_manager
    ).run()
    self.assertFalse(res, 'Colabfold container should have exited unsuccessfully')
