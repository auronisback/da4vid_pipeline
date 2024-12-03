import os.path
import shutil
import unittest

import docker
import dotenv

from da4vid.docker.colabfold import ColabFoldContainer
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class ColabFoldContainerTest(unittest.TestCase):
  def setUp(self):
    self.client = docker.from_env()
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['COLABFOLD_MODEL_FOLDER']
    self.resource_path = os.path.join(RESOURCES_ROOT, 'docker_test', 'colabfold_test')
    os.makedirs(os.path.join(self.resource_path, 'single', 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(self.resource_path, 'multiple', 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(self.resource_path, 'multiple_parallel', 'outputs'), exist_ok=True)

  def tearDown(self):
    shutil.rmtree(os.path.join(self.resource_path, 'single', 'outputs'))
    shutil.rmtree(os.path.join(self.resource_path, 'multiple', 'outputs'))
    shutil.rmtree(os.path.join(self.resource_path, 'multiple_parallel', 'outputs'))
    self.client.close()

  def test_colabfold_container_on_single_fasta(self):
    input_dir = os.path.join(self.resource_path, 'single', 'inputs')
    output_dir = os.path.join(self.resource_path, 'single', 'outputs')
    res = ColabFoldContainer(
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
    ).run(client=self.client)
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
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1
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
      model_dir=self.model_weights,
      input_dir=input_dir,
      output_dir=output_dir,
      num_models=2,
      num_recycle=1,
      max_parallel=4
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
