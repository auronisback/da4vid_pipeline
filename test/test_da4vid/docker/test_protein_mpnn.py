import os.path
import shutil
import unittest
import warnings

import docker

from da4vid.docker.base import BaseContainer
from da4vid.docker.pmpnn import ProteinMPNNContainer
from da4vid.gpus.cuda import CudaDeviceManager
from test.cfg import RESOURCES_ROOT
from test.test_da4vid.docker.helpers import duplicate_image, remove_duplicate_image


class ProteinMPNNContainerTest(unittest.TestCase):

  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources_path = os.path.join(RESOURCES_ROOT, 'docker_test', 'pmpnn_test')
    self.input_dir = os.path.join(self.resources_path, 'inputs')
    self.output_dir = os.path.join(self.resources_path, 'outputs')
    os.makedirs(self.output_dir, exist_ok=True)
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    duplicate_image(self.client, 'da4vid/protein-mpnn', 'pmpnn_duplicate')

  def tearDown(self):
    shutil.rmtree(self.output_dir)
    remove_duplicate_image(self.client, 'pmpnn_duplicate')
    self.client.close()

  def test_should_raise_error_if_invalid_image(self):
    pmpnn = ProteinMPNNContainer(
      image='invalid_image',
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10,
      client=self.client,
      gpu_manager=self.gpu_manager
    )
    with self.assertRaises(BaseContainer.DockerImageNotFoundException):
      pmpnn.run()

  def test_should_create_output_sequences_with_default_image(self):
    res = ProteinMPNNContainer(
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10,
      client=self.client,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertTrue(res, 'ProteinMPNN container container stopped with errors!')
    seq_folder = os.path.join(self.output_dir, 'seqs')
    sequences = [f for f in os.listdir(seq_folder) if f.endswith('.fa')]
    self.assertEqual(2, len(sequences))

  def test_should_create_output_sequences_with_specified_image(self):
    res = ProteinMPNNContainer(
      image='pmpnn_duplicate',
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10,
      client=self.client,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertTrue(res, 'ProteinMPNN container container stopped with errors!')
    seq_folder = os.path.join(self.output_dir, 'seqs')
    sequences = [f for f in os.listdir(seq_folder) if f.endswith('.fa')]
    self.assertEqual(2, len(sequences))

  def test_should_return_false_if_error_in_container_occurs(self):
    res = ProteinMPNNContainer(
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=3,
      batch_size=-10,  # Invalid batch size
      client=self.client,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertFalse(res, 'Error in the ProteinMPNN container have not been discovered')
