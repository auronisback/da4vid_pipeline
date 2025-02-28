import os.path
import shutil
import unittest
import warnings

import docker
import dotenv
import spython.main

from da4vid.containers.docker import DockerExecutorBuilder, DockerExecutor
from da4vid.containers.pmpnn import ProteinMPNNContainer
from da4vid.containers.singularity import SingularityExecutorBuilder, SingularityExecutor
from da4vid.gpus.cuda import CudaDeviceManager
from test.cfg import RESOURCES_ROOT, DOTENV_FILE
from test.test_da4vid.containers.helpers import duplicate_image, remove_duplicate_image


class ProteinMPNNDockerContainerTest(unittest.TestCase):

  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources_path = os.path.join(RESOURCES_ROOT, 'container_test', 'pmpnn_test')
    self.input_dir = os.path.join(self.resources_path, 'inputs')
    self.output_dir = os.path.join(self.resources_path, 'outputs')
    os.makedirs(self.output_dir, exist_ok=True)
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(ProteinMPNNContainer.DEFAULT_IMAGE)
    duplicate_image(self.client, 'da4vid/protein-mpnn', 'pmpnn_duplicate')

  def tearDown(self):
    shutil.rmtree(self.output_dir)
    remove_duplicate_image(self.client, 'pmpnn_duplicate')
    self.client.close()

  def test_should_raise_error_if_invalid_image(self):
    self.builder.set_image('invalid_image')
    pmpnn = ProteinMPNNContainer(
      builder=self.builder,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10,
      gpu_manager=self.gpu_manager
    )
    with self.assertRaises(DockerExecutor.DockerImageNotFoundException):
      pmpnn.run()

  def test_should_create_output_sequences_with_default_image(self):
    pmpnn = ProteinMPNNContainer(
      builder=self.builder,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10,
      gpu_manager=self.gpu_manager
    )
    pmpnn.add_fixed_chain('A', [i for i in range(25, 32)])
    res = pmpnn.run()
    self.assertTrue(res, 'ProteinMPNN container container stopped with errors!')
    seq_folder = os.path.join(self.output_dir, 'seqs')
    sequences = [f for f in os.listdir(seq_folder) if f.endswith('.fa')]
    self.assertEqual(2, len(sequences))

  def test_should_create_output_sequences_with_specified_image(self):
    self.builder.set_image('pmpnn_duplicate')
    pmpnn = ProteinMPNNContainer(
      builder=self.builder,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10,
      gpu_manager=self.gpu_manager
    )
    pmpnn.add_fixed_chain('A', [i for i in range(25, 32)])
    res = pmpnn.run()
    self.assertTrue(res, 'ProteinMPNN container container stopped with errors!')
    seq_folder = os.path.join(self.output_dir, 'seqs')
    sequences = [f for f in os.listdir(seq_folder) if f.endswith('.fa')]
    self.assertEqual(2, len(sequences))

  def test_should_raise_error_if_invalid_seqs_per_target(self):
    with self.assertRaises(ValueError):
      ProteinMPNNContainer(
        builder=self.builder,
        input_dir=self.input_dir,
        output_dir=self.output_dir,
        seqs_per_target=-3,  # Invalid seqs_per_target
        batch_size=-3,
        gpu_manager=self.gpu_manager
      )

  def test_should_raise_error_if_seqs_per_target_is_not_a_multiple_of_batch_size(self):
    with self.assertRaises(ValueError):
      ProteinMPNNContainer(
        builder=self.builder,
        input_dir=self.input_dir,
        output_dir=self.output_dir,
        seqs_per_target=10,
        batch_size=32,
        gpu_manager=self.gpu_manager
      )


class ProteinMPNNSingularityContainerTest(unittest.TestCase):

  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources_path = os.path.join(RESOURCES_ROOT, 'container_test', 'pmpnn_test')
    self.input_dir = os.path.join(self.resources_path, 'inputs')
    self.output_dir = os.path.join(self.resources_path, 'outputs')
    os.makedirs(self.output_dir, exist_ok=True)
    self.client = spython.main.get_client()
    self.sif_path = dotenv.dotenv_values(DOTENV_FILE)['PROTEIN_MPNN_SIF']
    self.gpu_manager = CudaDeviceManager()
    self.builder = SingularityExecutorBuilder().set_client(self.client).set_sif_path(self.sif_path)

  def tearDown(self):
    shutil.rmtree(self.output_dir)

  def test_should_raise_error_if_invalid_sif(self):
    self.builder.set_sif_path('./invalid_image.sif')
    pmpnn = ProteinMPNNContainer(
      builder=self.builder,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10,
      batch_size=10,
      gpu_manager=self.gpu_manager
    )
    with self.assertRaises(SingularityExecutor.SifFileNotFoundException):
      pmpnn.run()

  def test_should_create_output_sequences_with_default_sif(self):
    pmpnn = ProteinMPNNContainer(
      builder=self.builder,
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10,
      batch_size=10,
      gpu_manager=self.gpu_manager
    )
    pmpnn.add_fixed_chain('A', [i for i in range(25, 32)])
    res = pmpnn.run()
    self.assertTrue(res, 'ProteinMPNN container container stopped with errors!')
    seq_folder = os.path.join(self.output_dir, 'seqs')
    sequences = [f for f in os.listdir(seq_folder) if f.endswith('.fa')]
    self.assertEqual(2, len(sequences))

  def test_should_raise_error_if_seqs_per_target_is_not_a_multiple_of_batch_size(self):
    with self.assertRaises(ValueError):
      ProteinMPNNContainer(
        builder=self.builder,
        input_dir=self.input_dir,
        output_dir=self.output_dir,
        seqs_per_target=10,
        batch_size=32,
        gpu_manager=self.gpu_manager
      )

  def test_should_raise_error_if_invalid_seqs_per_target(self):
    with self.assertRaises(ValueError):
      ProteinMPNNContainer(
        builder=self.builder,
        input_dir=self.input_dir,
        output_dir=self.output_dir,
        seqs_per_target=-20,
        batch_size=10,
        gpu_manager=self.gpu_manager
      )


if __name__ == '__main__':
  unittest.main()
