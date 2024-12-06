import os.path
import shutil
import unittest

import docker

from da4vid.docker.base import BaseContainer
from da4vid.docker.pmpnn import ProteinMPNNContainer
from test.cfg import RESOURCES_ROOT


class ProteinMPNNContainerTest(unittest.TestCase):

  def setUp(self):
    self.resources_path = os.path.join(RESOURCES_ROOT, 'docker_test', 'pmpnn_test')
    self.input_dir = os.path.join(self.resources_path, 'inputs')
    self.output_dir = os.path.join(self.resources_path, 'outputs')
    os.makedirs(self.output_dir, exist_ok=True)
    self.client = docker.from_env()
    self.client.images.get('da4vid/protein-mpnn').tag('pmpnn_duplicate', 'latest')

  def tearDown(self):
    shutil.rmtree(self.output_dir)
    self.client.images.remove('pmpnn_duplicate')
    self.client.close()

  def test_should_raise_error_if_invalid_image(self):
    pmpnn = ProteinMPNNContainer(
      image='invalid_image',
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10
    )
    with self.assertRaises(BaseContainer.DockerImageNotFoundException):
      pmpnn.run()

  def test_should_create_output_sequences_with_default_image(self):
    res = ProteinMPNNContainer(
      input_dir=self.input_dir,
      output_dir=self.output_dir,
      seqs_per_target=10
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
      seqs_per_target=10
    ).run()
    self.assertTrue(res, 'ProteinMPNN container container stopped with errors!')
    seq_folder = os.path.join(self.output_dir, 'seqs')
    sequences = [f for f in os.listdir(seq_folder) if f.endswith('.fa')]
    self.assertEqual(2, len(sequences))
