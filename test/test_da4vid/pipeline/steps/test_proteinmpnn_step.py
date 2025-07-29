import os.path
import shutil
import unittest
import warnings

import docker
import dotenv
import spython

from da4vid.containers.docker import DockerExecutorBuilder
from da4vid.containers.pmpnn import ProteinMPNNContainer
from da4vid.containers.singularity import SingularityExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io.sample_io import sample_set_from_backbones
from da4vid.model.proteins import Epitope
from da4vid.pipeline.generation import ProteinMPNNStep
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class ProteinMPNNStepWithDockerTest(unittest.TestCase):
  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.input_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'inputs')
    self.folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'step_test')
    self.resume_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'pmpnn_resume')
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(ProteinMPNNContainer.DEFAULT_IMAGE)

  def tearDown(self):
    self.client.close()
    if os.path.isdir(self.folder):
      shutil.rmtree(self.folder)

  def test_protein_mpnn_step_on_multiple_backbone(self):
    sample_set = sample_set_from_backbones(self.input_folder)
    config = ProteinMPNNStep.ProteinMPNNConfig(
      seqs_per_target=5,
      sampling_temp=0.2,
      backbone_noise=0.3,
      batch_size=5
    )
    new_set = ProteinMPNNStep(
      builder=self.builder,
      name='proteinmpnn_demo',
      folder=self.folder,
      epitope=Epitope('A', 19, 27, sample_set.samples()[0].protein),
      client=self.client,
      gpu_manager=self.gpu_manager,
      config=config,
    ).execute(sample_set)
    samples = new_set.samples()
    self.assertEqual(2, len(samples),
                     f'Invalid number of backbones: {len(samples)} (exp: 4)')
    epitopes = {
      'test_1': 'PEAKALLAK',
      'test_2': 'VLPGGAVVK',
    }
    for i, sample in enumerate(samples):
      sequences = sample.sequences()
      self.assertEqual(5, len(sequences),
                       f'Invalid number of samples for protein {i}: {len(sequences)} (exp: 5)')
      for sequence in sequences:
        sample_name = '_'.join(sequence.name.split('_')[:-1])
        self.assertIsNotNone(sequence.protein.get_prop('protein_mpnn'), 'protein_mpnn not found in props')
        resi = sequence.sequence_to_str()
        self.assertEqual(epitopes[sample_name], resi[18:27], f'Epitope mismatch for {sequence.name}: {resi[18:27]}'
                                                             f' (exp {epitopes[sample_name]})')


class ProteinMPNNStepWithSingularityTest(unittest.TestCase):

  def setUp(self):
    self.input_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'inputs')
    self.folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'step_test')
    self.resume_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'pmpnn_resume')
    self.client = spython.main.get_client()
    self.sif_path = dotenv.dotenv_values(DOTENV_FILE)['PROTEIN_MPNN_SIF']
    self.gpu_manager = CudaDeviceManager()
    self.builder = SingularityExecutorBuilder().set_client(self.client).set_sif_path(self.sif_path)

  def tearDown(self):
    if os.path.isdir(self.folder):
      shutil.rmtree(self.folder)

  def test_protein_mpnn_step_on_multiple_backbone(self):
    sample_set = sample_set_from_backbones(self.input_folder)
    config = ProteinMPNNStep.ProteinMPNNConfig(
      seqs_per_target=5,
      sampling_temp=0.2,
      backbone_noise=0.3,
      batch_size=5
    )
    new_set = ProteinMPNNStep(
      builder=self.builder,
      name='proteinmpnn_demo',
      folder=self.folder,
      epitope=Epitope('A', 19, 27, sample_set.samples()[0].protein),
      client=self.client,
      gpu_manager=self.gpu_manager,
      config=config,
    ).execute(sample_set)
    samples = new_set.samples()
    self.assertEqual(2, len(samples),
                     f'Invalid number of backbones: {len(samples)} (exp: 4)')
    epitopes = {
      'test_1': 'PEAKALLAK',
      'test_2': 'VLPGGAVVK',
    }
    for i, sample in enumerate(samples):
      sequences = sample.sequences()
      self.assertEqual(5, len(sequences),
                       f'Invalid number of samples for protein {i}: {len(sequences)} (exp: 5)')
      for sequence in sequences:
        sample_name = '_'.join(sequence.name.split('_')[:-1])
        self.assertIsNotNone(sequence.protein.get_prop('protein_mpnn'), 'protein_mpnn not found in props')
        resi = sequence.sequence_to_str()
        self.assertEqual(epitopes[sample_name], resi[18:27], f'Epitope mismatch for {sequence.name}: {resi[18:27]}'
                                                             f' (exp {epitopes[sample_name]})')


if __name__ == '__main__':
  unittest.main()
