import os.path
import shutil
import unittest
import warnings

import docker

from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io.sample_io import sample_set_from_backbones
from da4vid.model.proteins import Epitope
from da4vid.pipeline.generation import ProteinMPNNStep
from test.cfg import RESOURCES_ROOT


class ProteinMPNNStepTest(unittest.TestCase):
  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.input_dir = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'inputs')
    self.output_dir = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'outputs')
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()

  def tearDown(self):
    self.client.close()
    shutil.rmtree(self.output_dir)

  def test_protein_mpnn_step_on_multiple_backbone(self):
    sample_set = sample_set_from_backbones(self.input_dir)
    config = ProteinMPNNStep.ProteinMPNNConfig(
      output_dir=self.output_dir,
      epitope=Epitope('A', 19, 27, sample_set.samples()[0].protein),
      seqs_per_target=5,
      sampling_temp=0.2,
      backbone_noise=0.3,
      batch_size=5
    )
    new_set = ProteinMPNNStep(
      input_dir=self.input_dir,
      client=self.client,
      gpu_manager=self.gpu_manager,
      config=config
    ).execute(sample_set)
    samples = new_set.samples()
    self.assertEqual(5, len(samples),
                     f'Invalid number of backbones: {len(samples)} (exp: 5)')
    for i, sample in enumerate(samples):
      sequences = sample.sequences()
      self.assertEqual(5, len(sequences),
                       f'Invalid number of samples for protein {i}: {len(sequences)} (exp: 5)')
      epitope = sample.protein.sequence()[18:27]
      for sequence in sequences:
        self.assertIsNotNone(sequence.protein.get_prop('protein_mpnn'), 'protein_mpnn not found in props')
        resi = sequence.sequence_to_str()
        self.assertEqual(epitope, resi[18:27], f'Epitope mismatch: {resi[18:27]} (exp {epitope})')
