import os.path
import unittest

import docker

from da4vid.io import read_from_pdb, read_pdb_folder
from da4vid.pipeline.generation import ProteinMPNNStep
from test.cfg import RESOURCES_ROOT


class ProteinMPNNStepTest(unittest.TestCase):
  def setUp(self):
    self.client = docker.from_env()

  def tearDown(self):
    self.client.close()

  def test_protein_mpnn_step_on_single_backbone(self):
    input_dir = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'inputs')
    output_dir = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'outputs')
    protein = read_from_pdb(os.path.join(input_dir, 'pmpnn_test.pdb'))
    sample_set = ProteinMPNNStep(
      backbones=[protein],
      input_dir=input_dir,
      output_dir=output_dir,
      chain='A',
      epitope=(19, 27),
      seqs_per_target=5,
      sampling_temp=0.1,
      backbone_noise=0,
      client=self.client
    ).execute()
    samples = sample_set.samples()
    self.assertEqual(1, len(samples),
                     f'Invalid number of backbones: {len(samples)} (exp: 1)')
    sequences = samples[0].sequences()
    self.assertEqual(5, len(sequences),
                     f'Invalid number of samples: {len(sequences)} (exp: 10)')
    epitope = samples[0].protein.sequence()[18:27]
    for sequence in sequences:
      self.assertIsNotNone(sequence.protein.get_prop('protein_mpnn'), 'protein_mpnn not found in props')
      resi = sequence.sequence_to_str()
      self.assertEqual(epitope, resi[18:27], f'Epitope mismatch: {resi[18:27]} (exp {epitope})')

  def test_protein_mpnn_step_on_multiple_backbone(self):
    input_dir = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'inputs')
    output_dir = os.path.join(RESOURCES_ROOT, 'steps_test', 'pmpnn_test', 'outputs')
    proteins = read_pdb_folder(input_dir)
    sample_set = ProteinMPNNStep(
      backbones=proteins,
      input_dir=input_dir,
      output_dir=output_dir,
      chain='A',
      epitope=(19, 27),
      seqs_per_target=5,
      sampling_temp=0.2,
      backbone_noise=0.3
    ).execute()
    samples = sample_set.samples()
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

