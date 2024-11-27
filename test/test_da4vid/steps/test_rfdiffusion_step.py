import os.path
import unittest

import docker
import dotenv

from da4vid.io import read_from_pdb
from da4vid.model.samples import SampleSet, Sample
from da4vid.pipeline.generation import RFdiffusionStep
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class RFdiffusionStepTest(unittest.TestCase):
  def setUp(self):
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['RFDIFFUSION_MODEL_FOLDER']
    self.client = docker.from_env()

  def tearDown(self):
    self.client.close()

  def test_rfdiffusion_step_with_one_sample(self):
    pdb_demo = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'demo.pdb')
    output_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'outputs')
    orig_set = SampleSet()
    orig_set.add_samples(Sample(
      name='DEMO',
      filepath=pdb_demo,
      protein=read_from_pdb(pdb_demo)
    ))
    step = RFdiffusionStep(
      model_dir=self.model_weights,
      output_dir=output_folder,
      epitope=(21, 30),
      num_designs=3,
      contacts_threshold=4,
      rog_potential=11,
      partial_T=5,
      client=self.client
    )
    sample_set = step.execute(orig_set)
    self.assertEqual(3, len(sample_set.samples()))
    orig_sequence = orig_set.samples()[0].protein.sequence()
    for sample in sample_set.samples():
      self.assertEqual(orig_sequence[20:30], sample.protein.sequence()[20:30])
