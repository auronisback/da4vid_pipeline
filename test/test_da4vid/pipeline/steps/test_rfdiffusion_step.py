import os.path
import shutil
import unittest
import warnings

import docker
import dotenv

from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io import read_from_pdb
from da4vid.model.proteins import Epitope
from da4vid.model.samples import SampleSet, Sample
from da4vid.pipeline.generation import RFdiffusionStep
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class RFdiffusionStepTest(unittest.TestCase):
  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['RFDIFFUSION_MODEL_FOLDER']
    self.output_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'outputs')
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()

  def tearDown(self):
    shutil.rmtree(self.output_folder)
    self.client.close()

  def test_rfdiffusion_step_with_one_sample(self):
    pdb_demo = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'demo.pdb')
    orig_set = SampleSet()
    orig_set.add_samples(Sample(
      name='DEMO',
      filepath=pdb_demo,
      protein=read_from_pdb(pdb_demo)
    ))
    config = RFdiffusionStep.RFdiffusionConfig(
      output_dir=self.output_folder,
      epitope=Epitope('A', 21, 30),
      num_designs=3,
      contacts_threshold=4,
      rog_potential=11,
      partial_T=5
    )
    step = RFdiffusionStep(
      model_dir=self.model_weights,
      client=self.client,
      gpu_manager=self.gpu_manager,
      config=config
    )
    sample_set = step.execute(orig_set)
    self.assertEqual(3, len(sample_set.samples()))
    orig_sequence = orig_set.samples()[0].protein.sequence()
    for sample in sample_set.samples():
      self.assertEqual(orig_sequence[20:30], sample.protein.sequence()[20:30])
