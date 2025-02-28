import os.path
import shutil
import unittest
import warnings

import docker
import dotenv
import spython

from da4vid.containers.docker import DockerExecutorBuilder
from da4vid.containers.rfdiffusion import RFdiffusionContainer
from da4vid.containers.singularity import SingularityExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io import read_from_pdb
from da4vid.model.proteins import Epitope
from da4vid.model.samples import SampleSet, Sample
from da4vid.pipeline.config import StaticConfig
from da4vid.pipeline.generation import RFdiffusionStep
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class RFdiffusionStepWithDockerTest(unittest.TestCase):
  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.model_weights = StaticConfig.get(DOTENV_FILE).rfdiffusion_models_dir
    self.folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'step_folder')
    self.resume_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'rf_resume')
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(RFdiffusionContainer.DEFAULT_IMAGE)

  def tearDown(self):
    if os.path.isdir(self.folder):
      shutil.rmtree(self.folder)
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
      num_designs=3,
      contacts_threshold=4,
      rog_potential=11,
      partial_T=5
    )
    step = RFdiffusionStep(
      builder=self.builder,
      name='rfdiffusion_demo',
      folder=self.folder,
      epitope=Epitope('A', 21, 30),
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

  def test_rfdiffusion_resume(self):
    pdb_demo = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'demo.pdb')
    orig_set = SampleSet()
    orig_set.add_samples(Sample(
      name='DEMO',
      filepath=pdb_demo,
      protein=read_from_pdb(pdb_demo)
    ))
    config = RFdiffusionStep.RFdiffusionConfig(
      num_designs=3,
      contacts_threshold=4,
      rog_potential=11,
      partial_T=5
    )
    step = RFdiffusionStep(
      builder=self.builder,
      name='rfdiffusion_demo',
      folder=self.resume_folder,
      epitope=Epitope('A', 21, 30),
      model_dir=self.model_weights,
      client=self.client,
      gpu_manager=self.gpu_manager,
      config=config
    )
    sample_set = step.resume(orig_set)
    self.assertEqual(3, len(sample_set.samples()))
    orig_sequence = orig_set.samples()[0].protein.sequence()
    for sample in sample_set.samples():
      self.assertEqual(orig_sequence[20:30], sample.protein.sequence()[20:30])


class RFdiffusionStepWithSingularityTest(unittest.TestCase):

  def setUp(self):
    self.model_weights = StaticConfig.get(DOTENV_FILE).rfdiffusion_models_dir
    self.folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'step_folder')
    self.resume_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'rf_resume')
    self.client = spython.main.get_client()
    self.sif_path = dotenv.dotenv_values(DOTENV_FILE)['RFDIFFUSION_SIF']
    self.gpu_manager = CudaDeviceManager()
    self.builder = SingularityExecutorBuilder().set_client(self.client).set_sif_path(self.sif_path)
    self.builder.preserve_quotes_in_cmds(['"'])

  def tearDown(self):
    if os.path.isdir(self.folder):
      shutil.rmtree(self.folder)

  def test_rfdiffusion_step_with_one_sample(self):
    pdb_demo = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'demo.pdb')
    orig_set = SampleSet()
    orig_set.add_samples(Sample(
      name='DEMO',
      filepath=pdb_demo,
      protein=read_from_pdb(pdb_demo)
    ))
    config = RFdiffusionStep.RFdiffusionConfig(
      num_designs=3,
      contacts_threshold=4,
      rog_potential=11,
      partial_T=5
    )
    step = RFdiffusionStep(
      builder=self.builder,
      name='rfdiffusion_demo',
      folder=self.folder,
      epitope=Epitope('A', 21, 30),
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

  def test_rfdiffusion_resume(self):
    pdb_demo = os.path.join(RESOURCES_ROOT, 'steps_test', 'rfdiffusion_test', 'demo.pdb')
    orig_set = SampleSet()
    orig_set.add_samples(Sample(
      name='DEMO',
      filepath=pdb_demo,
      protein=read_from_pdb(pdb_demo)
    ))
    config = RFdiffusionStep.RFdiffusionConfig(
      num_designs=3,
      contacts_threshold=4,
      rog_potential=11,
      partial_T=5
    )
    step = RFdiffusionStep(
      builder=self.builder,
      name='rfdiffusion_demo',
      folder=self.resume_folder,
      epitope=Epitope('A', 21, 30),
      model_dir=self.model_weights,
      client=self.client,
      gpu_manager=self.gpu_manager,
      config=config
    )
    sample_set = step.resume(orig_set)
    self.assertEqual(3, len(sample_set.samples()))
    orig_sequence = orig_set.samples()[0].protein.sequence()
    for sample in sample_set.samples():
      self.assertEqual(orig_sequence[20:30], sample.protein.sequence()[20:30])


if __name__ == '__main__':
  unittest.main()
