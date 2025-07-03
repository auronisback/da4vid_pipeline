import docker
import os
import shutil
import unittest
import warnings

import dotenv
import spython.main

from da4vid.containers.docker import DockerExecutorBuilder
from da4vid.containers.pesto import PestoContainer
from da4vid.containers.singularity import SingularityExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io.sample_io import sample_set_from_backbones
from da4vid.pipeline.interaction import PestoStep
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class PestoStepWithDockerTest(unittest.TestCase):
  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.resource_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'pesto_test')
    self.exec_folder = os.path.join(self.resource_folder, 'step_folder')
    self.input_folder = os.path.join(self.resource_folder, 'inputs')
    self.resume_folder = os.path.join(self.resource_folder, 'masif_resume')
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(PestoContainer.DEFAULT_IMAGE)

  def tearDown(self):
    if os.path.isdir(self.exec_folder):
      shutil.rmtree(self.exec_folder)
    self.client.close()

  def test_should_evaluate_interactions(self):
    sample_set = sample_set_from_backbones(self.input_folder)
    step = PestoStep(
      builder=self.builder,
      name='PeSTo_STEP',
      folder=self.exec_folder,
      gpu_manager=self.gpu_manager
    )
    res_set = step.execute(sample_set)
    self.assertEqual(len(res_set.samples()), 3)
    for sample in sample_set.samples():
      for resi in sample.protein.residues():
        self.assertTrue(resi.props.has_key(PestoStep.PESTO_INTERACTION_PROP_KEY),
                        f'Interaction Probability not found for {sample.name} (res number: {resi.number})')
        self.assertIsNotNone(resi.props.get_value(PestoStep.PESTO_INTERACTION_PROP_KEY),
                             f'Interaction Probability is None for {sample.name} (res number: {resi.number})')


class PestoStepWithSingularityTest(unittest.TestCase):
  def setUp(self):
    self.client = spython.main.get_client()
    self.gpu_manager = CudaDeviceManager()
    self.sif_path = dotenv.dotenv_values(DOTENV_FILE)['PESTO_SIF']
    self.resource_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'pesto_test')
    self.exec_folder = os.path.join(self.resource_folder, 'step_folder')
    self.input_folder = os.path.join(self.resource_folder, 'inputs')
    self.resume_folder = os.path.join(self.resource_folder, 'masif_resume')
    self.builder = SingularityExecutorBuilder().set_client(self.client).set_sif_path(self.sif_path)

  def tearDown(self):
    if os.path.isdir(self.exec_folder):
      shutil.rmtree(self.exec_folder)

  def test_should_evaluate_interactions(self):
    sample_set = sample_set_from_backbones(self.input_folder)
    step = PestoStep(
      builder=self.builder,
      name='PeSTo_STEP',
      folder=self.exec_folder,
      gpu_manager=self.gpu_manager
    )
    res_set = step.execute(sample_set)
    self.assertEqual(len(res_set.samples()), 3)
    for sample in sample_set.samples():
      for resi in sample.protein.residues():
        self.assertTrue(resi.props.has_key(PestoStep.PESTO_INTERACTION_PROP_KEY),
                        f'Interaction Probability not found for {sample.name} (res number: {resi.number})')
        self.assertIsNotNone(resi.props.get_value(PestoStep.PESTO_INTERACTION_PROP_KEY),
                             f'Interaction Probability is None for {sample.name} (res number: {resi.number})')

