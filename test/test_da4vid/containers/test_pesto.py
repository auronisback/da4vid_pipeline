import os
import shutil
import unittest
import warnings
import docker
import dotenv
import spython.main

from da4vid.containers.docker import DockerExecutorBuilder
from da4vid.containers.pesto import PestoContainer
from da4vid.containers.singularity import SingularityExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class PestoWithDockerTest(unittest.TestCase):

  def setUp(self):
    # Ignoring docker SDK warnings
    warnings.simplefilter('ignore', ResourceWarning)
    self.gpu_manager = CudaDeviceManager()
    self.client = docker.from_env()
    self.resource_path = os.path.join(RESOURCES_ROOT, 'container_test', 'pesto_test')
    self.input_dir = os.path.join(self.resource_path, 'inputs')
    self.output_dir = os.path.join(self.resource_path, 'outputs')
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(PestoContainer.DEFAULT_IMAGE)
    os.makedirs(self.output_dir, exist_ok=True)

  def tearDown(self):
    # if os.path.exists(self.output_dir):
    #   shutil.rmtree(self.output_dir)
    self.client.close()

  def test_pesto_should_succeed_with_default_image(self):
    pc = PestoContainer(
      builder=self.builder,
      input_folder=self.input_dir,
      output_folder=self.output_dir,
      gpu_manager=self.gpu_manager
    )
    res = pc.run()
    self.assertTrue(res, 'PeSTo container ended unsuccessfully')
    self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'example1_if.pdb')))
    self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'example2_if.pdb')))

  # TODO: implement resume test


class PestoWithSingularityTest(unittest.TestCase):

  def setUp(self):
    self.gpu_manager = CudaDeviceManager()
    self.client = spython.main.get_client()
    self.resource_path = os.path.join(RESOURCES_ROOT, 'container_test', 'pesto_test')
    self.input_dir = os.path.join(self.resource_path, 'inputs')
    self.output_dir = os.path.join(self.resource_path, 'outputs')
    self.sif_path = dotenv.dotenv_values(DOTENV_FILE)['PESTO_SIF']
    self.builder = SingularityExecutorBuilder().set_client(self.client).set_sif_path(self.sif_path)
    os.makedirs(self.output_dir, exist_ok=True)

  def tearDown(self):
    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)

  def test_pesto_should_succeed_with_default_image(self):
    pc = PestoContainer(
      builder=self.builder,
      input_folder=self.input_dir,
      output_folder=self.output_dir,
      gpu_manager=self.gpu_manager
    )
    res = pc.run()
    self.assertTrue(res, 'PeSTo container ended unsuccessfully')
    self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'example1_if.pdb')))
    self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'example2_if.pdb')))
