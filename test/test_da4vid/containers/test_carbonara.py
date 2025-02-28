import os.path
import shutil
import unittest
import warnings

import docker

from da4vid.containers.carbonara import CARBonAraContainer
from da4vid.containers.docker import DockerExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from test.cfg import RESOURCES_ROOT


class CARBonAraContainerTest(unittest.TestCase):

  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.resources = os.path.join(RESOURCES_ROOT, 'container_test', 'carbonara_test')
    self.input_folder = os.path.join(self.resources, 'inputs')
    self.output_folder = os.path.join(self.resources, 'outputs')
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(CARBonAraContainer.DEFAULT_IMAGE)
    os.makedirs(self.output_folder)

  def tearDown(self):
    self.client.close()
    if os.path.exists(self.output_folder):
      shutil.rmtree(self.output_folder)

  def test_should_produce_sequences_from_pdf_files_with_default_image(self):
    carbonara = CARBonAraContainer(
      builder=self.builder,
      input_dir=self.input_folder,
      output_dir=self.output_folder,
      num_sequences=20,
      gpu_manager=self.gpu_manager,
    )
    res = carbonara.run()
    self.assertTrue(res, 'CARBonAra container exited unsuccessfully')
