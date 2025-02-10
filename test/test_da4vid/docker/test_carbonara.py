import os.path
import shutil
import unittest
import warnings

import docker

from da4vid.docker.carbonara import CARBonAraContainer
from da4vid.gpus.cuda import CudaDeviceManager
from test.cfg import RESOURCES_ROOT


class CARBonAraContainerTest(unittest.TestCase):

  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.resources = os.path.join(RESOURCES_ROOT, 'docker_test', 'carbonara_test')
    self.input_folder = os.path.join(self.resources, 'inputs')
    self.output_folder = os.path.join(self.resources, 'outputs')
    os.makedirs(self.output_folder)

  def tearDown(self):
    self.client.close()
    if os.path.exists(self.output_folder):
      shutil.rmtree(self.output_folder)

  def test_should_produce_sequences_from_pdf_files_with_default_image(self):
    carbonara = CARBonAraContainer(
      input_dir=self.input_folder,
      output_dir=self.output_folder,
      num_sequences=20,
      gpu_manager=self.gpu_manager,
      client=self.client
    )
    res = carbonara.run()
    self.assertTrue(res, 'CARBonAra container exited unsuccessfully')

