import os.path
import shutil
import unittest
import warnings

import docker

from da4vid.containers.docker import DockerExecutorBuilder, DockerExecutor
from da4vid.containers.masif import MasifContainer
from da4vid.gpus.cuda import CudaDeviceManager
from test.cfg import RESOURCES_ROOT
from test.test_da4vid.containers.helpers import duplicate_image, remove_duplicate_image


class MasifTest(unittest.TestCase):

  def setUp(self):
    # Ignoring docker SDK warnings
    warnings.simplefilter('ignore', ResourceWarning)
    self.gpu_manager = CudaDeviceManager()
    self.client = docker.from_env()
    duplicate_image(self.client, 'da4vid/masif', 'masif_duplicate')
    self.resource_path = os.path.join(RESOURCES_ROOT, 'container_test', 'masif_test')
    self.input_dir = os.path.join(self.resource_path, 'inputs')
    self.output_dir = os.path.join(self.resource_path, 'outputs')
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(MasifContainer.DEFAULT_IMAGE)
    os.makedirs(self.output_dir, exist_ok=True)

  def tearDown(self):
    shutil.rmtree(os.path.join(self.resource_path, 'outputs'))
    remove_duplicate_image(self.client, 'masif_duplicate')
    self.client.close()

  def test_masif_should_raise_error_if_invalid_image(self):
    self.builder.set_image('invalid_image')
    mc = MasifContainer(
      builder=self.builder,
      gpu_manager=self.gpu_manager,
      input_folder=self.input_dir,
      output_folder=self.output_dir
    )
    with self.assertRaises(DockerExecutor.DockerImageNotFoundException):
      mc.run()

  def test_masif_should_succeed_with_default_image(self):
    mc = MasifContainer(
      builder=self.builder,
      input_folder=self.input_dir,
      output_folder=self.output_dir,
      gpu_manager=self.gpu_manager
    )
    res = mc.run()
    self.assertTrue(res, 'Masif container ended unsuccessfully')

  def test_masif_should_succeed_with_specified_image(self):
    self.builder.set_image('masif_duplicate')
    mc = MasifContainer(
      builder=self.builder,
      input_folder=self.input_dir,
      output_folder=self.output_dir,
      gpu_manager=self.gpu_manager
    )
    res = mc.run()
    self.assertTrue(res, 'Masif container ended unsuccessfully')
    self.assertTrue(os.path.isdir(os.path.join(self.output_dir, 'meshes')),
                    'Folder with meshes does not exists')
    self.assertTrue(os.path.isdir(os.path.join(self.output_dir, 'pred_data')),
                    'Folder with predictions does not exists')
    meshes = os.listdir(os.path.join(self.output_dir, 'meshes'))
    self.assertIn('foo_A', meshes, '1st protein does not have a mesh')
    self.assertIn('bar_A', meshes, '2nd protein does not have a mesh')
    predictions = os.listdir(os.path.join(self.output_dir, 'pred_data'))
    self.assertIn('pred_foo_A.npy', predictions, '1st protein does not have predictions')
    self.assertIn('pred_bar_A.npy', predictions, '2nd protein does not have predictions')
