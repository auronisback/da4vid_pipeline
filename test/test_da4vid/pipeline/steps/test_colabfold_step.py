import os.path
import shutil
import unittest
import warnings

import docker
import dotenv

from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.pipeline.config import StaticConfig
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class ColabFoldStepTest(unittest.TestCase):

  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.image = StaticConfig.get(DOTENV_FILE).colabfold_image
    self.model_weights = StaticConfig.get(DOTENV_FILE).colabfold_models_dir
    self.folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'colabfold_test', 'step_folder')

  def tearDown(self):
    shutil.rmtree(self.folder, ignore_errors=True)
    self.client.close()

  def test_colabfold_predictions(self):
    pass #ColabFoldStep