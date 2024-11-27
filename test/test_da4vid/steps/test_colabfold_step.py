import os.path
import shutil
import unittest

import docker
import dotenv

from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class ColabFoldStepTest(unittest.TestCase):

  def setUp(self):
    self.client = docker.from_env()
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['COLABFOLD_MODEL_FOLDER']
    self.output_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'colabfold_test', 'outputs')

  def tearDown(self):
    shutil.rmtree(self.output_folder)
    self.client.close()

  def test_colabfold_predictions(self):
    pass #ColabFoldStep