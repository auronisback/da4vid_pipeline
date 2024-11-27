import unittest

import dotenv

from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class ColabFoldStepTest(unittest.TestCase):

  def setUp(self):
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['COLABFOLD_MODEL_FOLDER']

  def test_colabfold_predictions(self):
    pass #ColabFoldStep