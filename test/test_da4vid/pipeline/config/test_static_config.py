import unittest

from da4vid.pipeline.config import StaticConfig
from test.cfg import DOTENV_FILE


class StaticConfigTest(unittest.TestCase):

  def test_should_load_static_configuration_from_environment(self):
    config = StaticConfig.get(DOTENV_FILE)
    print(config)
