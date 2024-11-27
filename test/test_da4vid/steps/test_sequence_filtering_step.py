import os
import shutil
import unittest

from da4vid.io.sample_io import sample_set_from_folders
from da4vid.model.samples import Fold
from da4vid.pipeline.validation import SequenceFilteringStep
from test.cfg import RESOURCES_ROOT


class SequenceFilteringStepTest(unittest.TestCase):

  def setUp(self):
    self.backbones_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'sequence_filtering_test', 'backbones')
    self.folds_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'sequence_filtering_test', 'folds')
    self.output_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'sequence_filtering_test', 'output')

  def tearDown(self):
    shutil.rmtree(self.output_folder, ignore_errors=True)

  def test_sequence_filtering_step_by_plddt_threshold_and_cutoff(self):
    sample_set = sample_set_from_folders(self.backbones_folder, self.folds_folder,
                                         model='omegafold', b_fact_prop='plddt')
    filtered_set = SequenceFilteringStep(
      model='omegafold',
      plddt_threshold=25,
      avg_cutoff=3
    ).execute(sample_set)
    self.assertEqual(9, len(filtered_set.samples()))
    for sample in filtered_set.samples():
      self.assertIsInstance(sample, Fold)
      self.assertGreaterEqual(sample.metrics.get_metric('plddt'), 25)
