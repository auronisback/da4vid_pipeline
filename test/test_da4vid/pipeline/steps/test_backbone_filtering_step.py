import os.path
import shutil
import unittest

from da4vid.io.sample_io import sample_set_from_backbones
from da4vid.pipeline.generation import BackboneFilteringStep
from test.cfg import RESOURCES_ROOT


class BackboneFilteringStepTest(unittest.TestCase):

  def setUp(self):
    self.backbone_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'backbone_filtering_test', 'inputs')
    self.step_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'backbone_filtering_test', 'step_folder')

  def tearDown(self):
    shutil.rmtree(self.step_folder)

  def test_filtering_with_ss_threshold(self):
    sample_set = sample_set_from_backbones(self.backbone_folder)
    filtered_set = BackboneFilteringStep(
      name='bb_filter_test',
      ss_threshold=4,
      rog_cutoff=3,
      rog_percentage=False,
      folder=self.step_folder
    ).execute(sample_set)
    self.assertEqual(17, len(filtered_set.samples()))
    counts = {}
    for sample in filtered_set.samples():
      self.assertTrue(os.path.isfile(sample.filepath))
      self.assertTrue(os.path.isfile(sample.protein.filename))
      ss = sample.protein.get_prop('ss')
      if ss not in counts.keys():
        counts[ss] = []
      counts[ss].append(sample)
    for ss in counts.keys():
      self.assertGreaterEqual(ss, 4)
      self.assertLessEqual(len(counts[ss]), 3)

  def test_resume_filtering(self):
    sample_set = sample_set_from_backbones(self.backbone_folder)
    filtered_set = BackboneFilteringStep(
      name='bb_filter_test',
      ss_threshold=4,
      rog_cutoff=3,
      rog_percentage=False,
      folder=self.step_folder
    ).resume(sample_set)
    self.assertEqual(17, len(filtered_set.samples()))
    counts = {}
    for sample in filtered_set.samples():
      self.assertTrue(os.path.isfile(sample.filepath))
      self.assertTrue(os.path.isfile(sample.protein.filename))
      ss = sample.protein.get_prop('ss')
      if ss not in counts.keys():
        counts[ss] = []
      counts[ss].append(sample)
    for ss in counts.keys():
      self.assertGreaterEqual(ss, 4)
      self.assertLessEqual(len(counts[ss]), 3)
