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
    self.folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'sequence_filtering_test', 'step_folder')
    self.resume_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'sequence_filtering_test', 'resume_folder')

  def tearDown(self):
    shutil.rmtree(self.folder, ignore_errors=True)

  def test_sequence_filtering_step_by_plddt_threshold_and_cutoff(self):
    sample_set = sample_set_from_folders(self.backbones_folder, self.folds_folder,
                                         model='omegafold', b_fact_prop='plddt')
    filtered_set = SequenceFilteringStep(
      name='seq_filtering_test',
      folder=self.folder,
      model='omegafold',
      plddt_threshold=51,
      average_cutoff=3
    ).execute(sample_set)
    samples = filtered_set.samples()
    self.assertEqual(2, len(samples))
    sample = samples[0]
    self.assertEqual(2, len(sample.sequences()))
    for fold in sample.get_folds_for_model('omegafold'):
      self.assertIsInstance(fold, Fold)
      self.assertTrue(fold.protein.props.has_key('omegafold.plddt'))
      self.assertGreaterEqual(fold.metrics.get_metric('plddt'), 51)
      self.assertEqual(fold.protein.props.get_value('omegafold.plddt'), fold.metrics.get_metric('plddt'))
    sample = samples[1]
    self.assertEqual(5, len(sample.sequences()))
    for fold in sample.get_folds_for_model('omegafold'):
      self.assertIsInstance(fold, Fold)
      self.assertTrue(fold.protein.props.has_key('omegafold.plddt'))
      self.assertGreaterEqual(fold.metrics.get_metric('plddt'), 51)
      self.assertEqual(fold.protein.props.get_value('omegafold.plddt'), fold.metrics.get_metric('plddt'))

  def test_resume_sequence_filtering_step(self):
    sample_set = sample_set_from_folders(self.backbones_folder, self.folds_folder,
                                         model='omegafold', b_fact_prop='plddt')
    filtered_set = SequenceFilteringStep(
      name='seq_filtering_test',
      folder=self.folder,
      model='omegafold',
      plddt_threshold=51,
      average_cutoff=3
    ).resume(sample_set)
    samples = filtered_set.samples()
    self.assertEqual(2, len(samples))
    sample = samples[0]
    self.assertEqual(2, len(sample.sequences()))
    for fold in sample.get_folds_for_model('omegafold'):
      self.assertIsInstance(fold, Fold)
      self.assertTrue(fold.protein.props.has_key('omegafold.plddt'))
      self.assertGreaterEqual(fold.metrics.get_metric('plddt'), 51)
      self.assertEqual(fold.protein.props.get_value('omegafold.plddt'), fold.metrics.get_metric('plddt'))
    sample = samples[1]
    self.assertEqual(5, len(sample.sequences()))
    for fold in sample.get_folds_for_model('omegafold'):
      self.assertIsInstance(fold, Fold)
      self.assertTrue(fold.protein.props.has_key('omegafold.plddt'))
      self.assertGreaterEqual(fold.metrics.get_metric('plddt'), 51)
      self.assertEqual(fold.protein.props.get_value('omegafold.plddt'), fold.metrics.get_metric('plddt'))

  def test_sequence_filtering_step_by_plddt_threshold_cutoff_and_maximum_number_of_samples(self):
    sample_set = sample_set_from_folders(self.backbones_folder, self.folds_folder,
                                         model='omegafold', b_fact_prop='plddt')
    filtered_set = SequenceFilteringStep(
      name='seq_filtering_test',
      folder=self.folder,
      model='omegafold',
      plddt_threshold=51,
      average_cutoff=3,
      max_samples=1,
      max_folds_per_sample=1
    ).execute(sample_set)
    samples = filtered_set.samples()
    self.assertEqual(1, len(samples))
    sample = samples[0]
    self.assertEqual(1, len(sample.sequences()))
    for fold in sample.get_folds_for_model('omegafold'):
      self.assertIsInstance(fold, Fold)
      self.assertTrue(fold.protein.props.has_key('omegafold.plddt'))
      self.assertGreaterEqual(fold.metrics.get_metric('plddt'), 51)
      self.assertEqual(fold.protein.props.get_value('omegafold.plddt'), fold.metrics.get_metric('plddt'))

  def test_sequence_filtering_step_by_with_max_folds_per_sequence(self):
    sample_set = sample_set_from_folders(self.backbones_folder, self.folds_folder,
                                         model='omegafold', b_fact_prop='plddt')
    filtered_set = SequenceFilteringStep(
      name='seq_filtering_test',
      folder=self.folder,
      model='omegafold',
      plddt_threshold=51,
      average_cutoff=3,
      max_folds_per_sample=3
    ).execute(sample_set)
    samples = filtered_set.samples()
    self.assertEqual(2, len(samples))
    sample = samples[0]
    self.assertEqual(2, len(sample.sequences()))
    for fold in sample.get_folds_for_model('omegafold'):
      self.assertIsInstance(fold, Fold)
      self.assertTrue(fold.protein.props.has_key('omegafold.plddt'))
      self.assertGreaterEqual(fold.metrics.get_metric('plddt'), 51)
      self.assertEqual(fold.protein.props.get_value('omegafold.plddt'), fold.metrics.get_metric('plddt'))
    sample = samples[1]
    self.assertEqual(3, len(sample.sequences()))
    for fold in sample.get_folds_for_model('omegafold'):
      self.assertIsInstance(fold, Fold)
      self.assertTrue(fold.protein.props.has_key('omegafold.plddt'))
      self.assertGreaterEqual(fold.metrics.get_metric('plddt'), 51)
      self.assertEqual(fold.protein.props.get_value('omegafold.plddt'), fold.metrics.get_metric('plddt'))