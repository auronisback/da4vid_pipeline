import os.path
import time
import unittest

from da4vid.pipeline.config import PipelineCreator
from da4vid.pipeline.utils import PipelinePrinter
from da4vid.pipeline.generation import RFdiffusionStep, BackboneFilteringStep, ProteinMPNNStep, CARBonAraStep
from da4vid.pipeline.steps import CompositeStep
from test.cfg import DOTENV_FILE, RESOURCES_ROOT


class PipelineCreatorTest(unittest.TestCase):
  def setUp(self):
    self.resources = os.path.join(RESOURCES_ROOT, 'pipeline_config_test')

  def test_simple_pipeline_creation_from_yml(self):
    yml_file = os.path.join(self.resources, 'simple_test.yml')
    creator = PipelineCreator(DOTENV_FILE)
    pipeline = creator.from_yml(yml_file)
    self.assertEqual('simple pipeline', pipeline.name)
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run')), pipeline.folder)
    self.assertEqual('A21-30', str(pipeline.epitope))
    self.assertEqual('antigen', pipeline.antigen.name)
    self.assertEqual(3, len(pipeline.steps))
    rf_step = pipeline.steps[0]
    self.assertIsInstance(rf_step, RFdiffusionStep)
    self.assertEqual('my_rfdiffusion', rf_step.name)
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', 'my_rfdiffusion')),
                     rf_step.get_context_folder())
    self.assertEqual(2000, rf_step.config.num_designs)
    self.assertEqual(23, rf_step.config.partial_T)
    self.assertEqual(5, rf_step.config.contacts_threshold)
    self.assertEqual(12, rf_step.config.rog_potential)
    bf_step = pipeline.steps[1]
    self.assertIsInstance(bf_step, BackboneFilteringStep)
    self.assertEqual('backbone_filtering', bf_step.name)
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', 'backbone_filtering')),
                     bf_step.get_context_folder())
    self.assertEqual(5, bf_step.ss_threshold)
    self.assertEqual(10, bf_step.rog_cutoff)
    self.assertFalse(bf_step.rog_percentage)
    pm_step = pipeline.steps[2]
    self.assertIsInstance(pm_step, ProteinMPNNStep)
    self.assertEqual('proteinmpnn', pm_step.name)
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', 'proteinmpnn')),
                     pm_step.get_context_folder())
    self.assertEqual(2000, pm_step.config.seqs_per_target)
    self.assertEqual(.5, pm_step.config.sampling_temp)
    self.assertEqual(.2, pm_step.config.backbone_noise)
    self.assertEqual(200, pm_step.config.batch_size)

  def test_two_iteration_pipeline_creation_from_yml(self):
    pipeline = PipelineCreator().from_yml(os.path.join(self.resources, 'two_iterations_test.yml'))
    self.assertEqual(2, len(pipeline.steps))
    it1_step = pipeline.steps[0]
    self.assertIsInstance(it1_step, CompositeStep)
    self.assertEqual('1st iteration', it1_step.name)
    self.assertEqual(3, len(it1_step.steps))
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', '1st_iteration', 'rfdiffusion')),
                     it1_step.steps[0].get_context_folder())
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', '1st_iteration', 'backbone_filtering')),
                     it1_step.steps[1].get_context_folder())
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', '1st_iteration', 'proteinmpnn')),
                     it1_step.steps[2].get_context_folder())
    it2_step = pipeline.steps[1]
    self.assertIsInstance(it2_step, CompositeStep)
    self.assertEqual(5, len(it2_step.steps))
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', '2nd_iteration', 'omegafold')),
                     it2_step.steps[0].get_context_folder())
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', '2nd_iteration', 'sequence_filtering')),
                     it2_step.steps[1].get_context_folder())
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', '2nd_iteration', 'colabfold')),
                     it2_step.steps[2].get_context_folder())
    self.assertEqual(os.path.abspath(os.path.join(self.resources, 'run', '2nd_iteration', 'colabfold_filtering')),
                     it2_step.steps[3].get_context_folder())
    self.assertEqual('my_collector', it2_step.steps[4].name)

  def test_pipeline_from_yml_raise_error_if_two_steps_have_the_same_context_folder(self):
      with self.assertRaises(PipelineCreator.PipelineCreationError):
        PipelineCreator().from_yml(os.path.join(self.resources, 'two_iterations_test_folder_mismatch.yml'))

  def test_pipeline_from_yml_with_carbonara_step(self):
    pipeline = PipelineCreator().from_yml(os.path.join(self.resources, 'carbonara_test.yml'))
    self.assertEqual(3, len(pipeline.steps))
    cb_step = pipeline.steps[2]
    self.assertEqual('CARBonAra', cb_step.name)
    self.assertIsInstance(cb_step, CARBonAraStep)
    self.assertEqual(100, cb_step.config.num_sequences)
    self.assertEqual(.2, cb_step.config.imprint_ratio)
    self.assertTrue(cb_step.config.ignore_het_atm)
    self.assertFalse(cb_step.config.ignore_water)
    self.assertEqual(4, len(cb_step.config.ignored_amino_acids))
    for ignored_aa in ['A', 'C', 'K', 'Y']:
      self.assertIn(ignored_aa, cb_step.config.ignored_amino_acids)



