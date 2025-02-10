import os
import time
import unittest

from da4vid.io import read_from_pdb
from da4vid.model.proteins import Epitope
from da4vid.model.samples import SampleSet, Sample
from da4vid.pipeline.callbacks import ElapsedTimeSaver
from da4vid.pipeline.steps import PipelineStep, PipelineRootStep
from test.cfg import RESOURCES_ROOT


class ElapsedTimeSaverTest(unittest.TestCase):
  def setUp(self):
    self.antigen_path = os.path.join(RESOURCES_ROOT, 'callbacks_test', 'antigen.pdb')
    self.time_elapsed_file = os.path.join(RESOURCES_ROOT, 'callbacks_test', 'elapsed_time.csv')
    self.already_existing = os.path.join(RESOURCES_ROOT, 'callbacks_test', 'already_existing.csv')

  def tearDown(self):
    if os.path.exists(self.time_elapsed_file):
      os.unlink(self.time_elapsed_file)

  class __WaitingStep(PipelineStep):

    def __init__(self, secs_waiting: int):
      super().__init__(f'waiting_{secs_waiting}_seconds')
      self.secs_waiting = secs_waiting

    def _execute(self, sample_set: SampleSet | None) -> SampleSet:
      time.sleep(self.secs_waiting)
      return sample_set

    def _resume(self, sample_set: SampleSet | None) -> SampleSet:
      time.sleep(self.secs_waiting)
      return sample_set

    def output_folder(self) -> str:
      return ''

    def input_folder(self) -> str:
      return ''

  def __create_dummy_pipeline(self) -> PipelineRootStep:
    antigen = Sample(
      name='DEMO',
      filepath=self.antigen_path,
      protein=read_from_pdb(self.antigen_path)
    )
    root = PipelineRootStep('root', antigen, Epitope('A', 21, 30),
                            os.path.dirname(self.time_elapsed_file))
    root.add_step(self.__WaitingStep(1))
    root.add_step(self.__WaitingStep(2))
    root.add_step(self.__WaitingStep(3))
    return root

  def test_should_store_elapsed_time(self):
    pipeline = self.__create_dummy_pipeline()
    time_saver = ElapsedTimeSaver(self.time_elapsed_file)
    time_saver.register(pipeline)
    pipeline.execute()
    self.assertTrue(os.path.isfile(self.time_elapsed_file))
    times = {}
    with open(self.time_elapsed_file) as f:
      f.readline()  # Skipping headers
      for line in f:
        tokens = line.strip().split(';')
        times[tokens[0]] = float(tokens[1])
    self.assertEqual(4, len(times.keys()))
    self.assertIn('root', times.keys())
    self.assertIn('root.waiting_1_seconds', times.keys())
    self.assertIn('root.waiting_2_seconds', times.keys())
    self.assertIn('root.waiting_3_seconds', times.keys())
    self.assertAlmostEqual(1., times['root.waiting_1_seconds'], 1)
    self.assertAlmostEqual(2., times['root.waiting_2_seconds'], 1)
    self.assertAlmostEqual(3., times['root.waiting_3_seconds'], 1)
    self.assertAlmostEqual(6., times['root'], 1)

  def test_should_raise_an_error_if_the_elapsed_time_file_already_exists_and_no_force_rewrite(self):
    with self.assertRaises(FileExistsError):
      ElapsedTimeSaver(self.already_existing)
