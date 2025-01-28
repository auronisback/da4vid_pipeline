import os.path
import unittest

from da4vid.io import read_from_pdb
from da4vid.model.proteins import Epitope
from da4vid.model.samples import SampleSet, Sample
from da4vid.pipeline.callbacks import ProgressSaver
from da4vid.pipeline.steps import PipelineStep, PipelineRootStep
from test.cfg import RESOURCES_ROOT


class ProgressSaverTest(unittest.TestCase):

  def setUp(self):
    self.antigen_path = os.path.join(RESOURCES_ROOT, 'callbacks_test', 'antigen.pdb')
    self.progress_file = os.path.join(RESOURCES_ROOT, 'callbacks_test', 'pipeline.progress')

  def tearDown(self):
    if os.path.exists(self.progress_file):
      os.unlink(self.progress_file)

  class __DummyStep(PipelineStep):

    def __init__(self, name: str, **kwargs):
      super().__init__(name, **kwargs)

    def _execute(self, sample_set: SampleSet) -> SampleSet:
      return sample_set

    def _resume(self, sample_set: SampleSet) -> SampleSet:
      return sample_set

    def output_folder(self) -> str:
      pass

    def input_folder(self) -> str:
      pass

  def __create_dummy_pipeline(self, saver: ProgressSaver) -> PipelineRootStep:
    antigen = Sample(
      name='DEMO',
      filepath=self.antigen_path,
      protein=read_from_pdb(self.antigen_path)
    )
    root = PipelineRootStep('root', antigen, Epitope('A', 21, 30),
                            os.path.dirname(self.progress_file), post_step_fn=saver.save_completed_step)
    root.add_step(ProgressSaverTest.__DummyStep('first_step', post_step_fn=saver.save_completed_step))
    root.add_step(ProgressSaverTest.__DummyStep('second_step', post_step_fn=saver.save_completed_step))
    root.add_step(ProgressSaverTest.__DummyStep('third_step', post_step_fn=saver.save_completed_step))
    return root

  def test_progress_saver_should_raise_error_if_invalid_file(self):
    with self.assertRaises(FileExistsError):
      ProgressSaver(os.path.dirname(self.progress_file))

  def test_progress_saver_on_file(self):
    saver = ProgressSaver(self.progress_file)
    pipeline = self.__create_dummy_pipeline(saver)
    pipeline.execute(None)
    self.assertTrue(os.path.exists(self.progress_file))
    with open(self.progress_file) as f:
      self.assertEqual(pipeline.steps[0].full_name(), f.readline().strip())
      self.assertEqual(pipeline.steps[1].full_name(), f.readline().strip())
      self.assertEqual(pipeline.steps[2].full_name(), f.readline().strip())
      self.assertEqual(pipeline.full_name(), f.readline().strip())

  def test_progress_saver_on_existing_file(self):
    pass