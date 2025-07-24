import os
import shutil
import unittest

from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io import read_pdb_folder
from da4vid.model.proteins import Epitope
from da4vid.model.samples import SampleSet, Sample
from da4vid.pipeline.interaction import InteractionWindowEvaluationStep
from test.cfg import RESOURCES_ROOT


class InteractionFilterTest(unittest.TestCase):

  def setUp(self):
    self.input_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'interaction_filter_test', 'inputs')
    self.step_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'interaction_filter_test', 'step_folder')

  def tearDown(self):
    if os.path.exists(self.step_folder):
      shutil.rmtree(self.step_folder)

  def test_evaluate_interaction_windows(self):
    sample_set = self.__read_pdb_folder_and_assign_interactions(self.input_folder, 'if')
    epitope = Epitope('A', 27, 36)
    step = InteractionWindowEvaluationStep(
      epitope=epitope,
      offset=3,
      gpu_manager=CudaDeviceManager(),
      interaction_key='if',
      name='interaction_test',
      folder=self.step_folder
    )
    res = step.execute(sample_set)
    for s in res.samples():
      self.assertTrue(s.protein.props.has_key('interaction_score'))
      self.assertIsNotNone(s.protein.props.get_value('interaction_score'))
    out_log = os.path.join(step.output_dir, 'interactions.txt')
    self.assertTrue(os.path.exists(out_log))
    values = []
    with open(out_log) as f:
      f.readline()  # Skipping header
      for line in f:
        name, value = line.split(';')
        values.append(float(value.strip()))
        self.assertTrue(name in [s.protein.name for s in res.samples()])
    for i in range(len(values) - 1):
      self.assertGreaterEqual(values[i], values[i+1])

  @staticmethod
  def __read_pdb_folder_and_assign_interactions(pdb_folder: str, prop_name: str) -> SampleSet:
    sample_set = SampleSet()
    proteins = read_pdb_folder(pdb_folder, b_fact_prop='if')
    for p in proteins:
      for r in p.residues():
        r.props.add_value(prop_name, r.atoms[0].props.get('if'))
    sample_set.add_samples([Sample(p.name, p.filename, p) for p in proteins])
    return sample_set
