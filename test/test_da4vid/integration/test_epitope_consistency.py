import os.path
import shutil
import unittest

from da4vid.io import read_from_pdb
from da4vid.pipeline.config import PipelineCreator
from test.cfg import RESOURCES_ROOT


class EpitopeConsistencyTest(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.resources = os.path.join(RESOURCES_ROOT, 'integration_test', 'epitope_consistency')
    self.pipeline_yml = os.path.join(self.resources, 'pipeline.yml')

  def test_epitopic_residues_should_be_the_same_between_rfdiffusion_and_protein_mpnn(self):
    pipeline = PipelineCreator().from_yml(self.pipeline_yml)
    try:
      res_set = pipeline.execute()
      diffused = read_from_pdb(os.path.join(pipeline.steps[0].get_context_folder(),
                                            'outputs', 'antigen_0.pdb'))
      expected_seq = 'GGGGGGGGGGGGGGGGGGGGGGGGGKGSGSTANLGGGGGGGGGGGGGGG'
      self.assertEqual(expected_seq, diffused.sequence())
      for s in res_set.sequences():
        self.assertEqual('KGSGSTANL', s.sequence_to_str()[pipeline.epitope.start - 1:
                                                          pipeline.epitope.end])
    finally:
      if os.path.isdir(pipeline.get_context_folder()):
        shutil.rmtree(pipeline.get_context_folder())
