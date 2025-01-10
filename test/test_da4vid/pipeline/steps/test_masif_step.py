import os
import shutil
import unittest
import warnings

import docker
import torch

from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io import read_from_pdb
from da4vid.io.sample_io import sample_set_from_backbones
from da4vid.pipeline.config import StaticConfig
from da4vid.pipeline.interaction import MasifStep, PointCloud2ResiPredictions
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class PointCloudTest(unittest.TestCase):
  def setUp(self):
    self.resource_root = os.path.join(RESOURCES_ROOT, 'steps_test', 'masif_test', 'point_clouds')

  def test_point_cloud_to_residues(self):
    folder = os.path.join(self.resource_root, 'sample1000')
    protein = read_from_pdb(os.path.join(self.resource_root, 'sample1000.pdb'))
    pc = PointCloud2ResiPredictions()
    pc.evaluate_interactions_for_protein(protein, folder)
    for resi in protein.residues():
      self.assertTrue(resi.props.has_key(MasifStep.MASIF_INTERACTION_PROP_KEY))
      self.assertFalse(torch.isnan(torch.tensor(resi.props.get_value(MasifStep.MASIF_INTERACTION_PROP_KEY))))


class MasifStepTest(unittest.TestCase):

  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.masif_image = StaticConfig.get(DOTENV_FILE).masif_image
    self.resource_folder = os.path.join(RESOURCES_ROOT, 'steps_test', 'masif_test')
    self.exec_folder = os.path.join(self.resource_folder, 'exec')
    self.input_folder = os.path.join(self.resource_folder, 'inputs')

  def test_should_evaluate_interactions(self):
    sample_set = sample_set_from_backbones(self.input_folder)
    step = MasifStep(
      name='masif',
      client=self.client,
      folder=self.exec_folder,
      gpu_manager=self.gpu_manager,
      image=self.masif_image
    )
    res_set = step.execute(sample_set)
    output_folders = os.listdir(step.output_dir)
    self.assertEqual(3, len(output_folders))
    self.assertIn('sample1000', output_folders)
    self.assertIn('sample1001', output_folders)
    self.assertIn('sample1002', output_folders)
    for sample in res_set.samples():
      for resi in sample.protein.residues():
        self.assertTrue(resi.props.has_key(MasifStep.MASIF_INTERACTION_PROP_KEY))
        self.assertFalse(torch.isnan(torch.tensor(resi.props.get_value(MasifStep.MASIF_INTERACTION_PROP_KEY))))

  def tearDown(self):
    shutil.rmtree(self.exec_folder)
    self.client.close()
