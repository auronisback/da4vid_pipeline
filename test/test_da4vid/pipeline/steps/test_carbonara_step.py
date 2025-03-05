import os
import shutil
import unittest
import warnings

import docker

from da4vid.containers.carbonara import CARBonAraContainer
from da4vid.containers.docker import DockerExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io.sample_io import sample_set_from_backbones
from da4vid.model.proteins import Epitope
from da4vid.pipeline.generation import CARBonAraStep
from test.cfg import RESOURCES_ROOT


class CARBonAraStepWithDockerTest(unittest.TestCase):

  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    resources = os.path.join(RESOURCES_ROOT, 'steps_test', 'carbonara_test')
    self.backbone_folder = os.path.join(resources, 'backbones')
    self.folder = os.path.join(resources, 'step_folder')
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(CARBonAraContainer.DEFAULT_IMAGE)

  def tearDown(self):
    self.client.close()
    if os.path.exists(self.folder):
      shutil.rmtree(self.folder)

  def test_carbonara_step_should_create_sequences_for_all_pdbs_in_input_folder(self):
    sample_set = sample_set_from_backbones(self.backbone_folder)
    epitope = Epitope('A', 27, 35, sample_set.samples()[0].protein)
    cb_step = CARBonAraStep(
      builder=self.builder,
      epitope=epitope,
      config=CARBonAraStep.CARBonAraConfig(
        num_sequences=5,
        imprint_ratio=.3,
        sampling_method=CARBonAraContainer.SAMPLING_SAMPLED
      ),
      gpu_manager=self.gpu_manager,
      client=self.client,
      name='CB_DEMO',
      folder=self.folder
    )
    res_set = cb_step.execute(sample_set)
    self.assertEqual(len(sample_set.samples()), len(res_set.samples()),
                     f'Number of resulting samples does not match')
    for i, sample in enumerate(res_set.samples()):
      self.assertEqual(5, len(sample.sequences()),
                       f'Sample {i} does not have 20 sequences')
      for j, sequence in enumerate(sample.sequences()):
        epi_resi = sequence.sequence_to_str()[epitope.start - 1: epitope.end]
        self.assertEqual(epitope.sequence(), epi_resi, f'Epitope mismatch for sample {i} sequence {j}: '
                                                       f'{epitope.sequence()} expected, but found {epi_resi}')
