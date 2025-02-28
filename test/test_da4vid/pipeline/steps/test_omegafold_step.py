import os.path
import shutil
import unittest
import warnings

import docker
import dotenv
import spython.main

from da4vid.containers.docker import DockerExecutorBuilder
from da4vid.containers.omegafold import OmegaFoldContainer
from da4vid.containers.singularity import SingularityExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.io.sample_io import sample_set_from_fasta_folders
from da4vid.pipeline.validation import OmegaFoldStep
from test.cfg import RESOURCES_ROOT, DOTENV_FILE


class OmegaFoldStepWithDockerTest(unittest.TestCase):

  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources = os.path.join(RESOURCES_ROOT, 'steps_test', 'omegafold_test')
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['OMEGAFOLD_MODEL_FOLDER']
    self.exec_folder = os.path.join(self.resources, 'of_step_test')
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    self.builder = DockerExecutorBuilder().set_client(self.client).set_image(OmegaFoldContainer.DEFAULT_IMAGE)

  def tearDown(self):
    if os.path.isdir(self.exec_folder):
      shutil.rmtree(self.exec_folder)
    self.client.close()

  def test_omegafold_on_single_sample(self):
    backbone_folder = os.path.join(self.resources, 'single', 'backbones')
    input_folder = os.path.join(self.resources, 'single', 'inputs')
    sample_set = sample_set_from_fasta_folders(backbone_folder, input_folder, from_pmpnn=False)
    config = OmegaFoldStep.OmegaFoldConfig(
      num_recycles=1,
      model_weights='2',
    )
    of_step = OmegaFoldStep(
      builder=self.builder,
      name='OF_STEP',
      folder=self.exec_folder,
      model_dir=self.model_weights,
      config=config,
      gpu_manager=self.gpu_manager
    )
    sample_set = of_step.execute(sample_set)
    output_folder = of_step.output_folder()
    pdb_list = [d for d in os.listdir(output_folder)
                if os.path.isdir(os.path.join(output_folder, d))]
    self.assertEqual(1, len(pdb_list),
                     f'Invalid number of predicted folders: {len(pdb_list)} (exp 1)')
    self.assertEqual(1, len(sample_set.samples()))
    sample = sample_set.samples()[0]
    sequences = sample.sequences()
    self.assertEqual(5, len(sequences), f'Invalid number of samples: {len(sequences)} (exp 5)')
    for seq in sequences:
      self.assertEqual(1, len(seq.folds()))
      self.assertIsNotNone(seq.get_fold_for_model('omegafold'))

  def test_omegafold_on_multiple_samples(self):
    backbone_folder = os.path.join(self.resources, 'multiple', 'backbones')
    fasta_input_folder = os.path.join(self.resources, 'multiple', 'inputs')
    sample_set = sample_set_from_fasta_folders(backbone_folder, fasta_input_folder, from_pmpnn=False)
    print([s.filepath for s in sample_set.sequences()])
    config = OmegaFoldStep.OmegaFoldConfig(
      num_recycles=1,
      model_weights='2',
    )
    of_step = OmegaFoldStep(
      name='OF_STEP',
      folder=self.exec_folder,
      builder=self.builder,
      model_dir=self.model_weights,
      config=config,
      gpu_manager=self.gpu_manager
    )
    sample_set = of_step.execute(sample_set)
    self.__check_multiple_sequences(of_step, sample_set)

  def test_omegafold_resume_step(self):
    resume_folder = os.path.join(self.resources, 'of_resume')
    backbone_folder = os.path.join(self.resources, 'multiple', 'backbones')
    fasta_input_folder = os.path.join(self.resources, 'multiple', 'inputs')
    sample_set = sample_set_from_fasta_folders(backbone_folder, fasta_input_folder, from_pmpnn=False)
    config = OmegaFoldStep.OmegaFoldConfig(
      num_recycles=1,
      model_weights='2',
    )
    of_step = OmegaFoldStep(
      builder=self.builder,
      name='OF_STEP',
      folder=resume_folder,
      image=OmegaFoldContainer.DEFAULT_IMAGE,
      model_dir=self.model_weights,
      config=config,
      gpu_manager=self.gpu_manager
    )
    sample_set = of_step.resume(sample_set)
    self.__check_multiple_sequences(of_step, sample_set)

  def __check_multiple_sequences(self, step, sample_set):
    output_folder = step.output_folder()
    pdb_list = [d for d in os.listdir(output_folder)
                if os.path.isdir(os.path.join(output_folder, d))]
    self.assertEqual(2, len(pdb_list),
                     f'Invalid number of predicted folders: {len(pdb_list)} (exp 2)')
    self.assertEqual(2, len(sample_set.samples()))
    sample = sample_set.get_sample_by_name('sample_1000')
    self.assertIsNotNone(sample, 'sample_1000 not found')
    sequences = sample.sequences()
    self.assertEqual(2, len(sequences), f'Invalid number of samples for sample_1000'
                                        f': {len(sequences)} (exp 2)')
    for seq in sequences:
      self.assertEqual(1, len(seq.folds()))
      self.assertIsNotNone(seq.get_fold_for_model('omegafold'))
    sample = sample_set.get_sample_by_name('sample_1001')
    self.assertIsNotNone(sample, 'sample_1001 not found')
    sequences = sample.sequences()
    self.assertEqual(3, len(sequences), f'Invalid number of samples for sample_1001'
                                        f': {len(sequences)} (exp 3)')
    for seq in sequences:
      self.assertEqual(1, len(seq.folds()))
      self.assertIsNotNone(seq.get_fold_for_model('omegafold'))


class OmegaFoldStepWithSingularityTest(unittest.TestCase):

  def setUp(self):
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources = os.path.join(RESOURCES_ROOT, 'steps_test', 'omegafold_test')
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['OMEGAFOLD_MODEL_FOLDER']
    self.sif_path = dotenv.dotenv_values(DOTENV_FILE)['OMEGAFOLD_SIF']
    self.exec_folder = os.path.join(self.resources, 'of_step_test')
    self.client = spython.main.get_client()
    self.gpu_manager = CudaDeviceManager()
    self.builder = SingularityExecutorBuilder().set_client(self.client).set_sif_path(self.sif_path)

  def tearDown(self):
    if os.path.isdir(self.exec_folder):
      shutil.rmtree(self.exec_folder)

  def test_omegafold_on_single_sample(self):
    backbone_folder = os.path.join(self.resources, 'single', 'backbones')
    input_folder = os.path.join(self.resources, 'single', 'inputs')
    sample_set = sample_set_from_fasta_folders(backbone_folder, input_folder, from_pmpnn=False)
    config = OmegaFoldStep.OmegaFoldConfig(
      num_recycles=1,
      model_weights='2',
    )
    of_step = OmegaFoldStep(
      builder=self.builder,
      name='OF_STEP',
      folder=self.exec_folder,
      model_dir=self.model_weights,
      config=config,
      gpu_manager=self.gpu_manager
    )
    sample_set = of_step.execute(sample_set)
    output_folder = of_step.output_folder()
    pdb_list = [d for d in os.listdir(output_folder)
                if os.path.isdir(os.path.join(output_folder, d))]
    self.assertEqual(1, len(pdb_list),
                     f'Invalid number of predicted folders: {len(pdb_list)} (exp 1)')
    self.assertEqual(1, len(sample_set.samples()))
    sample = sample_set.samples()[0]
    sequences = sample.sequences()
    self.assertEqual(5, len(sequences), f'Invalid number of samples: {len(sequences)} (exp 5)')
    for seq in sequences:
      self.assertEqual(1, len(seq.folds()))
      self.assertIsNotNone(seq.get_fold_for_model('omegafold'))

  def test_omegafold_on_multiple_samples(self):
    backbone_folder = os.path.join(self.resources, 'multiple', 'backbones')
    fasta_input_folder = os.path.join(self.resources, 'multiple', 'inputs')
    sample_set = sample_set_from_fasta_folders(backbone_folder, fasta_input_folder, from_pmpnn=False)
    print([s.filepath for s in sample_set.sequences()])
    config = OmegaFoldStep.OmegaFoldConfig(
      num_recycles=1,
      model_weights='2',
    )
    of_step = OmegaFoldStep(
      name='OF_STEP',
      folder=self.exec_folder,
      builder=self.builder,
      model_dir=self.model_weights,
      config=config,
      gpu_manager=self.gpu_manager
    )
    sample_set = of_step.execute(sample_set)
    self.__check_multiple_sequences(of_step, sample_set)

  def test_omegafold_resume_step(self):
    resume_folder = os.path.join(self.resources, 'of_resume')
    backbone_folder = os.path.join(self.resources, 'multiple', 'backbones')
    fasta_input_folder = os.path.join(self.resources, 'multiple', 'inputs')
    sample_set = sample_set_from_fasta_folders(backbone_folder, fasta_input_folder, from_pmpnn=False)
    config = OmegaFoldStep.OmegaFoldConfig(
      num_recycles=1,
      model_weights='2',
    )
    of_step = OmegaFoldStep(
      builder=self.builder,
      name='OF_STEP',
      folder=resume_folder,
      image=OmegaFoldContainer.DEFAULT_IMAGE,
      model_dir=self.model_weights,
      config=config,
      gpu_manager=self.gpu_manager
    )
    sample_set = of_step.resume(sample_set)
    self.__check_multiple_sequences(of_step, sample_set)

  def __check_multiple_sequences(self, step, sample_set):
    output_folder = step.output_folder()
    pdb_list = [d for d in os.listdir(output_folder)
                if os.path.isdir(os.path.join(output_folder, d))]
    self.assertEqual(2, len(pdb_list),
                     f'Invalid number of predicted folders: {len(pdb_list)} (exp 2)')
    self.assertEqual(2, len(sample_set.samples()))
    sample = sample_set.get_sample_by_name('sample_1000')
    self.assertIsNotNone(sample, 'sample_1000 not found')
    sequences = sample.sequences()
    self.assertEqual(2, len(sequences), f'Invalid number of samples for sample_1000'
                                        f': {len(sequences)} (exp 2)')
    for seq in sequences:
      self.assertEqual(1, len(seq.folds()))
      self.assertIsNotNone(seq.get_fold_for_model('omegafold'))
    sample = sample_set.get_sample_by_name('sample_1001')
    self.assertIsNotNone(sample, 'sample_1001 not found')
    sequences = sample.sequences()
    self.assertEqual(3, len(sequences), f'Invalid number of samples for sample_1001'
                                        f': {len(sequences)} (exp 3)')
    for seq in sequences:
      self.assertEqual(1, len(seq.folds()))
      self.assertIsNotNone(seq.get_fold_for_model('omegafold'))
