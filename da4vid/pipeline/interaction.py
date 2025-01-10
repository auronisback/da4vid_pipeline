import abc
import os
import shutil
import sys

import torch
import numpy as np

from da4vid.docker.masif import MasifContainer
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.model.proteins import Protein
from da4vid.model.samples import SampleSet
from da4vid.pipeline.steps import DockerStep


class MasifStep(DockerStep):

  MASIF_INTERACTION_PROP_KEY = 'masif.interaction_prob'

  class MasifConfig:
    """
    Encapsulates MaSIF configuration used to run the container.
    """
    pass

  def __init__(self, gpu_manager: CudaDeviceManager, **kwargs):
    super().__init__(**kwargs)
    self.gpu_manager = gpu_manager
    self.input_dir = os.path.join(self.get_context_folder(), 'inputs')
    self.output_dir = os.path.join(self.get_context_folder(), 'outputs')
    self.container_out_dir = os.path.join(self.get_context_folder(), 'tmp_out')

  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes the MaSIF site interaction evaluation on the sample set.
    :param sample_set: The sample set of proteins whose interaction needs to be evaluated
    :return: A SampleSet object with interaction scores for each residue in the protein
    """
    self.__create_input_folder(sample_set)
    self.__execute_container()
    self.__refactor_outputs()
    return self.__evaluate_interactions(sample_set)

  def __create_input_folder(self, sample_set: SampleSet):
    os.makedirs(self.input_dir, exist_ok=True)
    with open(os.path.join(self.input_dir, 'list.txt'), 'w') as f:
      for sample in sample_set.samples():
        basename = os.path.basename(sample.filepath)
        list_name = f'{sample.name}_A'
        f.write(f'{basename} {list_name}\n')
        shutil.copy2(sample.filepath, os.path.join(self.input_dir, basename))

  def __execute_container(self):
    os.makedirs(self.container_out_dir, exist_ok=True)
    masif = MasifContainer(
      client=self.client,
      image=self.image,
      gpu_manager=self.gpu_manager,
      input_folder=self.input_dir,
      output_folder=self.container_out_dir
    )
    masif.run()

  def __refactor_outputs(self):
    os.makedirs(self.output_dir, exist_ok=True)
    meshes_folder = os.path.join(self.container_out_dir, 'meshes')
    for mesh_folder in os.listdir(meshes_folder):
      out_mesh_folder = os.path.join(self.output_dir, mesh_folder.replace('_A', ''))
      shutil.move(os.path.join(meshes_folder, mesh_folder), out_mesh_folder)
      pred_data_file = os.path.join(self.container_out_dir, 'pred_data', f'pred_{mesh_folder}.npy')
      if not os.path.exists(pred_data_file):
        print(f'Unable to find predictions for {mesh_folder}', file=sys.stderr)
      shutil.move(pred_data_file, out_mesh_folder)
    shutil.rmtree(self.container_out_dir)

  def __evaluate_interactions(self, sample_set: SampleSet) -> SampleSet:
    for sample in sample_set.samples():
      masif_folder = os.path.join(self.output_dir, sample.name)
      PointCloud2ResiPredictions.evaluate_interactions_for_protein(sample.protein, masif_folder)
    return sample_set

  def input_folder(self) -> str:
    return self.input_dir

  def output_folder(self) -> str:
    return self.output_dir


class PointCloud2ResiPredictions(abc.ABC):
  """
  Utility class to retrieve per-residue interaction probabilities
  from point clouds produced by MaSIF. Separated from the MaSIF step
  to ease testing.
  """
  @staticmethod
  def evaluate_interactions_for_protein(protein: Protein, masif_folder: str) -> None:
    """
    Evaluates the per-residue interaction of a protein given the folder of point cloud
    coordinates and per-point predictions evaluated by MaSIF. A property for each residue
    in the protein will be set accordingly.
    :param protein: The protein whose per-residue interactions should be evaluated
    :param masif_folder: The folder with MaSIF point clouds and predictions
    """
    point_cloud = PointCloud2ResiPredictions.__point_cloud_from_folder(masif_folder)
    predictions = PointCloud2ResiPredictions.__predictions_from_folder(masif_folder)
    atom_coords = protein.coords().double()
    dist = torch.cdist(point_cloud, atom_coords)
    min_dist = torch.min(dist, dim=1)
    atoms = protein.atoms()
    resi_labels = torch.tensor([atoms[index].residue.id for index in min_dist.indices])
    # Evaluating average
    M = torch.zeros(max(resi_labels) + 1, predictions.shape[1])
    M[resi_labels, torch.arange(predictions.shape[1])] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    avg = torch.mm(M, predictions.T)
    # Assigning values to residues
    for i, resi in enumerate(protein.residues()):
      resi.props.add_value(MasifStep.MASIF_INTERACTION_PROP_KEY, avg[i].item())

  @staticmethod
  def __point_cloud_from_folder(folder: str) -> torch.Tensor:
    X = torch.from_numpy(np.load(os.path.join(folder, 'p1_X.npy')))
    Y = torch.from_numpy(np.load(os.path.join(folder, 'p1_Y.npy')))
    Z = torch.from_numpy(np.load(os.path.join(folder, 'p1_Z.npy')))
    return torch.stack([X, Y, Z], dim=1)

  @staticmethod
  def __predictions_from_folder(folder: str) -> torch.Tensor:
    return torch.from_numpy(np.load(os.path.join(folder, f'pred_{os.path.basename(folder)}_A.npy')))
