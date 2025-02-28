import logging
from typing import List

from da4vid.containers.base import BaseContainer
from da4vid.containers.executor import ContainerExecutorBuilder
from da4vid.gpus.cuda import CudaDeviceManager


class MasifContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/masif:latest'

  CONTAINER_INPUT_FOLDER = '/masif/data/masif_site/inputs'
  CONTAINER_OUTPUT_FOLDER = '/masif/data/masif_site/outputs'

  PREPARE_LIST_SCRIPT = '/masif/data/masif_site/prepare_from_list.sh'
  PREDICT_SITE_SCRIPT = '/masif/data/masif_site/predict_site.sh'
  RAW_DATA_FOLDER = '/masif/data/masif_site/data_preparation/00-raw-pdbs'
  # Mesh coordinates are stored here, in a subfolder for each PDB in the list, as separate files (p1_<axis>.npy)
  PRECOMPUTED_FOLDER = '/masif/data/masif_site/data_preparation/04a-precomputation_9A/precomputation'
  # Predicted sites per-vertex are stored here in npy files
  PREDICTED_DATA = '/masif/data/masif_site/output/all_feat_3l/pred_data'

  def __init__(self, builder: ContainerExecutorBuilder, gpu_manager: CudaDeviceManager,
               input_folder: str, output_folder: str, out_logfile: str = None, err_logfile: str = None):
    super().__init__(
      builder=builder,
      gpu_manager=gpu_manager
    )
    self.input_folder = input_folder
    self.output_folder = output_folder
    self.out_logfile = out_logfile
    self.err_logfile = err_logfile

  def run(self) -> bool:
    self.builder.set_logs(self.out_logfile, self.err_logfile).set_volumes({
      self.input_folder: MasifContainer.CONTAINER_INPUT_FOLDER,
      self.output_folder: MasifContainer.CONTAINER_OUTPUT_FOLDER
    }).set_device(self.gpu_manager.next_device())
    with self.builder.build() as executor:
      logging.info(f"[HOST] Running MaSIF on device {executor.device().name}")
      res = True
      for cmd in self.__get_commands():
        res = executor.execute(cmd)
        if not res:
          break
      executor.execute(f'/bin/chmod 0777 --recursive {self.CONTAINER_OUTPUT_FOLDER}')
      return res

  def __get_commands(self) -> List[str]:
    return [
      f'/bin/bash {self.PREPARE_LIST_SCRIPT} {self.CONTAINER_INPUT_FOLDER}/list.txt {self.CONTAINER_INPUT_FOLDER}',
      f"/bin/bash -c '/bin/ls {self.PRECOMPUTED_FOLDER} > {self.CONTAINER_INPUT_FOLDER}/pred_list.txt'",
      f'/bin/bash {self.PREDICT_SITE_SCRIPT} -l {self.CONTAINER_INPUT_FOLDER}/pred_list.txt',
      f'/bin/rm {self.CONTAINER_INPUT_FOLDER}/pred_list.txt',
      f'/bin/cp -r {self.PREDICTED_DATA} {self.CONTAINER_OUTPUT_FOLDER}/pred_data',
      f'/bin/cp -r {self.PRECOMPUTED_FOLDER} {self.CONTAINER_OUTPUT_FOLDER}/meshes',
    ]
