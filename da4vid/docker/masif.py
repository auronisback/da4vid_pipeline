import sys
from typing import List

import docker

from da4vid.docker.base import BaseContainer, ContainerLogs
from da4vid.gpus.cuda import CudaDeviceManager


class MasifContainer(BaseContainer):
  DEFAULT_IMAGE = 'da4vid/masif:latest'

  INPUT_FOLDER = '/masif/data/masif_site/inputs'
  OUTPUT_FOLDER = '/masif/data/masif_site/outputs'

  PREPARE_LIST_SCRIPT = '/masif/data/masif_site/prepare_from_list.sh'
  PREDICT_SITE_SCRIPT = '/masif/data/masif_site/predict_site.sh'
  RAW_DATA_FOLDER = '/masif/data/masif_site/data_preparation/00-raw-pdbs'
  # Mesh coordinates are stored here, in a subfolder for each PDB in the list, as separate files (p1_<axis>.npy)
  PRECOMPUTED_FOLDER = '/masif/data/masif_site/data_preparation/04a-precomputation_9A/precomputation'
  # Predicted sites per-vertex are stored here in npy files
  PREDICTED_DATA = '/masif/data/masif_site/output/all_feat_3l/pred_data'

  def __init__(self, client: docker.DockerClient, gpu_manager: CudaDeviceManager,
               input_folder: str, output_folder: str, image: str = DEFAULT_IMAGE,
               out_logfile: str = None, err_logfile: str = None):
    super().__init__(
      image=image,
      entrypoint='/bin/bash',
      volumes={
        input_folder: MasifContainer.INPUT_FOLDER,
        output_folder: MasifContainer.OUTPUT_FOLDER
      },
      client=client,
      gpu_manager=gpu_manager
    )
    self.out_logfile = out_logfile
    self.err_logfile = err_logfile

  def run(self) -> bool:
    container, device = super()._create_container()
    with ContainerLogs(self.out_logfile, self.err_logfile) as logs:
      print(f"Running MaSIF on device {device.name}", file=logs.out)
      res = True
      for cmd in self.__get_commands():
        res &= super()._execute_command(container, cmd, output_log=logs.out, error_log=logs.err)
        if not res:
          break
    super()._stop_container(container)
    return res

  def __get_commands(self) -> List[str]:
    return [
      f'/bin/bash {self.PREPARE_LIST_SCRIPT} {self.INPUT_FOLDER}/list.txt {self.INPUT_FOLDER}',
      f"/bin/bash -c '/bin/ls {self.PRECOMPUTED_FOLDER} > {self.INPUT_FOLDER}/pred_list.txt'",
      f'/bin/bash {self.PREDICT_SITE_SCRIPT} -l {self.INPUT_FOLDER}/pred_list.txt',
      f'/bin/rm {self.INPUT_FOLDER}/pred_list.txt',
      f'/bin/cp -r {self.PREDICTED_DATA} {self.OUTPUT_FOLDER}/pred_data',
      f'/bin/cp -r {self.PRECOMPUTED_FOLDER} {self.OUTPUT_FOLDER}/meshes',
      f'/bin/chmod 0777 --recursive {self.OUTPUT_FOLDER}'
    ]


