import math
from typing import List, Dict

from da4vid.metrics import rog, count_secondary_structures
from da4vid.model import Protein


def __check_cutoff(cutoff: int, percentage: bool):
  if cutoff < 0:
    raise ValueError(f'Invalid cutoff: {cutoff}')
  if percentage and cutoff > 100:
    raise ValueError(f'Invalid percentage cutoff: {cutoff}')


def filter_by_rog(proteins: List[Protein], cutoff: int = None, percentage: bool = False,
                  threshold: float = math.inf, device: str = 'cpu'):
  if cutoff is None:
    cutoff = len(proteins) if not percentage else 100
  __check_cutoff(cutoff, percentage)
  without_rog = [protein for protein in proteins if 'rog' not in protein.props.keys()]
  # Evaluating rog for missing proteins
  rog(without_rog, device)
  filtered = list(filter(lambda p: p.props['rog'] <= threshold, proteins))
  filtered.sort(key=lambda p: p.props['rog'])
  num_retained = int(cutoff*len(proteins)/100) if percentage else cutoff
  return filtered[:num_retained]


def filter_by_ss(proteins: List[Protein], cutoff: int = None, percentage: bool = False,
                 threshold: int = 0, device: str = 'cpu'):
  if cutoff is None:
    cutoff = len(proteins) if not percentage else 100
  __check_cutoff(cutoff, percentage)
  __add_ss_props(proteins, device)
  filtered = list(filter(lambda p: p.props['ss'] >= threshold, proteins))
  filtered.sort(key=lambda p: p.props['ss'], reverse=True)
  num_retained = int(cutoff*len(proteins)/100) if percentage else cutoff
  return filtered[:num_retained]


def cluster_by_ss(proteins: List[Protein], threshold: int = 0,
                  device: str = 'cpu') -> Dict[int, List[Protein]]:
  """
  Clusters proteins structures by their number of secondary structures
  and removes those whose number is lesser than a specified threshold
  :param proteins: The list of proteins to cluster
  :param threshold: The optional threshold for removing proteins whose
                    number of SSs is lesser than the threshold.
                    Defaults to 0
  :param device: The device on which run the SS prediction algorithm
  :return: A dictionary whose keys are the number of secondary structures
           and values are lists of proteins with that specific number of SSs
  """
  if threshold < 0:
    raise ValueError(f'Invalid threshold: {threshold}')
  __add_ss_props(proteins, device)
  ss_dict = {}
  for protein in filter(lambda p: p.props['ss'] >= threshold, proteins):
    ss_num = int(protein.props['ss'])
    if ss_num not in ss_dict.keys():
      ss_dict[ss_num] = [protein]
    else:
      ss_dict[ss_num].append(protein)
  return ss_dict


def __add_ss_props(proteins: List[Protein], device: str = 'cpu'):
  ss_num = count_secondary_structures(proteins, device)
  # Adding 'ss' prop to proteins
  for ss, protein in zip(ss_num, proteins):
    protein.props['ss'] = ss
