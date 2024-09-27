from typing import List

from da4vid.metrics import rog, count_secondary_structures
from da4vid.model import Protein


def __check_cutoff(cutoff: int, percentage: bool):
  # Checking parameters
  if cutoff <= 0:
    raise ValueError(f'Invalid cutoff: {cutoff}')
  if percentage and cutoff > 100:
    raise ValueError(f'Invalid percentage cutoff: {cutoff}')


def filter_by_rog(proteins: List[Protein], cutoff: int = 50, percentage: bool = False, device: str = 'cpu'):
  __check_cutoff(cutoff, percentage)
  without_rog = [protein for protein in proteins if 'rog' not in protein.props.keys()]
  # Evaluating rog for missing proteins
  rog(without_rog, device)
  proteins.sort(key=lambda p: p.props['rog'])
  num_retained = int(cutoff*len(proteins)/100) if percentage else cutoff
  return proteins[:num_retained]


def filter_by_ss(proteins: List[Protein], cutoff: int = 50, percentage: bool = False, device: str = 'cpu'):
  __check_cutoff(cutoff, percentage)
  ss_num = count_secondary_structures(proteins, device)
  # Adding 'ss' prop to proteins
  for ss, protein in zip(ss_num, proteins):
    protein.props['ss'] = ss
  proteins.sort(key=lambda p: p.props['ss'], reverse=True)
  num_retained = int(cutoff*len(proteins)/100) if percentage else cutoff
  return proteins[:num_retained]
