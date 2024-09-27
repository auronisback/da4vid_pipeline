import math
from typing import List

from da4vid.metrics import rog, count_secondary_structures
from da4vid.model import Protein


def __check_cutoff(cutoff: int, percentage: bool):
  # Checking parameters
  if cutoff <= 0:
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
  ss_num = count_secondary_structures(proteins, device)
  # Adding 'ss' prop to proteins
  for ss, protein in zip(ss_num, proteins):
    protein.props['ss'] = ss
  filtered = list(filter(lambda p: p.props['ss'] >= threshold, proteins))
  filtered.sort(key=lambda p: p.props['ss'], reverse=True)
  num_retained = int(cutoff*len(proteins)/100) if percentage else cutoff
  return filtered[:num_retained]
