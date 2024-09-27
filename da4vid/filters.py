from typing import List

from da4vid.metrics import rog
from da4vid.model import Protein


def filter_by_rog(proteins: List[Protein], cutoff: int = 50, percentage: bool = False, device: str = 'cpu'):
  # Checking parameters
  if cutoff <= 0:
    raise ValueError(f'Invalid cutoff: {cutoff}')
  if percentage and cutoff > 100:
    raise ValueError(f'Invalid percentage cutoff: {cutoff}')
  without_rog = [protein for protein in proteins if 'rog' not in protein.props.keys()]
  # Evaluating rog for missing proteins
  rog(without_rog, device)
  proteins.sort(key=lambda p: p.props['rog'])
  num_retained = int(cutoff*len(proteins)/100) if percentage else cutoff
  return proteins[:num_retained]
