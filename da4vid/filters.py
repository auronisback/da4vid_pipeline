import math
from typing import List, Dict, Tuple

import torch

from da4vid.metrics import rog, count_secondary_structures, evaluate_plddt
from da4vid.model.proteins import Protein


def __check_cutoff(cutoff: float, percentage: bool):
  if cutoff < 0:
    raise ValueError(f'Invalid cutoff: {cutoff}')
  if percentage and cutoff > 100:
    raise ValueError(f'Invalid percentage cutoff: {cutoff}')


def filter_by_rog(proteins: List[Protein], cutoff: float = None, percentage: bool = False,
                  threshold: float = math.inf, device: str = 'cpu') -> List[Protein]:
  """
  Filters proteins by their Radii of Gyration (in ascending order) and/or
  a threshold on such radii.
  :param proteins: The list of proteins to filter
  :param cutoff: An absolute or relative (according to the percentage parameter)
                 number of proteins which will be retained
  :param percentage: Flag to check whether the cutoff is absolute or a percentage
  :param threshold: The threshold over which the proteins are discarded
  :param device: The device on which execute the RoG algorithm
  :return: The list of filtered proteins
  :raise ValueError: if the threshold is negative or the cutoff is invalid
  """
  if threshold < 0:
    raise ValueError(f'given RoG threshold is negative: {threshold}')
  if cutoff is None:
    cutoff = len(proteins) if not percentage else 100.
  __check_cutoff(cutoff, percentage)
  without_rog = [protein for protein in proteins if not protein.has_prop('rog')]
  # Evaluating rog for missing proteins
  rog(without_rog, device)
  filtered = list(filter(lambda p: p.get_prop('rog') <= threshold, proteins))
  filtered.sort(key=lambda p: p.get_prop('rog'))
  num_retained = int(cutoff*len(proteins)/100) if percentage else cutoff
  return filtered[:num_retained]


def filter_by_ss(proteins: List[Protein], cutoff: float = None, percentage: bool = False,
                 threshold: int = 0, device: str = 'cpu') -> List[Protein]:
  """
  Filters a list of proteins by their number of secondary structures and/or
  a specific threshold on the number of SSs.
  :param proteins: The list of proteins to filter
  :param cutoff: The absolute or relative number (according to percentage param)
                 to filter proteins, in descending order of SSs number
  :param percentage: Flag to check if cutoff is absolute or a percentage
  :param threshold: The SSs number threshold under which the proteins are discarded
  :param device: The device on which evaluate the inner DSSP algorithm
  :return: The filtered list of proteins
  :raise ValueError: if the threshold is negative, or the cutoff is invalid
  """
  if cutoff is None:
    cutoff = len(proteins) if not percentage else 100.
  __check_cutoff(cutoff, percentage)
  __add_ss_props(proteins, device)
  filtered = list(filter(lambda p: p.get_prop('ss') >= threshold, proteins))
  filtered.sort(key=lambda p: p.get_prop('ss'), reverse=True)
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
  :raise ValueError: if the threshold is negative, or the cutoff is invalid
  """
  if threshold < 0:
    raise ValueError(f'Invalid threshold: {threshold}')
  __add_ss_props(proteins, device)
  ss_dict = {}
  for protein in filter(lambda p: p.get_prop('ss') >= threshold, proteins):
    ss_num = int(protein.get_prop('ss'))
    if ss_num not in ss_dict.keys():
      ss_dict[ss_num] = [protein]
    else:
      ss_dict[ss_num].append(protein)
  return ss_dict


def __add_ss_props(proteins: List[Protein], device: str = 'cpu'):
  ss_num = count_secondary_structures(proteins, device)
  # Adding 'ss' prop to proteins
  for ss, protein in zip(ss_num, proteins):
    protein.add_prop('ss', ss)


def filter_by_plddt(proteins: List[Protein], cutoff: float = None, percentage: bool = False,
                    threshold: float = 0, plddt_prop: str = 'plddt', device: str = 'cpu'):
  """
  Filters a list of proteins according to the pLDDT values for the
  proteins. It needs the "plddt" prop assigned to the protein or to
  residues/atoms in the protein. In this case, the average protein pLDDT
  is evaluated.
  :param proteins: The list of proteins to filter
  :param cutoff: The absolute or relative (according to the percentage parameter)
                 number of retained proteins
  :param percentage: Flag to check whether cutoff is absolute or a percentage
  :param threshold: The threshold below which a protein is discarded
  :param plddt_prop: The key for pLDDT value in properties
  :param device: The device on which average the pLDDT if needed
  :return: The list of filtered proteins
  :raise ValueError: if either the cutoff or the threshold are invalid
  :raise AttributeError: if any of the proteins has a pLDDT score equal to NaN
  """
  if threshold < 0:
    raise ValueError(f'Invalid pLDDT threshold: {threshold}')
  if cutoff is None:
    cutoff = len(proteins) if not percentage else 100.
  __check_cutoff(cutoff, percentage)
  if not proteins:
    return []
  proteins_plddt = evaluate_plddt(proteins, plddt_prop=plddt_prop, device=device)
  if torch.any(torch.isnan(proteins_plddt)):
    raise AttributeError('at least one protein has a NaN pLDDT score')
  for protein, plddt in zip(proteins, proteins_plddt):
    protein.add_prop(plddt_prop, plddt.item())
  filtered = list(filter(lambda p: p.get_prop(plddt_prop) >= threshold, proteins))
  filtered.sort(key=lambda p: p.get_prop(plddt_prop), reverse=True)
  num_retained = int(cutoff*len(proteins)/100) if percentage else cutoff
  return filtered[:num_retained]


def evaluate_interaction_window(protein: Protein, epitope_position: Tuple[int, int],
                                interaction_metric: str, offset: int = 3) -> float:
  """
  Evaluates the softmax interactions between a sliding window around the epitope on the sliding
  window across the whole protein. The window has the same size of the epitope and is slided according
  to the offset parameter. For example, if the epitope is in position (24,33) and the offset is 3, the
  window will have a size of 9 residues, and the considered windows will be 21-30, 22-31, ..., 24-33, 25-24,
  ..., 26-35, 27-36. A softmax between these windows and a sliding window across all residues will be then
  evaluated, and its result returned.
  :param protein: The protein whose interaction window has to be evaluated
  :param epitope_position: A tuple with the starting index and the ending index of the epitope
  :param interaction_metric: The metric referring to a per-residue interaction score, which will be
                             used as a key in protein residues' property dictionary
  :param offset: The offset around which the window should slide.
  :return: The softmax between the sum of the interaction window around the epitope on all windows
           across the whole protein
  """
  pass
