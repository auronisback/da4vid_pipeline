from typing import List, Union

import pydssp
import torch

from da4vid.model.proteins import Protein


def dssp(proteins: Union[List[Protein], Protein], device: str = 'cpu') -> Union[str, List[str]]:
  """
  Evaluates the Dictionary of Secondary Structure of Protein (DSSP) for a given
  protein.
  :param proteins: A single protein or a list of proteins
  :param device: The device on which execute operations
  :return: A string with secondary structures code if only one protein was given,
           or a list of such strings for each input protein
  """
  if isinstance(proteins, Protein):
    proteins = [proteins]
  if len(proteins) == 0:
    return []
  batches = []
  for protein in proteins:
    coords = []
    residues = [residue for chain in protein.chains for residue in chain.residues]
    for residue in residues:
      coords.append([[*atom.coords] for atom in residue.get_backbone_atoms()])
    batches.append(coords)
  # Checking if they are stackable
  stackable = True
  n_resi = len(batches[0])
  for batch in batches:
    if len(batch) != n_resi:
      stackable = False
      break
  assignments = []
  if stackable:
    batches = torch.stack([torch.tensor(prot) for prot in batches]).to(device)
    assignments = pydssp.assign(batches, out_type='c3')
  else:  # Not stackable: cycling one by one
    for batch in batches:
      assignment = pydssp.assign(torch.tensor(batch).unsqueeze(0), out_type='c3')
      assignments.append(*assignment)
  if len(assignments) == 1:
    return ''.join(assignments[0])
  return [''.join(assignments[i]) for i in range(len(assignments))]


def count_secondary_structures(proteins: Protein | List[Protein], device: str = 'cpu') -> List[int]:
  """
  Counts the number of secondary structures in a single protein or
  in a set of proteins.
  :param proteins: A single protein or a list of proteins
  :param device: The device on which execute the evaluation
  :return: A number or a list of numbers of secondary structures
           in the input proteins
  """
  ss_seqs = dssp(proteins, device=device)
  if isinstance(ss_seqs, str):
    return [__count_ss_from_seq(ss_seqs)]
  return [__count_ss_from_seq(seq) for seq in ss_seqs]


def __count_ss_from_seq(ss_seq: str) -> int:
  act = '-'
  count = 0
  for c in ss_seq:
    if c != act:
      act = c
      if c != '-':
        count += 1
  return count
