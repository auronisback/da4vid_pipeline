from typing import List, Union

import torch, pydssp

from da4vid.model import Protein


def dssp(proteins: Union[List[Protein], Protein]) -> Union[str, List[str]]:
  """
  Evaluates the Dictionary of Secondary Structure of Protein (DSSP) for a given
  protein.
  :param proteins: A single protein or a list of proteins
  :return: A string with secondary structures code if only one protein was given,
           or a list of such strings for each input protein
  """
  if isinstance(proteins, Protein):
    proteins = [proteins]
  batches = []
  for protein in proteins:
    coords = []
    residues = [residue for chain in protein.chains for residue in chain.residues]
    for residue in residues:
      coords.append([[*atom.coords] for atom in residue.get_backbone_atoms()])
    batches.append(coords)
  batches = torch.stack([torch.tensor(prot) for prot in batches])
  assignments = pydssp.assign(batches, out_type='c3')
  if len(assignments) == 1:
    return ''.join(assignments[0])
  return [''.join(assignments[i]) for i in range(len(assignments))]


def count_secondary_structures(proteins: Union[Protein, List[Protein]]) -> Union[int, List[int]]:
  """
  Counts the number of secondary structures in a single protein or
  in a set of proteins.
  :param proteins: A single protein or a list of proteins
  :return: A number or a list of numbers of secondary structures
           in the input proteins
  """
  ss_seqs = dssp(proteins)
  if isinstance(ss_seqs, str):
    return __count_ss_from_seq(ss_seqs)
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
