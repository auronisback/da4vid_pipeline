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
