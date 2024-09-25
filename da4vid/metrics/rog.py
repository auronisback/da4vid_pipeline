import os
from typing import List, Tuple, Union

import torch

from da4vid.model import Protein

ATOM_MASSES = torch.Tensor([
  12.0107,  # C
  14.0067,  # N
  15.9994,  # O
  1.00794,  # H
  32.065,  # S
])

ATOM_TO_MASS = {'C': 0, 'N': 1, 'O': 2, 'H': 3, 'S': 4}


def rog(proteins: Union[Protein, List[Protein]], device: str = 'cpu'):
  if isinstance(proteins, Protein):
    proteins = [proteins]
  coords = []
  atoms = []
  for protein in proteins:
    coords.append(protein.coords())
    atoms.append(protein.get_atom_symbols())
  X = torch.stack(coords)
  A = __atoms_to_one_hot(atoms)
  return __rog(X, A, device=device).squeeze()


def __rog(X: torch.Tensor, A: torch.Tensor, device='cpu') -> torch.Tensor:
  """
  Evaluates the radius-of-gyration of the given proteins, expressed with
  atomic coordinates and one-hot encoding of its atoms.
  :param X: Atom coordinates tensor with shape B x N x 3
  :param A: One-hot encoding atom types tensor with shape B x 5
  :param device: The device on which perform the calculation. Defaults to 'cpu'
  :return: A vector with shape B with RoG values for each protein in the batch
  """
  X = X.to(device)
  masses = ATOM_MASSES.to(device)

  assert X.ndim == 3, f"X must be a 3D tensor, found shape {X.shape}"
  assert A.dim() == 3, f"a must be a 3D tensor, found shape {A.shape}"
  assert X.shape[0] == A.shape[0], f"X {X.shape[0]} and a {A.shape[0]} must have the same batch dimension"
  assert X.shape[2] == 3, f"X {X.shape[2]} should have 3D coordinates in its last dimension"
  assert A.shape[2] == 5, f"Invalid one-hot encoding for a: {A.shape[2]}"

  # Evaluating center of mass
  W = (A * masses.t()).max(dim=-1).values  # Tensor of masses
  m_tot = W.sum(dim=1, keepdim=True)
  CoM = (X.mul(W.unsqueeze(2).repeat(1, 1, 3))).sum(dim=1, keepdim=True) / m_tot.unsqueeze(2)
  rr = (X - CoM).square().sum(dim=-1)
  rr = rr.mul(W).sum(dim=-1, keepdim=True)
  rog_2 = rr / m_tot
  return rog_2.sqrt()


def __atoms_to_one_hot(seqs: List[List[str]], device='cpu') -> torch.Tensor:
  """
  Converts a list of atom symbols to the related one-hot encoding.
  :param seqs: The sequence of atom symbols
  :param device: The device on which store the tensor. Defaults to 'cpu'
  :return: A torch.Tensor with the one-hot encoding, on the specified device
  """
  A = torch.zeros([len(seqs), len(seqs[0]), 5]).to(device)
  for i, b in enumerate(seqs):
    for j, elem in enumerate(b):
      if elem is not None:
        A[i, j, ATOM_TO_MASS[elem]] = 1.
  return A
