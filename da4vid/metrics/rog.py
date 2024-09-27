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


def rog(proteins: Union[Protein, List[Protein]], device: str = 'cpu') -> torch.Tensor:
  """
  Evaluates Radius of Gyration of the given protein or proteins.
  :param proteins: A single protein or a list of proteins. A 'rog' prop
                   will be inserted into the proteins
  :param device: the device on which perform the calculation
  :return: A tensor with radii of gyration of the given proteins, or
           an empty tensor if no proteins have been specified
  """
  if isinstance(proteins, Protein):
    proteins = [proteins]
  if len(proteins) == 0:
    return torch.Tensor()  # 0-dimensional tensor
  coords = []
  atoms = []
  for protein in proteins:
    atom_symbols = protein.get_atom_symbols()
    coords.append(protein.coords())
    atoms.append(atom_symbols)
  # Check if proteins can be stacked together in torch
  stackable = True
  shape = coords[0].shape
  for c in coords:
    if shape != c.shape:
      stackable = False
      break
  if stackable:
    X = torch.stack(coords)
    A = __atoms_to_one_hot(atoms)
    radii = __rog(X, A, device=device)
  else:
    radii = []
    # Not stackable, processing proteins one by one
    for atom, coord in zip(atoms, coords):
      radii.append(__rog(coord.unsqueeze(0), __atoms_to_one_hot([atom]), device=device))
    radii = torch.stack(radii)
  for protein, radius in zip(proteins, radii):
    protein.props['rog'] = radius
  return radii.squeeze()


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
  A = A.to(device)
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
