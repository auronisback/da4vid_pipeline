import sys

import torch
import os
from typing import List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASSES = torch.Tensor([
  12.0107,  # C
  14.0067,  # N
  15.9994,  # O
  1.00794,  # H
  32.065,  # S
]).to(device)

ATOM_TO_MASS = {'C': 0, 'N': 1, 'O': 2, 'H': 3, 'S': 4}


def rog(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
  """
  Evaluates the radius-of-gyration of the given proteins, expressed with
  atomic coordinates and one-hot encoding of its atoms
  :param X: Atom coordinates tensor with shape B x N x 3
  :param A: One-hot encoding atom types tensor with shape B x 5
  :return: A vector with shape B with RoG values for each protein in the batch
  """
  assert X.ndim == 3, f"X must be a 3D tensor, found shape {X.shape}"
  assert A.dim() == 3, f"a must be a 3D tensor, found shape {A.shape}"
  assert X.shape[0] == A.shape[0], f"X {X.shape[0]} and a {A.shape[0]} must have the same batch dimension"
  assert X.shape[2] == 3, f"X {X.shape[2]} should have 3D coordinates in its last dimension"
  assert A.shape[2] == 5, f"Invalid one-hot encoding for a: {A.shape[2]}"

  # Evaluating center of mass
  W = (A * MASSES.t()).max(dim=-1).values  # Tensor of masses
  m_tot = W.sum(dim=1, keepdim=True)
  CoM = (X.mul(W.unsqueeze(2).repeat(1, 1, 3))).sum(dim=1, keepdim=True) / m_tot.unsqueeze(2)
  rr = (X - CoM).square().sum(dim=-1)
  rr = rr.mul(W).sum(dim=-1, keepdim=True)
  rog_2 = rr / m_tot
  return rog_2.sqrt()


def atoms_to_one_hot(seqs: List[List[str]], device=device) -> torch.Tensor:
  A = torch.zeros([len(seqs), len(seqs[0]), 5]).to(device)
  for i, b in enumerate(seqs):
    for j, elem in enumerate(b):
      if elem != '':
        A[i, j, ATOM_TO_MASS[elem]] = 1.
  return A


def load_folder(folder: str, device=device) -> Tuple[List[str], torch.Tensor, List[List[str]], torch.Tensor]:
  """

  :param folder:
  :param device:
  :return: A list with names of samples, a tensor with coordinates, a list of atom sequences and a tensor
           with mean pLDDT values for each sample in the batch
  """
  files = [f for f in os.listdir(folder) if f.endswith(".pdb")]
  proteins = []
  seqs = []
  names = []
  plddts = []
  max_n = 0
  for f in files:
    full_path = os.path.join(folder, f)
    coords, seq, plddt = __load_single_protein(full_path)
    if len(coords) > max_n:
      max_n = len(coords)
    proteins.append(coords)
    seqs.append(seq)
    names.append('.'.join(os.path.basename(f).split('.')[:-1]))
    plddts.append(plddt)
  # Creating the batch padding proteins and sequence
  X = []
  padded_seqs = []
  for coords, seq in zip(proteins, seqs):
    Xi = torch.Tensor(coords).to(device)
    if len(coords) < max_n:
      remainder = max_n - len(coords)
      Xi = torch.cat([Xi, torch.zeros(remainder, 3).to(device)], dim=0)
      seq += [''] * remainder
    X.append(Xi)
    padded_seqs.append(seq)
  return names, torch.stack(X), padded_seqs, __mean_plddt(plddts)


def __load_single_protein(filename: str) -> Tuple[List[List[float]], List[str], List[float]]:
  X = []
  seq = []
  plddts = []
  with open(filename, "r") as f:
    for line in f:
      if line.startswith("ATOM"):
        X.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        seq.append(line[13:14].strip())
        plddts.append(float(line[60:66]))
  return X, seq, plddts


def __mean_plddt(plddts: List[List[float]], device=device) -> torch.Tensor:
  ps = []
  for plddt in plddts:
    ps.append(torch.Tensor(plddt).to(device).mean(-1).unsqueeze(0))
  return torch.cat(ps)


def load_protein(filename: str, device=device) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
  coords, seq, plddt = __load_single_protein(filename)
  return torch.Tensor(coords).to(device), seq, __mean_plddt([plddt])


def write_csv(names: List[str], rogs: torch.Tensor, plddts: torch.Tensor, filename: str) -> None:
  with open(filename, "w") as f:
    f.write("Name;RoG;pLDDT\n")
    for name, r, plddt in zip(names, rogs, plddts):
      f.write(f"{name};{r.item()};{plddt.item()}\n")
    f.flush()


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <folder_with_pdb_structures>')
    exit(1)
  folder = sys.argv[1]
  if not os.path.isdir(folder):
    print(f'Folder {folder} does not exists.')
    exit(1)

  output_file = f"{folder}.csv"
  names, X, seqs, plddts = load_folder(folder)
  A = atoms_to_one_hot(seqs)
  rogs = rog(X, A)
  write_csv(names, rogs, plddts, output_file)
