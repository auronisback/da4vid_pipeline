from typing import Tuple, Union, List

import torch

from da4vid.model.proteins import Protein


def evaluate_rmsd(first: Union[Protein, List[Protein]], second: Union[Protein, List[Protein]],
                  device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Evaluates the Root Mean Square Deviation (RMSD), the rotation matrix and
  the translation vector between sets of proteins. If both first and second
  are single proteins, the outputs are evaluated between them. If the first
  is a single protein and the second a list of proteins or vice-versa, then
  the outputs are returned between the single protein and each other proteins.
  If both first and second are list of proteins, the RMSD score, rotation matrix
  and translation vectors are returned between each protein in the first list
  and each protein in the second list.
  :param first: The first protein or list of proteins
  :param second: The second protein or list of proteins
  :param device: The device on which perform operations
  :return: Three tensors, in which the first represent RMSD values, the
           second the rotation matrices and the third are the translation
           vectors. Specifically:
           - In one-vs-one scenarios, the RMSD is a 1x1 tensor, the rotation
             matrix is a 3x3 tensor and the translation vector is 1x3 tensor
           - In one-vs-all scenarios, the RMSD is a 1xN tensor, the rotation
             matrices are in a Nx3x3 tensor and the translation vectors are
             in a Nx3 tensor, with N number of protein in the list
           - In all-vs-all scenarios, the RMSD is a MxN tensor, the rotation
             matrices are in a MxNx3x3 tensors and the translation vectors
             are in a MxNx3 tensor
  :raise ValueError: When the proteins do not share the same number of atoms
  """
  # Checking 1-vs-1 case
  if isinstance(first, Protein) and isinstance(second, Protein):
    return __rmsd_one_vs_one(first, second, device)
  elif isinstance(first, Protein):  # 1-vs-all
    return __rmsd_one_vs_all(first, second, device)
  elif isinstance(second, Protein):  # all-vs-1
    return __rmsd_one_vs_all(second, first, device)
  else:  # all-vs-all
    return __rmsd_all_vs_all(first, second, device)


def __rmsd_one_vs_one(first: Protein, second: Protein,
                      device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  err, R, t = kabsch(first.ca_coords(), second.ca_coords(), device=device)
  return err.squeeze().to(device), R.squeeze().to(device), t.squeeze().to(device)


def __rmsd_one_vs_all(first: Protein, second: List[Protein],
                      device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  Y = torch.stack([p.ca_coords() for p in second])
  err, R, t = kabsch(first.ca_coords(), Y, device=device)
  return err.squeeze().to(device), R.squeeze().to(device), t.squeeze().to(device)


def __rmsd_all_vs_all(first: List[Protein], second: List[Protein],
                      device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  X = torch.stack([p.ca_coords() for p in first])
  Y = torch.stack([p.ca_coords() for p in second])
  err, R, t = kabsch(X, Y, device=device)
  return err.squeeze().to(device), R.squeeze().to(device), t.squeeze().to(device)


def kabsch(X: torch.Tensor, Y: torch.Tensor,
           device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Computes the Kabsch algorithm to find the best rotation matrix and translation vector
  between the two set of points. The minimum value of n should be > 2, or the results can
  be ambiguous.
  :param X: The first set of points, with shape (N_x, n, 3)
  :param Y: The second set of points, with shape (N_y, n, 3)
  :param device: The device on which execute the evaluation. Defaults to 'cpu'.
  :return: The rotation matrix and the translation vector with the optimal affine transformation
           to superimpose x and y. Rotation matrix will have shape (N_x, N_y, 3, 3) and the
           translation vector is embedded in a tensor (N_x, N_y, n, 3). If rmsd is True, then
           a (N_x, N_y) matrix will be produced, where its i,j-th element is the RMSD between
           structure X_i and structure Y_i
  :raise ValueError: When input shapes do not match in all but the batch dimension
  """
  if X.dim() == 2 and Y.dim() == 2:  # Batch is one
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0)
  if X.dim() == Y.dim() - 1:
    X = X.unsqueeze(0)
  if X.shape[1:] != Y.shape[1:]:
    raise ValueError(f"Shapes must match in all but batch dimension (x: {X.shape}, y: {Y.shape})")

  # Unsqueezing X and Y to ensure evaluation of pairs
  X = X.unsqueeze(1)  # N_x x 1 x n x 3
  Y = Y.unsqueeze(0)  # 1 x N_y x n x 3

  # Evaluating centroids and centering points
  centroid_X = torch.mean(X, dim=-2, keepdim=True)
  centroid_Y = torch.mean(Y, dim=-2, keepdim=True)

  # Rotation: centering points
  X_c = X - centroid_X
  Y_c = Y - centroid_Y
  # Rotation: covariance matrices
  H = torch.matmul(X_c.transpose(-2, -1), Y_c)  # N_x x N_y x 3 x 3
  # Rotation: SVD decomposition
  U, S, Vt = torch.linalg.svd(H)  # N_x x N_y x 3 x 3
  # Rotation: correcting reflection of shape N_x x N_y
  d = torch.linalg.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))
  flip_mask = d < 0
  if flip_mask.any():
    Vt[flip_mask, :, -1] *= -1  # Changing sign of last column

  # Rotation: evaluating
  R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))

  # Evaluating translation by rotating the first set and calculating centroid distance
  t = (centroid_Y - torch.matmul(X, R.transpose(-2, -1))
       .mean(dim=-2, keepdim=True)).squeeze()  # Removing dummy column dimension

  # Evaluating RMSD
  rmsd_eval = torch.sqrt(
    torch.sum(
      torch.square(torch.matmul(X_c, R.transpose(-2, -1)) - Y_c), dim=(-2, -1)) / X.shape[-2])
  return rmsd_eval, R, t
