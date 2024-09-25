from typing import List, Union

import torch

from da4vid.model import Protein


def evaluate_plddt(proteins: Union[Protein, List[Protein]], device: str = 'cpu') -> torch.Tensor:
  """
  Evaluates pLDDT values for the given proteins, if their atoms or
  residues have a 'plddt' prop.
  :param proteins: A single protein or a list of proteins of which
                   evaluate pLDDT
  :param device: The device on which evaluate pLDDT calculations
  :return: A tensor with mean pLDDT value for all atoms/residues in
           the given proteins, or NaNs for proteins which do not have
           'plddt' props at atom/residue level
  """
  if isinstance(proteins, Protein):
    proteins = [proteins]
  proteins_plddt = []
  for protein in proteins:
    resi_plddt = []
    for chain in protein.chains:
      for resi in chain.residues:
        if 'plddt' not in resi.props:  # Evaluating mean pLDDT for residue by its atoms
          atoms_plddt = (torch.tensor([atom.props['plddt'] for atom in resi.atoms if 'plddt' in atom.props]).
                         to(device))
          resi.props['plddt'] = atoms_plddt.nanmean().item()
        resi_plddt.append(resi.props['plddt'])
    proteins_plddt.append(torch.nanmean(torch.tensor(resi_plddt).to(device)))
  return torch.stack(proteins_plddt).squeeze()
