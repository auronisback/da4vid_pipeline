from typing import List, Union

import torch

from da4vid.model.proteins import Protein


def evaluate_plddt(proteins: Union[Protein, List[Protein]], plddt_prop: str = 'plddt',
                   device: str = 'cpu') -> torch.Tensor:
  """
  Evaluates pLDDT values for the given proteins, if their atoms or
  residues have a 'plddt' prop.
  :param proteins: A single protein or a list of proteins of which
                   evaluate pLDDT
  :param plddt_prop: The key of plddt_prop to check or assign, in dot notation
  :param device: The device on which evaluate pLDDT calculations
  :return: A tensor with mean pLDDT value for all atoms/residues in
           the given proteins, or NaNs for proteins which do not have
           'plddt' props at atom/residue level
  """
  if isinstance(proteins, Protein):
    proteins = [proteins]
  proteins_plddt = []
  for protein in proteins:
    if protein.has_prop(plddt_prop):
      prev_plddt = protein.get_prop(plddt_prop)
      proteins_plddt.append(prev_plddt.to(device) if isinstance(prev_plddt, torch.Tensor)
                            else torch.tensor(prev_plddt))
    else:
      resi_plddt = []
      for chain in protein.chains:
        for resi in chain.residues:
          if not resi.props.has_key(plddt_prop):  # Evaluating mean pLDDT for residue by its atoms
            atoms_plddt = (torch.tensor([atom.props['plddt'] for atom in resi.atoms
                                         if 'plddt' in atom.props]).to(device))
            resi.props.add_value(plddt_prop, atoms_plddt.nanmean().item())
          resi_plddt.append(resi.props.get_value(plddt_prop))
      mean_plddt = torch.nanmean(torch.tensor(resi_plddt).to(device))
      proteins_plddt.append(mean_plddt)
      # Adding pLDDT to protein if not NaN
      if not torch.isnan(mean_plddt):
        protein.add_prop(plddt_prop, mean_plddt)
  return torch.stack(proteins_plddt).squeeze()
