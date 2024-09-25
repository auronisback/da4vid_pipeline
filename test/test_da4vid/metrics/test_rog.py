import unittest

import torch

from da4vid.metrics import rog
from da4vid.model import Residues, Protein, Chain, Atom


class RoGTest(unittest.TestCase):

  def __get_demo_protein_1(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain_A.residues[0].atoms = [Atom(residue=chain_A.residues[0], symbol='C', coords=(1, 0, 0)),
                                 Atom(residue=chain_A.residues[0], symbol='N', coords=(0, 1, 0))]
    chain_A.residues[1].atoms = [Atom(residue=chain_A.residues[1], symbol='H', coords=(0, 0, 1))]
    chain_B.residues[0].atoms = [Atom(residue=chain_B.residues[0], symbol='H', coords=(-1, 0, 0)),
                                 Atom(residue=chain_B.residues[0], symbol='O', coords=(0, -1, 0))]
    chain_B.residues[1].atoms = [Atom(residue=chain_B.residues[1], symbol='C', coords=(0, 0, -1))]
    return protein

  def __get_demo_protein_2(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain_A.residues[0].atoms = [Atom(residue=chain_A.residues[0], symbol='C', coords=(1, 0.5, 0)),
                                 Atom(residue=chain_A.residues[0], symbol='N', coords=(0, 1, 0))]
    chain_A.residues[1].atoms = [Atom(residue=chain_A.residues[1], symbol='O', coords=(1, 0, 1))]
    chain_B.residues[0].atoms = [Atom(residue=chain_B.residues[0], symbol='H', coords=(-1, 0, 0)),
                                 Atom(residue=chain_B.residues[0], symbol='S', coords=(1, -1, 0))]
    chain_B.residues[1].atoms = [Atom(residue=chain_B.residues[1], symbol='C', coords=(0, 1, -1))]
    return protein

  def test_single_protein_rog(self):
    protein = self.__get_demo_protein_1()
    radius = rog(protein)
    torch.testing.assert_close(0.96, radius.item(), atol=1e-4, rtol=1e-6, msg='Invalid RoG')

  def test_multiple_proteins_rog(self):
    proteins = [self.__get_demo_protein_1(), self.__get_demo_protein_2()]
    radii = rog(proteins)
    self.assertEqual(2, radii.shape[0], msg='Invalid shape')
    torch.testing.assert_close(torch.tensor([0.96, 1.1235]), radii, atol=1e-4, rtol=1e-6, msg='Radii do not match')


if __name__ == '__main__':
  unittest.main()
