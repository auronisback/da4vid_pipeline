import unittest

import torch

from da4vid.metrics import rog
from da4vid.model.proteins import Residues, Protein, Chain, Atom
from test.cfg import TEST_GPU


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
    chain_B = Chain('B', residues=Residues.from_sequence('AAG'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain_A.residues[0].atoms = [Atom(residue=chain_A.residues[0], symbol='C', coords=(1, 0.5, 0)),
                                 Atom(residue=chain_A.residues[0], symbol='N', coords=(0, 1, 0))]
    chain_A.residues[1].atoms = [Atom(residue=chain_A.residues[1], symbol='O', coords=(1, 0, 1))]
    chain_B.residues[0].atoms = [Atom(residue=chain_B.residues[0], symbol='H', coords=(-1, 0, 0)),
                                 Atom(residue=chain_B.residues[0], symbol='S', coords=(1, -1, 0))]
    chain_B.residues[1].atoms = [Atom(residue=chain_B.residues[1], symbol='C', coords=(0, 1, -1))]
    chain_B.residues[2].atoms = [Atom(residue=chain_B.residues[2], symbol='N', coords=(1, 1, 1)),
                                 Atom(residue=chain_B.residues[2], symbol='C', coords=(2, 2, 2))]
    return protein

  def test_rog_with_empty_list(self):
    radius = rog([])
    self.assertEqual(torch.Size([0]), radius.size(), 'Empty tensor not returned!')

  def test_single_protein_rog(self):
    protein = self.__get_demo_protein_1()
    radius = rog(protein)
    ground_truth = 0.96
    torch.testing.assert_close(ground_truth, radius.item(), atol=1e-4, rtol=1e-6,
                               msg='Invalid RoG')
    self.assertTrue(protein.has_prop('rog'), 'rog prop not added to protein')
    torch.testing.assert_close(ground_truth, protein.get_prop('rog').item(), atol=1e-4, rtol=1e-6,
                               msg='Invalid rog prop')

  def test_multiple_stackable_proteins_rog(self):
    proteins = [self.__get_demo_protein_1(), self.__get_demo_protein_1()]
    radii = rog(proteins)
    ground_truths = torch.tensor([0.96, 0.96])
    self.assertEqual(2, radii.shape[0], msg='Invalid shape')
    torch.testing.assert_close(ground_truths, radii, atol=1e-4, rtol=1e-6,
                               msg=f'Radii do not match: Expected {ground_truths}, actual {radii}')
    for i, protein, radius, ground_truth in zip(range(2), proteins, radii, ground_truths):
      self.assertTrue(protein.has_prop('rog'), f'rog prop not added to protein {i}')
      torch.testing.assert_close(ground_truth.item(), protein.get_prop('rog').item(), atol=1e-4, rtol=1e-6,
                                 msg=f'Invalid rog prop for protein {i}: '
                                     f'[Expected] {ground_truth} != Actual {protein.get_prop("rog").item()}')

  def test_multiple_unstackable_proteins_rog(self):
    proteins = [self.__get_demo_protein_1(), self.__get_demo_protein_2()]
    radii = rog(proteins)
    ground_truths = torch.tensor([0.96, 1.4092])
    self.assertEqual(2, radii.shape[0], msg='Invalid shape')
    torch.testing.assert_close(ground_truths, radii, atol=1e-4, rtol=1e-6,
                               msg=f'Radii do not match: Expected {ground_truths}, actual {radii}')
    for i, protein, radius, ground_truth in zip(range(2), proteins, radii, ground_truths):
      self.assertTrue(protein.has_prop('rog'), f'rog prop not added to protein {i}')
      torch.testing.assert_close(ground_truth.item(), protein.get_prop('rog').item(), atol=1e-4, rtol=1e-6,
                                 msg=f'Invalid rog prop for protein {i}: '
                                     f'[Expected] {ground_truth} != Actual {protein.get_prop("rog").item()}')

  def test_rog_evaluation_on_gpu(self):
    proteins = [self.__get_demo_protein_1(), self.__get_demo_protein_2()]
    radii = rog(proteins, device=TEST_GPU)
    ground_truths = torch.tensor([0.96, 1.4092]).to(TEST_GPU)
    self.assertEqual(2, radii.shape[0], msg='Invalid shape')
    torch.testing.assert_close(ground_truths, radii, atol=1e-4, rtol=1e-6,
                               msg=f'Radii do not match: Expected {ground_truths}, actual {radii}')
    for i, protein, radius, ground_truth in zip(range(2), proteins, radii, ground_truths):
      self.assertTrue(protein.has_prop('rog'), f'rog prop not added to protein {i}')
      torch.testing.assert_close(ground_truth.item(), protein.get_prop('rog').item(), atol=1e-4, rtol=1e-6,
                                 msg=f'Invalid rog prop for protein {i}: '
                                     f'[Expected] {ground_truth} != Actual {protein.get_prop("rog").item()}')


if __name__ == '__main__':
  unittest.main()
