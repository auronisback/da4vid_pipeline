import unittest

import torch

from da4vid.metrics import evaluate_plddt
from da4vid.io.pdb_io import read_from_pdb, read_pdb_folder
from test.cfg import RESOURCES_ROOT


class PlddtTest(unittest.TestCase):

  def test_plddt_on_single_protein(self):
    in_pdb = f'{RESOURCES_ROOT}/plddt_test/plddt_test_1.pdb'
    protein = read_from_pdb(in_pdb, b_fact_prop='plddt')
    plddt = evaluate_plddt(protein)
    torch.testing.assert_close(79.3916, plddt.item(), atol=1e-4, rtol=1e-7,
                               msg='Mean pLDDT is not close enough')

  def test_plddt_on_multiple_proteins(self):
    in_folder = f'{RESOURCES_ROOT}/plddt_test/'
    proteins = read_pdb_folder(in_folder, b_fact_prop='omegafold.plddt')
    plddt = evaluate_plddt(proteins, 'omegafold.plddt')
    self.assertEqual(3, plddt.shape[0], msg='Invalid number of returned pLDDT values')
    ground_truth = torch.tensor([79.3916, 85.5686, 83.7320])
    for i in range(3):
      torch.testing.assert_close(ground_truth[i], plddt[i], atol=1e-4, rtol=1e-7,
                                 msg=f'Mean pLDDT is not close enough: exp {ground_truth[i]} and act {plddt[i]}')
    torch.testing.assert_close(ground_truth[0], proteins[0].get_prop('omegafold.plddt'))

  def test_plddt_returns_nan_when_props_not_set(self):
    in_pdb = f'{RESOURCES_ROOT}/plddt_test/plddt_test_1.pdb'
    protein = read_from_pdb(in_pdb, b_fact_prop='absolutely_not_plddt')
    plddt = evaluate_plddt(protein, 'alphafold.plddt')
    self.assertTrue(torch.isnan(plddt), 'Returned pLDDT value is not NaN')
    self.assertIsNone(protein.get_prop('alphafold.plddt'), f'Property was unexpectedly set: {protein.props}')

  def test_plddt_one_protein_with_plddt_already(self):
    in_folder = f'{RESOURCES_ROOT}/plddt_test/'
    proteins = read_pdb_folder(in_folder, b_fact_prop='plddt')
    proteins[0].add_prop('plddt', torch.tensor(79.3916))
    plddt = evaluate_plddt(proteins)
    self.assertEqual(3, plddt.shape[0], msg='Invalid number of returned pLDDT values')
    ground_truth = torch.tensor([proteins[0].get_prop('plddt'), 85.5686, 83.7320])
    for i in range(3):
      torch.testing.assert_close(ground_truth[i], plddt[i], atol=1e-4, rtol=1e-7,
                                 msg='Mean pLDDT is not close enough')


if __name__ == '__main__':
  unittest.main()
