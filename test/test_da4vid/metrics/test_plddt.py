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
    proteins = read_pdb_folder(in_folder, b_fact_prop='plddt')
    plddt = evaluate_plddt(proteins)
    self.assertEqual(3, plddt.shape[0], msg='UInvalid number of returned pLDDT values')
    ground_truth = torch.tensor([79.3916, 85.5686, 83.7320])
    for i in range(3):
      torch.testing.assert_close(ground_truth[i], plddt[i], atol=1e-4, rtol=1e-7,
                                 msg='Mean pLDDT is not close enough')

  def test_plddt_returns_nan_when_props_not_set(self):
    in_pdb = f'{RESOURCES_ROOT}/plddt_test/plddt_test_1.pdb'
    protein = read_from_pdb(in_pdb, b_fact_prop='absolutely_not_plddt')
    plddt = evaluate_plddt(protein)
    self.assertTrue(torch.isnan(plddt), 'Returned pLDDT value is not NaN')


if __name__ == '__main__':
  unittest.main()
