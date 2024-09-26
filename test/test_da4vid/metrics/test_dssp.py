import unittest

from test.cfg import RESOURCES_ROOT
from da4vid.metrics import dssp, count_secondary_structures
from da4vid.utils.io import read_from_pdb, read_pdb_folder


class DsspTest(unittest.TestCase):

  @staticmethod
  def __diff_letters(a, b):
    return sum([a[i] != b[i] for i in range(len(a))])

  def test_dssp_on_single_protein(self):
    in_pdb = f'{RESOURCES_ROOT}/dssp_test/dssp_test_1.pdb'
    protein = read_from_pdb(in_pdb)
    ss = dssp(protein)
    gold_truth = '--EEE-----HHHHHHH-HHHHHHHHHH---HHHHH-EEEE--EEEE--'
    self.assertEqual(len(gold_truth),
                     len(ss), 'Unequal length')
    self.assertTrue(DsspTest.__diff_letters(ss, gold_truth) < 4)

  def test_dssp_on_multiple_proteins(self):
    gold_truths = [
      '--EEE-----HHHHHHH-HHHHHHHHHH---HHHHH-EEEE--EEEE--',
      '-EEEEEE---EEEE--EEEEE-EEEEEE---------EEEE--EEEEEE',
      '----HHH---EEEEEE-------EEE----HHHHHHHH--EEE-HHH--'
    ]
    in_folder = f'{RESOURCES_ROOT}/dssp_test'
    proteins = read_pdb_folder(in_folder)
    sss = dssp(proteins)
    self.assertEqual(3, len(sss), 'Number of returned assignments does not match')
    for i in range(3):
      self.assertEqual(len(gold_truths[i]), len(sss[i]), f'Sequence {i} does not match in length')
      self.assertTrue(self.__diff_letters(gold_truths[i], sss[i]), f'Sequence {i} is too different')

  def test_count_secondary_structures_on_single_protein(self):
    in_pdb = f'{RESOURCES_ROOT}/dssp_test/dssp_test_1.pdb'
    protein = read_from_pdb(in_pdb)
    ss_num = count_secondary_structures(protein)
    self.assertEqual(6, ss_num, 'Number of SS does not match')

  def test_count_secondary_structures_on_multiple_proteins(self):
    in_folder = f'{RESOURCES_ROOT}/dssp_test'
    proteins = read_pdb_folder(in_folder)
    ground_truths = [8, 6, 5]
    counts = count_secondary_structures(proteins)
    self.assertEqual(3, len(counts), 'Number of counts does not match')
    for i, ground_truth, ss_num in zip(range(3), ground_truths, counts):
      self.assertEqual(ground_truth, ss_num, f'Count {i} does not match')


if __name__ == '__main__':
  unittest.main()
