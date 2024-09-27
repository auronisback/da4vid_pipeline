import unittest

from da4vid.filters import filter_by_rog, filter_by_ss
from da4vid.utils.io import read_pdb_folder
from test.cfg import RESOURCES_ROOT, TEST_GPU


class FiltersTest(unittest.TestCase):
  def test_filter_should_raise_error_with_negative_cutoff(self):
    with self.assertRaises(ValueError, msg='Error was never raised'):
      filter_by_rog([], cutoff=-2)

  def test_filter_should_raise_error_with_invalid_percentage_cutoff(self):
    with self.assertRaises(ValueError, msg='Error was never raised'):
      filter_by_rog([], cutoff=111, percentage=True)

  def test_filter_with_absolute_cutoff(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_rog(proteins, 3)
    self.assertEqual(3, len(filtered), 'Number of filtered proteins does not match')
    self.assertEqual('orig', filtered[0].name, 'Invalid filtering')
    self.assertEqual('sample_1002', filtered[1].name, 'Invalid filtering')
    self.assertEqual('sample_1007', filtered[2].name, 'Invalid filtering')

  def test_filter_with_percentage_cutoff(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_rog(proteins, 50, percentage=True)
    self.assertEqual(5, len(filtered), 'Number of filtered proteins does not match')
    self.assertEqual('orig', filtered[0].name, 'Invalid filtering')
    self.assertEqual('sample_1002', filtered[1].name, 'Invalid filtering')
    self.assertEqual('sample_1007', filtered[2].name, 'Invalid filtering')
    self.assertEqual('sample_1004', filtered[3].name, 'Invalid filtering')
    self.assertEqual('sample_1008', filtered[4].name, 'Invalid filtering')

  def test_filter_with_absolute_cutoff_on_gpu(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_rog(proteins, 50, percentage=True, device=TEST_GPU)
    self.assertEqual(5, len(filtered), 'Number of filtered proteins does not match')
    self.assertEqual('orig', filtered[0].name, 'Invalid filtering')
    self.assertEqual('sample_1002', filtered[1].name, 'Invalid filtering')
    self.assertEqual('sample_1007', filtered[2].name, 'Invalid filtering')
    self.assertEqual('sample_1004', filtered[3].name, 'Invalid filtering')
    self.assertEqual('sample_1008', filtered[4].name, 'Invalid filtering')

  def test_filter_by_secondary_structures_with_absolute_cutoff(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_ss(proteins, 3)
    self.assertEqual(3, len(filtered), 'Invalid number of filtered proteins')
    self.assertEqual('orig', filtered[0].name, 'Not valid protein at index 0')
    self.assertEqual('sample_1004', filtered[1].name, 'Not valid protein at index 1')
    self.assertEqual('sample_1001', filtered[2].name, 'Not valid protein at index 2')

  def test_filter_by_secondary_structures_with_percentage_cutoff(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_ss(proteins, 60, percentage=True)
    self.assertEqual(6, len(filtered), 'Invalid number of filtered proteins')
    self.assertEqual('orig', filtered[0].name, 'Not valid protein at index 0')
    self.assertEqual('sample_1004', filtered[1].name, 'Not valid protein at index 1')
    self.assertEqual('sample_1001', filtered[2].name, 'Not valid protein at index 2')
    self.assertEqual('sample_1000', filtered[3].name, 'Not valid protein at index 3')
    self.assertEqual('sample_1008', filtered[4].name, 'Not valid protein at index 4')
    self.assertEqual('sample_1003', filtered[5].name, 'Not valid protein at index 5')

  def test_filter_by_secondary_structures_with_absolute_cutoff_on_gpu(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_ss(proteins, 3, device=TEST_GPU)
    self.assertEqual(3, len(filtered), 'Invalid number of filtered proteins')
    self.assertEqual('orig', filtered[0].name, 'Not valid protein at index 0')
    self.assertEqual('sample_1004', filtered[1].name, 'Not valid protein at index 1')
    self.assertEqual('sample_1001', filtered[2].name, 'Not valid protein at index 2')


if __name__ == '__main__':
  unittest.main()
