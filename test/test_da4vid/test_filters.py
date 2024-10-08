import unittest

from da4vid.filters import filter_by_rog, filter_by_ss, cluster_by_ss, filter_by_plddt
from da4vid.io.pdb_io import read_pdb_folder, read_from_pdb
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

  def test_filter_by_rog_with_threshold(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_rog(proteins, threshold=14)
    self.assertEqual(4, len(filtered), 'Number of filtered proteins does not match')
    self.assertEqual('orig', filtered[0].name, 'Protein at index 0 does not match')
    self.assertEqual('sample_1002', filtered[1].name, 'Protein at index 1 does not match')
    self.assertEqual('sample_1007', filtered[2].name, 'Protein at index 2 does not match')
    self.assertEqual('sample_1004', filtered[3].name, 'Protein at index 3 does not match')

  def test_filter_by_rog_with_threshold_and_absolute_cutoff(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_rog(proteins, cutoff=2, threshold=14)
    self.assertEqual(2, len(filtered), 'Number of filtered proteins does not match')
    self.assertEqual('orig', filtered[0].name, 'Protein at index 0 does not match')
    self.assertEqual('sample_1002', filtered[1].name, 'Protein at index 1 does not match')

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

  def test_filter_by_secondary_structures_with_threshold(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_ss(proteins, threshold=4, percentage=True)
    self.assertEqual(2, len(filtered), 'Invalid number of filtered proteins')
    self.assertEqual('orig', filtered[0].name, 'Not valid protein at index 0')
    self.assertEqual('sample_1004', filtered[1].name, 'Not valid protein at index 1')

  def test_filter_by_secondary_structures_with_threshold_and_absolute_cutoff(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    filtered = filter_by_ss(proteins, threshold=3, cutoff=5)
    self.assertEqual(5, len(filtered), 'Invalid number of filtered proteins')
    self.assertEqual('orig', filtered[0].name, 'Not valid protein at index 0')
    self.assertEqual('sample_1004', filtered[1].name, 'Not valid protein at index 1')
    self.assertEqual('sample_1001', filtered[2].name, 'Not valid protein at index 2')
    self.assertEqual('sample_1000', filtered[3].name, 'Not valid protein at index 3')
    self.assertEqual('sample_1008', filtered[4].name, 'Not valid protein at index 4')

  def test_cluster_by_secondary_structures_with_invalid_threshold(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    with self.assertRaises(ValueError):
      cluster_by_ss(proteins, -1)

  def test_cluster_by_secondary_structures_without_threshold(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    clustering = cluster_by_ss(proteins)
    self.assertEqual(5, len(clustering.keys()), 'Invalid number of clusters')
    for i, k in enumerate(clustering.keys()):
      for j, protein in enumerate(clustering[k]):
        self.assertEqual(k, protein.props['ss'], f'Invalid protein {j} in cluster {i}-{k}')

  def test_cluster_by_secondary_structure_with_threshold(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder)
    clustering = cluster_by_ss(proteins, 3)
    self.assertEqual(3, len(clustering.keys()), 'Invalid number of clusters')
    self.assertNotIn(0, clustering.keys())
    self.assertNotIn(1, clustering.keys())
    self.assertNotIn(2, clustering.keys())

  def test_filter_by_plddt_with_invalid_threshold_raises_error(self):
    with self.assertRaises(ValueError):
      filter_by_plddt([], threshold=-1)

  def test_filter_by_plddt_with_invalid_cutoff_raises_error(self):
    with self.assertRaises(ValueError):
      filter_by_plddt([], cutoff=-11)

  def test_filter_by_plddt_with_invalid_percentage_cutoff_raises_error(self):
    with self.assertRaises(ValueError):
      filter_by_plddt([], cutoff=101, percentage=True)

  def test_filter_by_plddt_with_no_proteins(self):
    filtered = filter_by_plddt([], cutoff=20)
    self.assertEqual(0, len(filtered), 'Filtered protein set is not empty?!')

  def test_filter_by_plddt_with_threshold(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder, b_fact_prop='plddt')
    retained_names = ['sample_1000', 'sample_1003', 'sample_1007']
    filtered = filter_by_plddt(proteins, threshold=58.)
    self.assertEqual(3, len(filtered), 'Invalid number fo filtered proteins')
    for i, protein in enumerate(filtered):
      self.assertIn(protein.name, retained_names, f'Protein {i} not present')

  def test_filter_by_plddt_with_absolute_cutoff(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder, b_fact_prop='plddt')
    retained_names = ['sample_1000', 'sample_1003', 'sample_1007', 'sample_1008', 'sample_1004']
    filtered = filter_by_plddt(proteins, cutoff=5)
    self.assertEqual(5, len(filtered), 'Invalid number of filtered proteins')
    for i, true_name, protein in zip(range(5), retained_names, filtered):
      self.assertEqual(true_name, protein.name, f'Invalid protein with index {i}')

  def test_filter_by_plddt_with_percentage_cutoff(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder, b_fact_prop='plddt')
    filtered = filter_by_plddt(proteins, cutoff=25, percentage=True)
    retained_names = ['sample_1000', 'sample_1003']
    self.assertEqual(2, len(filtered), 'Invalid number of filtered proteins')
    for i, true_name, protein in zip(range(2), retained_names, filtered):
      self.assertEqual(true_name, protein.name, f'Invalid protein with index {i}')

  def test_filter_by_plddt_with_cutoff_and_threshold(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder, b_fact_prop='plddt')
    filtered = filter_by_plddt(proteins, cutoff=25, percentage=True, threshold=60.)
    self.assertEqual(1, len(filtered), 'Invalid number of filtered proteins')
    self.assertEqual('sample_1000', filtered[0].name, 'Invalid retained protein')

  def test_filter_by_plddt_with_one_nan_value_raise_error(self):
    pdb_folder = f'{RESOURCES_ROOT}/filter_test'
    proteins = read_pdb_folder(pdb_folder, b_fact_prop='plddt')
    # Appending another proteins without pLDDT
    proteins.append(read_from_pdb(f'{pdb_folder}/orig.pdb'))
    with self.assertRaises(AttributeError):
      filter_by_plddt(proteins, cutoff=5)

if __name__ == '__main__':
  unittest.main()
