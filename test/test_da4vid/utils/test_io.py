import unittest

import torch

from test.cfg import RESOURCES_ROOT

from da4vid.utils.io import read_from_pdb, read_pdb_folder


class IOTest(unittest.TestCase):

  def __check_io_test_1_protein(self, protein, b_fact_prop):
    self.assertEqual('io_test_1', protein.name, 'Name does not match')
    self.assertEqual(f'{RESOURCES_ROOT}/io_test_folder_pdb/io_test_1.pdb', protein.file, 'Filepath does not match')
    self.assertEqual(1, len(protein.chains), 'Number of chains does not match')
    chain = protein.chains[0]
    self.assertEqual('A', chain.name, 'Chain name does not match')
    self.assertEqual(1, len(chain.residues), 'Number of residues in chain does not match')
    residue = chain.residues[0]
    self.assertEqual(29, residue.number, 'Residue number does not match')
    self.assertEqual('THR', residue.get_three_letters_code(), 'Residue code does not match')
    self.assertEqual(8, len(residue.atoms), 'Atom number does not match')
    atom = residue.atoms[0]
    self.assertTrue(b_fact_prop in atom.props, 'Temperature property not found')
    self.assertEqual(239, atom.number, 'Atom number does not match')
    self.assertEqual('N', atom.symbol, 'Symbol does not match')
    atom_coords = torch.tensor([
      [3.391, 19.940, 12.762],
      [2.014, 19.761, 13.283],
      [0.826, 19.943, 12.332],
      [0.932, 19.600, 11.133],
      [1.845, 20.667, 14.505],
      [1.214, 21.893, 14.153],
      [3.180, 20.968, 15.185],
      [-0.317, 20.109, 12.824],
    ])
    torch.testing.assert_close(atom_coords, chain.coords())

  def __check_io_test_2_protein(self, protein, b_fact_prop):
    self.assertEqual('io_test_2', protein.name, 'Name does not match')
    self.assertEqual(f'{RESOURCES_ROOT}/io_test_folder_pdb/io_test_2.pdb', protein.file, 'Filepath does not match')
    self.assertEqual(2, len(protein.chains), 'Number of chains does not match')
    chain_A = protein.chains[0]
    self.assertEqual('A', chain_A.name, 'Chain name does not match')
    self.assertEqual(4, len(chain_A.residues), 'Number of residues in chain does not match')
    residue = chain_A.residues[0]
    self.assertEqual(1, residue.number, 'Residue number does not match')
    self.assertEqual('HIS', residue.get_three_letters_code(), 'Residue code does not match')
    self.assertEqual(10, len(residue.atoms), 'Atom number does not match')
    atom = residue.atoms[0]
    self.assertTrue(b_fact_prop in atom.props, 'B-factor property not found')
    self.assertEqual(1, atom.number, 'Atom number does not match')
    self.assertEqual('N', atom.symbol, 'Symbol does not match')
    atom_coords = torch.tensor([
      [49.668, 24.248, 10.436],
      [50.197, 25.578, 10.784],
      [49.169, 26.701, 10.917],
      [48.241, 26.524, 11.749],
      [51.312, 26.048, 9.843],
      [50.958, 26.068, 8.340],
      [49.636, 26.144, 7.860],
      [51.797, 26.043, 7.286],
      [49.691, 26.152, 6.454],
      [51.046, 26.090, 6.098],
      [49.788, 27.850, 10.784],
      [49.138, 29.147, 10.620],
      [47.713, 29.006, 10.110],
      [46.740, 29.251, 10.864],
      [49.875, 29.930, 9.569],
      [49.145, 31.057, 9.176],
      [47.620, 28.367, 8.973],
      [46.287, 28.193, 8.308],
      [45.406, 27.172, 8.963],
      [3.391, 19.940, 12.762],
      [2.014, 19.761, 13.283],
      [0.826, 19.943, 12.332],
      [0.932, 19.600, 11.133],
      [1.845, 20.667, 14.505],
      [1.214, 21.893, 14.153],
      [3.180, 20.968, 15.185],
      [-0.317, 20.109, 12.824],
      [9.143, -20.582, 1.231],
      [8.824, -20.084, -0.109],
      [9.440, -20.964, -1.190],
      [9.768, -22.138, -0.985],
      [9.314, -18.642, -0.302],
      [8.269, -17.606, 0.113],
      [10.683, -18.373, 0.331]
    ])
    torch.testing.assert_close(atom_coords, protein.coords())

  def test_read_pdb_raises_error_if_file_not_found(self):
    unknown_pdb = './unknown'
    with self.assertRaises(FileNotFoundError):
      read_from_pdb(unknown_pdb)

  def test_read_pdb_raises_error_if_folder_is_supplied(self):
    folder = f'{RESOURCES_ROOT}/io_test_folder_pdb'
    with self.assertRaises(ValueError):
      read_from_pdb(folder)

  def test_read_folder_raises_error_if_folder_does_not_exists(self):
    unknown_folder = './unknown'
    with self.assertRaises(FileNotFoundError):
      read_pdb_folder(unknown_folder)

  def test_read_folder_raises_error_if_file_is_supplied(self):
    pdb_in = f'{RESOURCES_ROOT}/io_test_folder_pdb/io_test_1.pdb'
    with self.assertRaises(ValueError):
      read_pdb_folder(pdb_in)

  def test_read_single_pdb(self):
    pdb_in = f'{RESOURCES_ROOT}/io_test_folder_pdb/io_test_1.pdb'
    protein = read_from_pdb(pdb_in)
    self.__check_io_test_1_protein(protein, 'temperature')

  def test_read_pdb_folder(self):
    folder = f'{RESOURCES_ROOT}/io_test_folder_pdb'
    proteins = read_pdb_folder(folder, b_fact_prop='b_factor')
    self.assertEqual(6, len(proteins), 'Read proteins number does not match')
    prot_dict = {protein.name: protein for protein in proteins}
    self.assertIn('io_test_1', prot_dict.keys(), 'Unable to found protein io_test_1')
    self.__check_io_test_1_protein(prot_dict['io_test_1'], 'b_factor')
    self.assertIn('io_test_2', prot_dict.keys(), 'Unable to found protein io_test_2')
    self.__check_io_test_2_protein(prot_dict['io_test_2'], 'b_factor')

  def test_read_single_pdb_with_atom_sanitizing(self):
    pdb_in = f'{RESOURCES_ROOT}/io_test_folder_pdb/orig.pdb'
    protein = read_from_pdb(pdb_in)
    for i, a in enumerate(protein.get_atom_symbols()):
      self.assertNotEqual('', a, f'Invalid symbol read at position {i}')
