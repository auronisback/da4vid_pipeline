import os.path
import unittest

from da4vid.io import read_protein_mpnn_fasta, read_pdb_folder
from da4vid.io.fasta_io import write_fasta, read_fasta
from test.cfg import RESOURCES_ROOT


class TestFastaIO(unittest.TestCase):

  def test_read_fasta_raises_error_if_file_does_not_exists(self):
    with self.assertRaises(FileNotFoundError):
      read_fasta('unknown_file.fa')

  def test_read_fasta_raises_error_if_not_regular_file(self):
    folder = f'{RESOURCES_ROOT}/io_test_folder_fasta'
    with self.assertRaises(FileExistsError):
      read_fasta(folder)

  def test_read_fasta_succeeds(self):
    fasta_file = f'{RESOURCES_ROOT}/io_test_folder_fasta/fasta.fa'
    proteins = read_fasta(fasta_file)
    self.assertEqual(4, len(proteins))
    self.assertEqual('demo_input_0', proteins[0].name)
    self.assertEqual('GGGGGGGGGGGGGGGGGGGGGGGGGKGSGSTANLGGGGGGGGGGGGGG', proteins[0].sequence())
    self.assertEqual('demo_input_1', proteins[1].name)
    self.assertEqual('GTVACSGGAAGDIPYGSATGANSDGKGSGSTANLTGGGGSVYFWGGCE', proteins[1].sequence())
    self.assertEqual('demo_input_2', proteins[2].name)
    self.assertEqual('SVLGCSSDGRDGPVLTSVGDPAGCGKGSGSTANCLTGGTGANCDSDCG', proteins[2].sequence())
    self.assertEqual('demo_input_3', proteins[3].name)
    self.assertEqual('AVYSLGPISGQCAEMPLAGEPAIGGKGSGSTANFMSGGPVICNPPTAG', proteins[3].sequence())

  def test_read_pmpnn_fasta_raises_error_if_file_does_not_exist(self):
    with self.assertRaises(FileNotFoundError):
      read_protein_mpnn_fasta('unknown_file.fa')

  def test_read_pmpnn_fasta_raise_error_if_file_is_directory(self):
    folder = f'{RESOURCES_ROOT}/io_test_folder_fasta'
    with self.assertRaises(FileExistsError):
      read_protein_mpnn_fasta(folder)

  def test_read_pmpnn_fasta(self):
    fasta_input = f'{RESOURCES_ROOT}/io_test_folder_fasta/pmpnn_fasta_1.fa'
    proteins = read_protein_mpnn_fasta(fasta_input)
    self.assertEqual(11, len(proteins), 'Invalid number of returned proteins')
    original = proteins[0]
    self.assertEqual('demo_input_0', original.name, 'Invalid protein name')
    orig_pmpnn_props = {
      'score': 1.7197,
      'global_score': 1.9520,
      'model_name': 'v_48_020',
      'seed': 993
    }
    self.assertIn('protein_mpnn', original.props.dict, 'protein_mpnn prop not found')
    self.assertDictEqual(orig_pmpnn_props, original.props.dict['protein_mpnn'], 'Invalid props')
    self.assertEqual('GGGGGGGGGGGGGGGGGGGGGGGGGKGSGSTANLGGGGGGGGGGGGGG', original.sequence())
    sampled = proteins[2]
    self.assertEqual('demo_input_0_2', sampled.name, 'Sampled name is invalid')
    sampled_pmpnn_props = {
      'T': 0.5,
      'score': 2.3172,
      'global_score': 2.4339,
      'seq_recovery': 0.2308
    }
    self.assertIn('protein_mpnn', sampled.props.dict, 'protein_mpnn prop is not present')
    self.assertDictEqual(sampled_pmpnn_props, sampled.props.dict['protein_mpnn'], 'Invalid props in sampled')
    self.assertEqual('SVLGCSSDGRDGPVLTSVGDPAGCGKGSGSTANCLTGGTGANCDSDCG', sampled.sequence(), 'Invalid sampled sequence')

  def test_fasta_write(self):
    protein_folder = f'{RESOURCES_ROOT}/io_test_folder_fasta'
    proteins = read_pdb_folder(protein_folder)
    fasta_out = f'{RESOURCES_ROOT}/io_test_folder_fasta/out.fa'
    write_fasta(proteins, fasta_out, overwrite=True)
    self.assertTrue(os.path.isfile(fasta_out), 'FASTA file has not been created')
    os.unlink(fasta_out)

  def test_fasta_write_fails_if_file_already_present(self):
    protein_folder = f'{RESOURCES_ROOT}/io_test_folder_fasta'
    proteins = read_pdb_folder(protein_folder)
    fasta_out = f'{RESOURCES_ROOT}/io_test_folder_fasta/out.fa'
    with open(fasta_out, 'w') as f:
      f.write('Some dummy text')
    with self.assertRaises(FileExistsError): 
      write_fasta(proteins, fasta_out)
    # TODO: Check consistency (maybe read back FASTA and compare sequences?)
    os.unlink(fasta_out)
      
  def test_fasta_write_overwrite_output_if_specified(self):
    protein_folder = f'{RESOURCES_ROOT}/io_test_folder_fasta'
    proteins = read_pdb_folder(protein_folder)
    fasta_out = f'{RESOURCES_ROOT}/io_test_folder_fasta/out.fa'
    with open(fasta_out, 'w') as f:
      f.write('Some dummy text')
    write_fasta(proteins, fasta_out, overwrite=True)
    os.unlink(fasta_out)
    
  def test_fasta_write_fails_if_output_is_a_directory(self):
    protein_folder = f'{RESOURCES_ROOT}/io_test_folder_fasta'
    proteins = read_pdb_folder(protein_folder)
    fasta_out = f'{RESOURCES_ROOT}/io_test_folder_fasta'
    with self.assertRaises(FileExistsError):
      write_fasta(proteins, fasta_out, overwrite=True)


if __name__ == '__main__':
  unittest.main()
