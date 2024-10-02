import unittest

from test.cfg import RESOURCES_ROOT

from da4vid.io import read_protein_mpnn_fasta


class TestFastaIO(unittest.TestCase):

  def test_read_pmpnn_fasta_raise_error_if_file_does_not_exist(self):
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
    self.assertIn('protein_mpnn', original.props, 'protein_mpnn prop not found')
    self.assertDictEqual(orig_pmpnn_props, original.props['protein_mpnn'], 'Invalid props')
    self.assertEqual('GGGGGGGGGGGGGGGGGGGGGGGGGKGSGSTANLGGGGGGGGGGGGGG', original.sequence())
    sampled = proteins[2]
    self.assertEqual('demo_input_0_2', sampled.name, 'Sampled name is invalid')
    sampled_pmpnn_props = {
      'T': 0.5,
      'score': 2.3172,
      'global_score': 2.4339,
      'seq_recovery': 0.2308
    }
    self.assertIn('protein_mpnn', sampled.props, 'protein_mpnn prop is not present')
    self.assertDictEqual(sampled_pmpnn_props, sampled.props['protein_mpnn'], 'Invalid props in sampled')
    self.assertEqual('SVLGCSSDGRDGPVLTSVGDPAGCGKGSGSTANCLTGGTGANCDSDCG', sampled.sequence(), 'Invalid sampled sequence')


if __name__ == '__main__':
  unittest.main()
