import unittest

import torch

from da4vid.model import Residue, Chain, Residues, Protein, Atom, Proteins


class ResidueTest(unittest.TestCase):

  def test_one_to_three_letters_convertion(self):
    resi = Residue(1, code='A')
    self.assertEqual(resi.get_three_letters_code(), 'ALA', 'Wrong 3-letters conversion')

  def test_three_to_one_letter_conversion(self):
    resi = Residue(1, code='LYS')
    self.assertEqual(resi.get_one_letter_code(), 'K', 'Wrong 1-letter conversion')

  def test_wrong_one_letter_code(self):
    resi = Residue(1)
    with self.assertRaises(ValueError):
      resi.set_code('?')

  def test_wrong_three_letters_code(self):
    resi = Residue(1)
    with self.assertRaises(ValueError):
      resi.set_code('UNK')

  def test_wrong_length_code(self):
    resi = Residue(1)
    with self.assertRaises(ValueError):
      resi.set_code('<UNKNOWN>')

  def test_residues_from_sequence(self):
    sequence = 'ACDC'
    residues = Residues.from_sequence(sequence)
    self.assertEqual(len(residues), len(sequence), 'Wrong number of residues')
    for i, c in enumerate(sequence):
      self.assertEqual(residues[i].get_one_letter_code(), c, f'Wrong code for residue {i}')

  def test_backbone_atom_recovery(self):
    residue = Residue(1, code='THR', atoms=[
      Atom(code='CA'), Atom(code='N'),
      Atom(code='C'), Atom(code='O'),
      Atom(code='CB'), Atom(code='OG1'),
      Atom(code='CG2'), Atom(code='OXT'),
    ])
    backbone = residue.get_backbone_atoms()
    self.assertEqual(4, len(backbone), 'Invalid backbone length')
    self.assertEqual('N', backbone[0].code)
    self.assertEqual('CA', backbone[1].code)
    self.assertEqual('C', backbone[2].code)
    self.assertEqual('O', backbone[3].code)


class ChainTest(unittest.TestCase):

  def test_sequence_extraction(self):
    residues = [Residue(1, code='A'), Residue(2, code='V'), Residue(3, code='I'),
                Residue(4, code='C'), Residue(5, code='I'), Residue(6, code='I')]
    chain = Chain('A', residues=residues)
    self.assertEqual(chain.sequence(), 'AVICII', 'Wrong sequence obtained from chain')

  def test_chain_coordinates(self):
    chain = Chain('A', residues=Residues.from_sequence('CC'))
    chain.residues[0].atoms = [Atom(residue=chain.residues[0], code='C', coords=(1, 0, 0)),
                               Atom(residue=chain.residues[0], code='N', coords=(0, -1, 0))]
    chain.residues[1].atoms = [Atom(residue=chain.residues[1], code='H', coords=(1, 0, 1))]
    coords = chain.coords()
    self.assertEqual(coords.shape, torch.Size((3, 3)), 'Wrong shape for coordinates tensor')
    torch.testing.assert_close(coords, torch.tensor([[1, 0, 0], [0, -1, 0], [1, 0, 1]]))

  def test_chain_backbone_atom_recovery(self):
    chain = Chain('A', residues=[
      Residue(1, code='SER', atoms=[
        Atom(code='CA'), Atom(code='N'),
        Atom(code='O'), Atom(code='C'),
        Atom(code='CB'), Atom(code='OG')
      ]),
      Residue(2, code='THR', atoms=[
        Atom(code='O'), Atom(code='C'),
        Atom(code='CA'), Atom(code='N'),
        Atom(code='CG2'), Atom(code='OG1'),
        Atom(code='CB'), Atom(code='OXT'),
      ])
    ])
    backbone = chain.get_backbone_atoms()
    self.assertEqual(8, len(backbone), 'Invalid backbone length')
    self.assertEqual('N', backbone[0].code)
    self.assertEqual('CA', backbone[1].code)
    self.assertEqual('C', backbone[2].code)
    self.assertEqual('O', backbone[3].code)
    self.assertEqual('N', backbone[4].code)
    self.assertEqual('CA', backbone[5].code)
    self.assertEqual('C', backbone[6].code)
    self.assertEqual('O', backbone[7].code)


class ProteinTest(unittest.TestCase):

  def test_protein_sequence(self):
    seq1 = 'MADQLTEEQIAEFKEAF'
    seq2 = 'EEFVQMM'
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence(seq1), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence(seq2), protein=protein)
    protein.chains = [chain_A, chain_B]
    self.assertEqual(protein.sequence(), seq1 + seq2, 'Wrong sequences')

  def test_protein_sequence_with_separator(self):
    sep = ':'
    seq1 = 'KSGSTANL'
    seq2 = 'AYEYE'
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence(seq1), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence(seq2), protein=protein)
    protein.chains = [chain_A, chain_B]
    self.assertEqual(protein.sequence(separator=sep), seq1 + sep + seq2, 'Wrong sequences')

  def test_atom_coordinates(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain_A.residues[0].atoms = [Atom(residue=chain_A.residues[0], code='C', coords=(1, 0, 0)),
                                 Atom(residue=chain_A.residues[0], code='N', coords=(0, 1, 0))]
    chain_A.residues[1].atoms = [Atom(residue=chain_A.residues[1], code='H', coords=(0, 0, 1))]
    chain_B.residues[0].atoms = [Atom(residue=chain_B.residues[0], code='H', coords=(-1, 0, 0)),
                                 Atom(residue=chain_B.residues[0], code='O', coords=(0, -1, 0))]
    chain_B.residues[1].atoms = [Atom(residue=chain_B.residues[1], code='C', coords=(0, 0, -1))]
    coords = protein.coords()
    self.assertEqual(coords.size(), torch.Size([6, 3]), 'Wrong shape for coordinate tensor')
    torch.testing.assert_close(coords,
                               torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]))

  def test_protein_cache_coordinates(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain_A.residues[0].atoms = [Atom(residue=chain_A.residues[0], code='C', coords=(1, 0, 0)),
                                 Atom(residue=chain_A.residues[0], code='N', coords=(0, 1, 0))]
    chain_A.residues[1].atoms = [Atom(residue=chain_A.residues[1], code='H', coords=(0, 0, 1))]
    chain_B.residues[0].atoms = [Atom(residue=chain_B.residues[0], code='H', coords=(-1, 0, 0)),
                                 Atom(residue=chain_B.residues[0], code='O', coords=(0, -1, 0))]
    chain_B.residues[1].atoms = [Atom(residue=chain_B.residues[1], code='C', coords=(0, 0, -1))]
    coords = protein.coords()
    cached_coords = protein.coords()
    torch.testing.assert_close(coords, cached_coords, msg='Invalid retrieval of cached values')

  def test_retrieve_atom_symbols(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain_A.residues[0].atoms = [Atom(residue=chain_A.residues[0], symbol='C', coords=(1, 0, 0)),
                                 Atom(residue=chain_A.residues[0], symbol='N', coords=(0, 1, 0))]
    chain_A.residues[1].atoms = [Atom(residue=chain_A.residues[1], symbol='O', coords=(0, 0, 1))]
    chain_B.residues[0].atoms = [Atom(residue=chain_B.residues[0], symbol='H', coords=(-1, 0, 0)),
                                 Atom(residue=chain_B.residues[0], symbol='S', coords=(0, -1, 0))]
    chain_B.residues[1].atoms = [Atom(residue=chain_B.residues[1], symbol='C', coords=(0, 0, -1))]
    atom_symbols = protein.get_atom_symbols()
    self.assertEqual(['C', 'N', 'O', 'H', 'S', 'C'], atom_symbols, 'Atom symbols do not match')

  def test_protein_has_chain(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain_found = protein.has_chain('A')
    self.assertTrue(chain_found, 'Unable to find chain')

  def test_protein_has_not_chain(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain_found = protein.has_chain('C')
    self.assertFalse(chain_found, 'Find an unknown chain')

  def test_retrieve_chain_from_protein(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain = protein.get_chain('A')
    self.assertEqual('A', chain.name, 'Wrong chain retrieved')

  def test_failing_in_retrieving_chain_from_protein_raises_error(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    with self.assertRaises(ValueError):
      protein.get_chain('C')

  def test_protein_ca_coordinates(self):
    protein = Protein('DEM0')
    chain_A = Chain('A', residues=Residues.from_sequence('CC'), protein=protein)
    chain_B = Chain('B', residues=Residues.from_sequence('AA'), protein=protein)
    protein.chains = [chain_A, chain_B]
    chain_A.residues[0].atoms = [Atom(residue=chain_A.residues[0], code='CA', symbol='C', coords=(1, 0, 0)),
                                 Atom(residue=chain_A.residues[0], symbol='N', coords=(0, 1, 0))]
    chain_A.residues[1].atoms = [Atom(residue=chain_A.residues[1], code='CA', symbol='C', coords=(0, 0, 1))]
    chain_B.residues[0].atoms = [Atom(residue=chain_B.residues[0], symbol='H', coords=(-1, 0, 0)),
                                 Atom(residue=chain_B.residues[0], code='CA', coords=(0, -1, 0))]
    chain_B.residues[1].atoms = [Atom(residue=chain_B.residues[1], code='CA', symbol='C', coords=(0, 0, -1))]
    ca_coords = protein.ca_coords()
    ground_truth = torch.tensor([
      [1, 0, 0],
      [0, 0, 1],
      [0, -1, 0],
      [0, 0, -1]
    ])
    self.assertEqual(torch.Size([4, 3]), ca_coords.shape, 'Invalid CA shape')
    torch.testing.assert_close(ground_truth, ca_coords, msg='CA coords are not close enough')


class ProteinsTest(unittest.TestCase):
  def test_merge_proteins_with_different_chains_raise_error(self):
    seq = Protein('DEM01', chains=[
      Chain('A', residues=Residues.from_sequence('CC')),
      Chain('B', residues=Residues.from_sequence('AAY'))
    ])
    struct = Protein('DEMO2', chains=[
      Chain('A', residues=[
        Residue(1, 'G', atoms=[
          Atom(code='CA', symbol='C', coords=(1, 0, 0)),
          Atom(symbol='N', coords=(0, 1, 0))
        ])
      ]),
      Chain('C', residues=[
        Residue(2, 'Y', atoms=[
          Atom(code='CA', symbol='C', coords=(0, 0, 1))
        ])
      ])
    ])
    with self.assertRaises(ValueError):
      Proteins.merge_sequence_with_structure(seq, struct)

  def test_merge_proteins_with_different_residue_numbers_raise_error(self):
    seq = Protein('DEM01', chains=[
      Chain('A', residues=Residues.from_sequence('CC')),
    ])
    struct = Protein('DEMO2', chains=[
      Chain('A', residues=[
        Residue(1, 'G', atoms=[
          Atom(code='CA', symbol='C', coords=(1, 0, 0)),
          Atom(symbol='N', coords=(0, 1, 0))
        ])
      ])
    ])
    with self.assertRaises(ValueError):
      Proteins.merge_sequence_with_structure(seq, struct)
      
  def test_merge_proteins(self):
    seq = Protein('DEM01', filename='demo1.fa', chains=[
      Chain('A', residues=Residues.from_sequence('CC')),
      Chain('B', residues=Residues.from_sequence('A')),
    ], props={'foo': 'bar'})
    struct = Protein('DEMO1', filename='demo1.pdb', chains=[
      Chain('A', residues=[
        Residue(1, 'G', atoms=[
          Atom(code='CA', symbol='C', coords=(1, 0, 0)),
          Atom(symbol='N', coords=(0, 1, 0))
        ]),
        Residue(2, 'K', atoms=[
          Atom(code='CA', symbol='C', coords=(1, -1, 1))
        ])
      ]),
      Chain('B', residues=[
        Residue(3, 'Y', atoms=[
          Atom(code='CA', symbol='C', coords=(0, 0, 1))
        ])
      ])
    ], props={'bar': 'baz'})
    seq = Proteins.merge_sequence_with_structure(seq, struct)
    self.assertEqual(seq.filename, struct.filename, 'Filename not switched')
    for seq_chain, struct_chain in zip(seq.chains, struct.chains):
      for seq_residue, struct_residue in zip(seq_chain.residues, struct_chain.residues):
        self.assertEqual(len(struct_residue.atoms), len(seq_residue.atoms),
                         f'Invalid number of atoms for chain {seq_chain.name} and residue {seq_residue.number}')
        self.assertEqual([a.coords for a in seq_residue.atoms], [a.coords for a in struct_residue.atoms],
                         f'Invalid atom coordinates for chain {seq_chain.name} and residue {seq_residue.number}')
    self.assertDictEqual({'foo':'bar', 'bar':'baz'}, seq.props, 'Invalid merged props')


if __name__ == '__main__':
  unittest.main()
