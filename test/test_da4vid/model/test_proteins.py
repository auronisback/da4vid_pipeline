import unittest

import torch

from da4vid.model.proteins import Residue, Chain, Residues, Protein, Atom, Proteins, NestedDict, Chains, Epitope


class NestedDictionaryTest(unittest.TestCase):
  def test_add_value_to_dictionary_with_new_key(self):
    d = NestedDict()
    d.add_value('my.new.value', 'the value')
    self.assertTrue(d.has_key('my.new.value'), f'Key not found!')

  def test_get_value_from_dictionary_with_key_present(self):
    d = NestedDict()
    d.add_value('my.new.value', 'the value')
    self.assertEqual('the value', d.get_value('my.new.value'))

  def test_get_value_returns_none_when_key_is_not_present(self):
    d = NestedDict()
    d.add_value('my.new.value', 'the value')
    self.assertIsNone(d.get_value('my.other.value'))

  def test_create_nested_dictionary_from_dict(self):
    d = NestedDict({'my': {'initial': 'dictionary'}, 'is': 'here'})
    self.assertEqual('dictionary',  d.get_value('my.initial'))
    self.assertEqual('here', d.get_value('is'))

  def test_add_value_updates_dictionary_if_key_already_present(self):
    d = NestedDict({'my': {'key': 'old_value'}})
    d.add_value('my.key', 'new_value')
    self.assertEqual('new_value', d.get_value('my.key'))

  def test_merge_dictionaries(self):
    d1 = NestedDict()
    d1.add_value('my.new.key', 'value')
    d1.add_value('my.brand.new.key', 'new_value')
    d2 = NestedDict()
    d2.add_value('my.other.key', 'other_value')
    d1.merge(d2)
    self.assertEqual('value', d1.get_value('my.new.key'))
    self.assertEqual('new_value', d1.get_value('my.brand.new.key'))
    self.assertEqual('other_value', d1.get_value('my.other.key'))
    # Checking d2 has not been modified
    self.assertFalse(d2.has_key('my.new.key'))
    self.assertFalse(d2.has_key('my.brand.new.key'))

  def test_merge_dictionaries_raises_error_if_conflicting_values(self):
    d1 = NestedDict({'my': {'new': 'key'}})
    d2 = NestedDict({'my': {'new': 'conflict'}})
    with self.assertRaises(ValueError):
      d1.merge(d2)


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


class ResiduesTest(unittest.TestCase):
  def test_residues_from_sequence(self):
    residues = Residues.from_sequence('KGSTANL')
    self.assertEqual(7, len(residues))
    for c, r in zip('KGSTANL', residues):
      self.assertEqual(c, r.get_one_letter_code())

  def test_residues_from_sequence_raises_error_if_invalid_aa_provided(self):
    with self.assertRaises(ValueError):
      Residues.from_sequence('KGSTANLB')


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


class ChainsTest(unittest.TestCase):
  def test_chains_with_one_chain_sequence(self):
    chains = Chains.from_sequence('KGSTANL')
    self.assertEqual(1, len(chains))
    self.assertEqual('A', chains[0].name)
    self.assertEqual('KGSTANL', chains[0].sequence())

  def test_chains_with_multiple_chain_sequences(self):
    chains = Chains.from_sequence('KGSTANL:LNATSGK:ACYK')
    self.assertEqual(3, len(chains))
    self.assertEqual('A', chains[0].name)
    self.assertEqual('B', chains[1].name)
    self.assertEqual('C', chains[2].name)
    self.assertEqual('KGSTANL', chains[0].sequence())
    self.assertEqual('LNATSGK', chains[1].sequence())
    self.assertEqual('ACYK', chains[2].sequence())

  def test_chains_with_multiple_chain_sequences_and_different_separator(self):
    chains = Chains.from_sequence('DIVMTQ$SLAMSV$RNQKY$DDSR', chain_separator='$')
    self.assertEqual(4, len(chains))

  def test_chains_with_invalid_separator(self):
    # Omitting the separator parameter
    with self.assertRaises(ValueError):
      chains = Chains.from_sequence('DIVMTQ$SLAMSV$RNQKY$DDSR')

  def test_chains_from_empty_sequence(self):
    chains = Chains.from_sequence('')
    self.assertEqual([], chains)

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

  def test_add_properties(self):
    p = Protein('DEM0')
    p.add_prop('my.first.wonderful.property', 10)
    p.add_prop('my.second.awful.property', 'foo')
    self.assertDictEqual({
      'my': {
        'first': {
          'wonderful': {
            'property': 10
          }
        }, 'second': {
          'awful': {
            'property': 'foo'
          }
        }
      }
    }, p.props.dict, 'Invalid property dictionary')

  def test_get_property(self):
    p = Protein('DEM0')
    p.add_prop('my.first.wonderful.property', 10)
    p.add_prop('my.second.awful.property', 'foo')
    value = p.get_prop('my.first.wonderful.property')
    self.assertEqual(10, value, 'Invalid retrieved property')

  def test_get_not_existing_property(self):
    p = Protein('DEM0')
    p.add_prop('my.first.wonderful.property', 10)
    p.add_prop('my.second.awful.property', 'foo')
    value = p.get_prop('my.second.wonderful.property')
    self.assertIsNone(value, 'Property has been found?!')

  def test_protein_length(self):
    p = Proteins.from_sequence('DEMO', 'KGSTANL')
    self.assertEqual(7, p.length())

  def test_protein_has_prop(self):
    p = Protein('DEMO', props={'my': {'prop': 42}})
    self.assertTrue(p.has_prop('my.prop'))

  def test_protein_has_not_prop(self):
    p = Protein('DEMO', props={'my': {'prop': 42}})
    self.assertFalse(p.has_prop('another.prop'))


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
    self.assertDictEqual({'foo': 'bar', 'bar': 'baz'}, seq.props.dict, 'Invalid merged props')

  def test_protein_from_sequence(self):
    p = Proteins.from_sequence('DEMO', 'KGSTANL:LNATSGK')
    self.assertEqual('DEMO', p.name)
    self.assertEqual(2, len(p.chains))
    self.assertEqual('KGSTANL', p.chains[0].sequence())
    self.assertEqual('LNATSGK', p.chains[1].sequence())


class EpitopeTest(unittest.TestCase):
  def test_create_valid_epitope_without_protein(self):
    epi = Epitope('A', 10, 19)
    self.assertEqual('A', epi.chain)
    self.assertEqual(10, epi.start)
    self.assertEqual(19, epi.end)

  def test_create_epitope_fails_when_start_is_negative(self):
    with self.assertRaises(ValueError):
      Epitope('A', -3, 10)

  def test_create_epitope_fails_when_start_greater_than_end(self):
    with self.assertRaises(ValueError):
      Epitope('A', 20, 19)

  def test_create_valid_epitope_with_specified_protein(self):
    protein = Proteins.from_sequence('DEM0', 'KGSTANLLNATSGK:AAYKLGGCINNN')
    epi = Epitope('B', 3, 7, protein)
    self.assertEqual('B', epi.chain)
    self.assertEqual(3, epi.start)
    self.assertEqual(7, epi.end)
    self.assertEqual(protein, epi.protein)

  def test_create_epitope_for_given_protein_should_rise_error_if_chain_is_missing(self):
    protein = Proteins.from_sequence('DEM0', 'KGSTANLLNATSGK:AAYKLGGCINNN')
    with self.assertRaises(ValueError):
      Epitope('C', 3, 7, protein)

  def test_create_epitope_for_given_protein_should_rise_error_if_invlid_residues(self):
    protein = Proteins.from_sequence('DEM0', 'KGSTANLLNATSGK:AAYKLGGCINNN')
    with self.assertRaises(ValueError):
      Epitope('A', 40, 50, protein)


if __name__ == '__main__':
  unittest.main()
