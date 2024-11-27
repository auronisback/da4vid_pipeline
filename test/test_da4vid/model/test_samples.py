import unittest

from da4vid.model.proteins import Protein, Residues, Chain, Chains, Proteins
from da4vid.model.samples import FoldMetrics, Sequence, Fold, Sample, SampleSet


class FoldMetricsTest(unittest.TestCase):
  def test_add_and_get_metric(self):
    m = FoldMetrics(None)
    m.add_metric('my.awesome.metric', 12)
    self.assertEqual(12, m.get_metric('my.awesome.metric'))

  def test_get_metric_not_present(self):
    m = FoldMetrics(None)
    m.add_metric('my.awesome.metric', 7)
    self.assertIsNone(m.get_metric('my.non.existing.metric'))

  def test_create_metrics_from_dict(self):
    m = FoldMetrics(None, {'metric1': {'first': 11, 'second': 14}, 'metric2': {'first': 7, 'second': 12}})
    self.assertEqual(11, m.get_metric('metric1.first'))
    self.assertEqual(14, m.get_metric('metric1.second'))
    self.assertEqual(7, m.get_metric('metric2.first'))
    self.assertEqual(12, m.get_metric('metric2.second'))


class SequenceTest(unittest.TestCase):
  def test_recover_sequence_string(self):
    seq = Sequence('DEMO', 'demo.fa', sample=None,
                   protein=Protein('DEMO', 'demo.fa',
                                   chains=Chains.from_sequence('KGSTANL')))
    self.assertEqual('KGSTANL', seq.sequence_to_str())

  def test_add_fold_to_sequence(self):
    seq = Sequence('DEMO', 'demo.fa', sample=None,
                   protein=Protein('DEMO', 'demo.fa',
                                   chains=Chains.from_sequence('KGSTANL')))
    fold = Fold(seq, '/some/path', 'mymodel1')
    seq.add_folds(fold)
    self.assertEqual(1, len(seq.folds()))

  def test_get_fold_from_sequence(self):
    seq = Sequence('DEMO', 'demo.fa', sample=None,
                   protein=Protein('DEMO', 'demo.fa',
                                   chains=Chains.from_sequence('KGSTANL')))
    fold1 = Fold(seq, '/some/path', 'mymodel1')
    fold2 = Fold(seq, '/some/path', 'mymodel2')
    seq.add_folds(fold1)
    seq.add_folds(fold2)
    self.assertEqual(fold2, seq.get_fold_for_model('mymodel2'))

  def test_get_fold_for_model_returns_none_when_no_model(self):
    seq = Sequence('DEMO', 'demo.fa', sample=None,
                   protein=Protein('DEMO', 'demo.fa',
                                   chains=Chains.from_sequence('KGSTANL')))
    fold1 = Fold(seq, '/some/path', 'mymodel1')
    fold2 = Fold(seq, '/some/path', 'mymodel2')
    seq.add_folds(fold1)
    seq.add_folds(fold2)
    self.assertIsNone(seq.get_fold_for_model('notmymodel'))


class SampleTest(unittest.TestCase):
  def test_add_sequence(self):
    s = Sample('DEMO', '/my/path')
    s.add_sequences(Sequence('seq1'))
    s.add_sequences(Sequence('seq2'))
    self.assertEqual(2, len(s.sequences()))

  def test_retrieve_sequence_by_name(self):
    s = Sample('DEMO', '/my/path')
    seq1 = Sequence('seq1')
    seq2 = Sequence('seq2')
    s.add_sequences(seq1)
    s.add_sequences(seq2)
    self.assertEqual(seq1, s.get_sequence_by_name('seq1'))
    self.assertEqual(seq2, s.get_sequence_by_name('seq2'))

  def test_retrieve_sequence_by_name_fails_when_name_not_present(self):
    s = Sample('DEMO', '/my/path')
    seq1 = Sequence('seq1')
    s.add_sequences(seq1)
    self.assertIsNone(s.get_sequence_by_name('other_seq'))

  def test_remove_duplicate_sequences(self):
    s = Sample('DEMO', '/my/path')
    seq1 = Sequence('seq1', protein=Proteins.from_sequence('seq1', 'KGSTANL'))
    seq2 = Sequence('seq2', protein=Proteins.from_sequence('seq2', 'YYKNAG'))
    seq3 = Sequence('seq3', protein=Proteins.from_sequence('seq3', 'KGSTANL'))
    s.add_sequences(seq1)
    s.add_sequences(seq2)
    s.add_sequences(seq3)
    s.remove_duplicate_sequences()
    self.assertEqual(2, len(s.sequences()))
    self.assertIn(seq1, s.sequences())
    self.assertIn(seq2, s.sequences())
    self.assertNotIn(seq3, s.sequences())


class SampleSetTest(unittest.TestCase):
  def test_add_sample(self):
    sample_set = SampleSet()
    s = Sample('DEMO')
    sample_set.add_samples(s)
    self.assertIn(s, sample_set.samples())

  def test_add_sample_updates_existing_sample(self):
    sample_set = SampleSet()
    s1 = Sample('DEMO')
    sample_set.add_samples(s1)
    s2 = Sample('DEMO')
    sample_set.add_samples(s2)
    self.assertNotIn(s1, sample_set.samples())
    self.assertIn(s2, sample_set.samples())

  def test_add_multiple_samples(self):
    sample_set = SampleSet()
    s1 = Sample('DEMO1')
    s2 = Sample('DEMO2')
    sample_set.add_samples([s1, s2])
    self.assertIn(s1, sample_set.samples())
    self.assertIn(s2, sample_set.samples())

  def test_add_multiple_samples_updating_previous_ones(self):
    sample_set = SampleSet()
    s1 = Sample('DEMO1')
    s2 = Sample('DEMO2')
    s3 = Sample('DEMO1')
    sample_set.add_samples(s1)
    sample_set.add_samples([s2, s3])
    self.assertNotIn(s1, sample_set.samples())
    self.assertIn(s2, sample_set.samples())
    self.assertIn(s3, sample_set.samples())

  def test_get_sample_by_name(self):
    sample_set = SampleSet()
    s1 = Sample('DEMO')
    s2 = Sample('OTHER')
    sample_set.add_samples(s1)
    sample_set.add_samples(s2)
    self.assertEqual(s1, sample_set.get_sample_by_name('DEMO'))
    self.assertEqual(s2, sample_set.get_sample_by_name('OTHER'))

  def test_get_sample_by_name_returns_none_when_name_not_present(self):
    sample_set = SampleSet()
    s1 = Sample('DEMO')
    s2 = Sample('OTHER')
    sample_set.add_samples(s1)
    sample_set.add_samples(s2)
    self.assertIsNone(sample_set.get_sample_by_name('unknown'))

  def test_folded_sample_set(self):
    sample_set = SampleSet()
    s1 = Sample('first')
    seq11 = Sequence('first_seq')
    f111 = Fold(seq11, '/my/path', model='my_model')
    f112 = Fold(seq11, '/my/path', model='my_model')
    seq11.add_folds(f111)
    seq11.add_folds(f112)
    s1.add_sequences(seq11)
    seq12 = Sequence('second_seq')
    f121 = Fold(seq12, '/my/path', model='my_model')
    f122 = Fold(seq12, '/my/path', model='other_model')
    seq12.add_folds(f121)
    seq12.add_folds(f122)
    s1.add_sequences(seq12)
    s2 = Sample('second')
    seq21 = Sequence('first_second_sample_seq')
    f211 = Fold(seq21, '/my/path', 'my_model')
    seq21.add_folds(f211)
    s2.add_sequences(seq21)
    sample_set.add_samples(s1)
    seq22 = Sequence('second_second_sample_seq')
    f221 = Fold(seq22, '/my/path', 'other_model')
    seq22.add_folds(f221)
    s2.add_sequences(seq22)
    sample_set.add_samples(s2)
    new_set = sample_set.folded_sample_set('my_model')
    self.assertEqual(3, len(new_set.samples()))
    self.assertNotIn(f111, new_set.samples())
    self.assertIn(f112, new_set.samples())
    self.assertIn(f121, new_set.samples())
    self.assertIn(f211, new_set.samples())
