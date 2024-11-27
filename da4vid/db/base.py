import abc
from typing import List

from da4vid.model import Protein
from da4vid.model.samples import SampleSet


class DB(abc.ABC):

  def save(self, proteins: SampleSet | Protein | List[Protein]):
    if isinstance(proteins, SampleSet):
      self._save_sample_set(proteins)
    elif isinstance(proteins, Protein):
      proteins = [proteins]
      self._save_proteins(proteins)

  @abc.abstractmethod
  def _save_sample_set(self, sample_set: SampleSet):
    return

  @abc.abstractmethod
  def _save_proteins(self, proteins: List[Protein]):
    return
