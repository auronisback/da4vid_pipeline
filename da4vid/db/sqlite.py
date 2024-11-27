from typing import List

from da4vid.db.base import DB
from da4vid.model import Protein
from da4vid.model.samples import SampleSet


class SqliteDB(DB):

  def __init__(self):
    pass

  def _save_sample_set(self, sample_set: SampleSet):
    pass

  def _save_proteins(self, proteins: List[Protein]):
    pass