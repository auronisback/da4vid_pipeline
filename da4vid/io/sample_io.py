import os
import sys

from tqdm import tqdm

from da4vid.io import read_pdb_folder, read_protein_mpnn_fasta
from da4vid.io.fasta_io import read_fasta
from da4vid.model.samples import SampleSet, Sample, Fold, Sequence


def sample_set_from_backbones(backbone_folder: str, b_fact_prop: str = 'plddt') -> SampleSet:
  """
  Creates a sample set object with only backbone samples.
  :param backbone_folder: The folder in which backbones are stored
  :param b_fact_prop: The semantic related to the B-factor column in PDB
  :return: The constructed sample set
  """
  return __create_sample_set(backbone_folder, b_fact_prop)


def sample_set_from_folders(backbone_folder: str, samples_folder: str,
                            model: str, b_fact_prop: str = 'plddt') -> SampleSet:
  """
  Creates a SampleSet object reading the original proteins from the PDB files in backbone folder
  and linking samples within the samples folder.
  :param backbone_folder: The folder where backbones PDB file are stored
  :param samples_folder: The folder where samples are stored. There should be one
                         subfolder for each backbone PDB file, where the samples reside
  :param model: The model used to predict the folding
  :param b_fact_prop: The temperature factor property semantic in PDB files. Defaults to 'plddt'
  :return: The new sample set with original backbones and their samples read from folder
  """
  sample_set = __create_sample_set(backbone_folder, b_fact_prop)
  # Adding samples to all originals
  for sample in tqdm(sample_set.samples(), file=sys.stdout):
    if os.path.exists(os.path.join(samples_folder, sample.name)):
      folded_proteins = read_pdb_folder(os.path.join(samples_folder, sample.name), b_fact_prop=b_fact_prop)
      for p in folded_proteins:
        sequence = Sequence(p.name, p.filename, sample=sample, protein=p)
        sequence.add_folds(Fold(sequence, p.filename, model, p))
        sample.add_sequences(sequence)
  return sample_set


def sample_set_from_fasta_folders(backbone_folder: str, fasta_folder: str,
                                  b_fact_prop: str = 'plddt', from_pmpnn: bool = True) -> SampleSet:
  """
  Creates a sample set in which samples are taken from FASTA sequences.
  :param backbone_folder: The folder in which there are backbones
  :param fasta_folder: The folder in which find FASTA sequences
  :param b_fact_prop: The property related to B-factors for backbone sequences
  :param from_pmpnn: Flag checking if FASTA sequences are in ProteinMPNN format
  :return: The sample set containing read samples
  """
  sample_set = __create_sample_set(backbone_folder, b_fact_prop)
  for sample in tqdm(sample_set.samples(), file=sys.stdout):
    full_fasta_path = os.path.join(fasta_folder, f'{sample.name}.fa')
    if os.path.isfile(full_fasta_path):
      sample.add_sequences([
        Sequence(s.name, s.filename, sample=sample, protein=s) for s in (
          read_protein_mpnn_fasta(full_fasta_path) if from_pmpnn else read_fasta(full_fasta_path))
        if s.name != sample.name
      ])
  return sample_set


def __create_sample_set(backbone_folder: str, b_fact_prop: str) -> SampleSet:
  sample_set = SampleSet()
  # Adding original backbones to sample set
  proteins = read_pdb_folder(backbone_folder, b_fact_prop=b_fact_prop)
  sample_set.add_samples([Sample(p.name, p.filename, p) for p in proteins])
  return sample_set
