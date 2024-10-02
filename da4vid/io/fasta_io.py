import os.path
from typing import List

from da4vid.model import Protein, Chain, Residues


def read_protein_mpnn_fasta(fasta_path: str) -> List[Protein]:
  """
  Reads a FASTA file produced by ProteinMPNN. This kind of files
  has the original protein as the first sequence, with some parameters,
  and the other sequences are titled with some properties related to
  the model (temperature, score, ...).
  :param fasta_path: The path to the FASTA file
  :return: The list of protein read from the file, each with the
           name of the original protein to which the sample number
           is appended
  :raise: FileNotFoundError if the file does not exist
  :raise: FileExistsError if the given input is a directory
  """
  if not os.path.exists(fasta_path):
    raise FileNotFoundError(f'input file does not exist: {fasta_path}')
  if not os.path.isfile(fasta_path):
    raise FileExistsError(f'given input is actually a directory: {fasta_path}')
  with open(fasta_path) as f:
    proteins = []
    # Starting: creating the original protein
    title = f.readline().strip()
    sequence = f.readline().strip()
    original = __create_original_protein(title, sequence)
    proteins.append(original)
    # Creating other proteins
    title = f.readline()
    while title:
      sequence = f.readline()
      proteins.append(__create_decoded_protein(title, sequence, original))
      title = f.readline()
    return proteins


def __create_original_protein(title: str, sequence: str) -> Protein:
  tokens = title[1:].split(',')
  name = tokens[0].strip()
  designed_chain = (__get_value(tokens[4]).replace("[", "")
                    .replace("'", "").replace("]", ""))
  return Protein(name, props={
    'protein_mpnn': {
      'score': float(__get_value(tokens[1])),
      'global_score': float(__get_value(tokens[2])),
      'model_name': __get_value(tokens[5]),
      'seed': int(__get_value(tokens[7]))
    }
  }, chains=[Chain(designed_chain, residues=Residues.from_sequence(sequence))])


def __create_decoded_protein(title: str, sequence: str, original: Protein) -> Protein:
  chain_name = original.chains[0].name
  tokens = title[1:].split(',')
  sample = int(__get_value(tokens[1]))
  name = f'{original.name}_{sample}'
  return Protein(name, props={
    'protein_mpnn': {
      'T': float(__get_value(tokens[0])),
      'score': float(__get_value(tokens[2])),
      'global_score': float(__get_value(tokens[3])),
      'seq_recovery': float(__get_value(tokens[4]))
    }
  }, chains=[Chain(chain_name, residues=Residues.from_sequence(sequence.strip()))])


def __get_value(token: str) -> str:
  return token.split('=')[1].strip()
