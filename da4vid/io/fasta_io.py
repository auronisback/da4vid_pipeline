import os.path
from typing import List

from da4vid.model.proteins import Protein, Chain, Residues


def read_fasta(fasta_path: str, chain_separator: chr = ':') -> List[Protein]:
  """
  Reads a FASTA file, with header and protein sequence.
  :param fasta_path: The path to the FASTA file
  :param chain_separator: The character used to separate chains
                          in a particular sequence
  :return: The list of proteins read from the file, whose
           name is the name of the sequence and whose filename
           is the FASTA input file
  :raise: FileNotFoundError if the file does not exist
  :raise: FileExistsError if the given input is a directory
  """
  if not os.path.exists(fasta_path):
    raise FileNotFoundError(f'input file does not exist: {fasta_path}')
  if not os.path.isfile(fasta_path):
    raise FileExistsError(f'given input is actually a directory: {fasta_path}')
  with open(fasta_path) as f:
    proteins = []
    title = f.readline()
    while title:
      title = title[1:].strip()
      sequence = f.readline().strip()
      proteins.append(Protein(name=title, filename=fasta_path,
                              chains=[Chain('A', residues=Residues.from_sequence(sequence))]))
      # TODO: add support for multiple chains with specified separator
      title = f.readline()
    return proteins


def read_protein_mpnn_fasta(fasta_path: str) -> List[Protein]:
  """
  Reads a FASTA file produced by ProteinMPNN. This kind of files
  has the original protein as the first sequence, with some parameters,
  and the other sequences are titled with some properties related to
  the model (temperature, score, ...).
  :param fasta_path: The path to the FASTA file
  :return: The list of proteins read from the file, each with the
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


def write_fasta(proteins: List[Protein], fasta_out: str, overwrite: bool = False,
                chain_separator: str = '') -> None:
  """
  Writes the sequences of a list of proteins in FASTA format.
  :param proteins: The list of proteins to write
  :param fasta_out: The path to the fasta output
  :param overwrite: Flag indicating if the output file should be overwritten. Defaults to False.
  :param chain_separator: The character to divide chains in a single sequence
  :raise FileExistsError: If the specified output file exists and overwrite has not
                          been set to True, or the output path is a directory
  """
  if not overwrite and os.path.exists(fasta_out):
    raise FileExistsError(f'specified FASTA output exists: {fasta_out}')
  if os.path.isdir(fasta_out):
    raise FileExistsError(f'specified FASTA output is a directory: {fasta_out}')
  with open(fasta_out, 'w') as f:
    for protein in proteins:
      f.write(f'>{protein.name}\n{protein.sequence(separator=chain_separator)}\n')
    f.flush()


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
