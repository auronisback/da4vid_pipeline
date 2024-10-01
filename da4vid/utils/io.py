from typing import Union, List

import os
from pathlib import Path

from da4vid.model import Protein, Residue, Chain, Atom


def read_pdb_folder(pdb_folder: Union[str, Path], b_fact_prop: str = 'temperature') -> List[Protein]:
  """
  Reads all PDB files in a folder and returns the list of read proteins.
  :param pdb_folder: The folder to analyze
  :param b_fact_prop: The semantic associated to the B-factor column in all PDBs. Defaults to 'temperature'
  :return: A list of read Proteins.
  """
  if not os.path.exists(pdb_folder):
    raise FileNotFoundError(f'Folder not exists: {pdb_folder}')
  if not os.path.isdir(pdb_folder):
    raise ValueError(f'Given path is not a folder: {pdb_folder}')
  proteins = []
  for f in os.listdir(pdb_folder):
    if str(f).endswith('.pdb'):
      proteins.append(read_from_pdb(os.path.join(pdb_folder, f), b_fact_prop=b_fact_prop))
  return proteins


def read_from_pdb(pdb_file: Union[str, Path], b_fact_prop: str = 'temperature') -> Protein:
  """
  Reads a single PDB files and creates the related Protein.
  :param pdb_file: The path to the PDB file
  :param b_fact_prop: The semantic related to the B-factor column
  :return: The read Protein object
  """
  if not os.path.exists(pdb_file):
    raise FileNotFoundError(f'PDB file not exists: {pdb_file}')
  if not os.path.isfile(pdb_file):
    raise ValueError(f'Given PDB file is actually a directory: {pdb_file}')
  return __load_protein(pdb_file, b_fact_prop=b_fact_prop)


def __load_protein(pdb_file: Union[str, Path], b_fact_prop: str) -> Protein:
  protein_name = '.'.join(os.path.basename(pdb_file).split('.')[:-1])
  protein = Protein(protein_name, file=str(pdb_file))
  with open(pdb_file) as f:
    resi_id = 0  # Global (outside chains) residue identifier
    chain: Chain = Chain('-1')
    resi: Residue = Residue(-1)
    for line in f:
      if line.startswith('ATOM'):
        line = line.strip()
        chain_id = line[21]
        if chain_id != chain.name:
          chain = Chain(chain_id, protein=protein)
          protein.chains.append(chain)
        resi_num = int(line[22:26])
        if resi_num != resi.number:
          resi_name = line[17:20].strip()
          resi = Residue(residue_id=resi_id, chain=chain, number=resi_num, code=resi_name)
          chain.residues.append(resi)
          resi_id += 1
        atom_number = int(line[6:11])
        atom_name = line[12:16].strip()
        coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        b_fact = float(line[60:66].strip())
        symbol = line[76:78].strip()
        # Sanitizing symbol if not present
        if not symbol:
          symbol = atom_name[0]  # Symbol is the firs character of atom_name
        atom = Atom(residue=resi, number=atom_number, code=atom_name,
                    coords=coords, symbol=symbol, props={b_fact_prop: b_fact})
        resi.atoms.append(atom)
  return protein


def write_pdb(proteins: Union[Protein, List[Protein]],
              output_folder: str, prefix: str = None) -> List[str]:
  """
  Writes a single protein or a list of proteins in the output folder
  in the PDB format.
  :param proteins: The single protein or the list of proteins to write
  :param output_folder: The folder in which to save the proteins, each
                        in a different .pdb file. If the directory path
                        does not exist, it will be created unless no
                        proteins have been specified.
  :param prefix: The prefix for .pdb filenames in the folder. If
                         given, each protein filename will be this prefix
                         concatenated to an incremental index, otherwise
                         the name of each protein will be its filename
  :return: The list of paths of written proteins
  :raise: FileExistsError if the specified folder is a regular file
  """
  # Checking if folder is accessible, otherwise it will be created
  if Path(output_folder).is_file():
    raise FileExistsError(f'specified output folder is a regular file: {output_folder}')
  # If no proteins are available, then just return
  if not proteins:
    return []
  # Creating directory path
  os.makedirs(output_folder, exist_ok=True)
  if isinstance(proteins, Protein):
    proteins = [proteins]
  paths = []
  for i, protein in enumerate(proteins):
    # Creating the output path
    filename = f'{protein.name}.pdb' if prefix is None else f'{prefix}_{i}.pdb'
    out_path = os.path.join(output_folder, filename)
    __write_protein_pdb(protein, out_path)
    paths.append(out_path)
  return paths


def __write_protein_pdb(protein: Protein, out_path: str):
  with open(out_path, 'w') as f:
    for chain in protein.chains:
      for residue in chain.residues:
        for atom in residue.atoms:
          atom_bfact = atom.props.get(list(atom.props.keys())[0], 0.)
          f.write(
            f'ATOM  {str(atom.number).rjust(5)}  '
            f'{atom.code.ljust(4)}{residue.get_three_letters_code().rjust(3)} '
            f'{chain.name}{str(residue.number).rjust(4)}    '
            f'{atom.coords[0]:8.3f}{atom.coords[1]:8.3f}{atom.coords[2]:8.3f}'
            f'{"1.00".rjust(6)}{atom_bfact:6.2f}{" " * 11}{atom.symbol}\n'
          )
    f.flush()

