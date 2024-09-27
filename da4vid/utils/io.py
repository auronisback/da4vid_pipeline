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
