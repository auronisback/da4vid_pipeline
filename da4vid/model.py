from typing import List, Dict, Any, Tuple

import abc
import torch

from da4vid.metrics.rog import rog, atoms_to_one_hot

IUPAC_DATA = {
  "A": "Ala",
  "C": "Cys",
  "D": "Asp",
  "E": "Glu",
  "F": "Phe",
  "G": "Gly",
  "H": "His",
  "I": "Ile",
  "K": "Lys",
  "L": "Leu",
  "M": "Met",
  "N": "Asn",
  "P": "Pro",
  "Q": "Gln",
  "R": "Arg",
  "S": "Ser",
  "T": "Thr",
  "V": "Val",
  "W": "Trp",
  "Y": "Tyr",
}

ONE_TO_THREE = {key: value.upper() for key, value in IUPAC_DATA.items()}
THREE_TO_ONE = {value: key for key, value in ONE_TO_THREE.items()}


class Atom:
  def __init__(self, residue=None, number: int = None, code: str = None,
               coords: Tuple[float, float, float] = None, symbol: str = None, props: Dict[str, Any] = None):
    self.residue = residue
    self.number = number
    self.code = code
    self.coords = coords if coords is not None else [None, None, None]
    self.symbol = symbol
    self.props = props if props is not None else {}  # Generic atom properties


class Residue:
  def __init__(self, residue_id: int, chain=None, number: int = None, code: str = None,
               atoms: List[Atom] = None, props: Dict[str, Any] = None):
    self.id = residue_id
    self.chain = chain
    self.number = number
    self.__code1 = self.__code3 = None
    if code is not None:
      self.set_code(code)
    self.atoms = atoms if atoms is not None else []
    self.props = props if props is not None else {}  # Generic properties of this residue

  def set_code(self, code: str):
    n = len(code)
    if n == 1:  # One-letter code, setting the three-letters code
      if code not in ONE_TO_THREE.keys():
        raise ValueError(f'Invalid one-letter code: {code}')
      self.__code1 = code
      self.__code3 = ONE_TO_THREE[code]
    elif n == 3:  # Given three-letter code
      if code not in THREE_TO_ONE.keys():
        raise ValueError(f'Invalid three-letter code: {code}')
      self.__code3 = code
      self.__code1 = THREE_TO_ONE[code]
    else:
      raise ValueError(f'Invalid one- or three-letters code: {code}')

  def get_one_letter_code(self):
    return self.__code1

  def get_three_letters_code(self):
    return self.__code3


class Residues(abc.ABC):
  """
  Utility class to obtain residues list
  """
  @staticmethod
  def from_sequence(sequence: str) -> List[Residue]:
    return [Residue(res_id, code=c) for res_id, c in enumerate(sequence)]


class Chain:
  def __init__(self, name: str = None, protein=None, residues: List[Residue] = None, device: str = 'cpu'):
    self.name = name
    self.protein = protein
    self.residues = residues if residues is not None else []
    # Setting the same device of protein if given
    self.device = protein.device if protein is not None else device
    self.__coords = None  # Caching coords

  def sequence(self):
    return ''.join([res.get_one_letter_code() for res in self.residues])

  def coords(self):
    if self.__coords is None:
      coords = []
      for residue in self.residues:
        for atom in residue.atoms:
          if atom.coords is not None:
            coords.append(atom.coords)
      self.__coords = torch.tensor(coords).to(self.device)
    return self.__coords


class Protein:
  def __init__(self, name, file: str = None,
               chains: List[Chain] = None, props: Dict[str, Any] = None, device: str = 'cpu'):
    self.name = name
    self.file = file  # Sequence or PDB file
    self.chains = chains if chains is not None else []
    self.props = props if props is not None else {}
    self.device = device
    self.__sequence = None  # Cache for sequence
    self.__coords = None  # Cache for coordinates
    self.__rog = None  # Cache for Radius of Gyration

  def sequence(self, separator: str = '') -> str:
    if self.__sequence is None:
      self.__sequence = separator.join([chain.sequence() for chain in self.chains])
    return self.__sequence

  def coords(self) -> torch.Tensor:
    """
    Gets the coordinates of all atoms in the protein
    :return: A torch.Tensor with all atom coordinates
    """
    if self.__coords is None and len(self.chains) > 0:
      # Moving chains to protein device
      for chain in self.chains:
        chain.device = self.device
      return torch.cat([chain.coords() for chain in self.chains])
    return self.__coords

  def rog(self):
    if self.__rog is None:
      atom_one_hot = atoms_to_one_hot([self.__get_atom_symbols()], device=self.device)
      coords = self.coords().unsqueeze(0)
      self.__rog = rog(coords, atom_one_hot, device=self.device).squeeze()
    return self.__rog

  def __get_atom_symbols(self):
    atom_symbols = []
    for chain in self.chains:
      for residue in chain.residues:
        for atom in residue.atoms:
          atom_symbols.append(atom.symbol)
    return atom_symbols
