from typing import List, Dict, Any, Tuple

import abc
import torch


class Atom:
  """
  Models an Atom in a protein.
  """

  BACKBONE_CODES = ['N', 'CA', 'C', 'O']

  def __init__(self, residue=None, number: int = None, code: str = None,
               coords: Tuple[float, float, float] = None, symbol: str = None, props: Dict[str, Any] = None):
    self.residue = residue
    self.number = number
    self.code = code
    self.coords = coords if coords is not None else [None, None, None]
    self.symbol = symbol
    self.props = props if props is not None else {}  # Generic atom properties


class Residue:
  """
  Models a Residue in a protein.
  """

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

  def __init__(self, residue_id: int, chain=None, number: int = None, code: str = None,
               atoms: List[Atom] = None, props: Dict[str, Any] = None):
    self.id = residue_id
    self.chain = chain
    self.number = number
    self.__code1 = self.__code3 = None
    if code is not None:
      self.set_code(code)
    self.atoms = atoms if atoms is not None else []
    for atom in self.atoms:  # Adding residues to atom
      atom.residue = self
    self.props = props if props is not None else {}  # Generic properties of this residue

  def set_code(self, code: str) -> None:
    """
    Sets the one- or three- letter code for this AA residue
    :param code: the one-letter or three-letter string with residue code
    """
    n = len(code)
    if n == 1:  # One-letter code, setting the three-letters code
      if code not in Residue.ONE_TO_THREE.keys():
        raise ValueError(f'Invalid one-letter code: {code}')
      self.__code1 = code
      self.__code3 = Residue.ONE_TO_THREE[code]
    elif n == 3:  # Given three-letter code
      if code not in Residue.THREE_TO_ONE.keys():
        raise ValueError(f'Invalid three-letter code: {code}')
      self.__code3 = code
      self.__code1 = Residue.THREE_TO_ONE[code]
    else:
      raise ValueError(f'Invalid one- or three-letters code: {code}')

  def get_one_letter_code(self):
    """
    Gets the one-letter code for the residue.
    :return: The single-character string with residue code
    """
    return self.__code1

  def get_three_letters_code(self):
    """
    Gets the three-letter code for the residue
    :return: The three-character string with residue code
    """
    return self.__code3

  def get_backbone_atoms(self) -> List[Atom]:
    """
    Gets only the backbone atoms in the residue (those with N, CA, C, O codes).
    :return: A list of backbone Atoms
    """
    atom_dict = {}
    for atom in self.atoms:
      if atom.code in Atom.BACKBONE_CODES:
        atom_dict[atom.code] = atom
    return [atom_dict['N'], atom_dict['CA'], atom_dict['C'], atom_dict['O']]


class Residues(abc.ABC):
  """
  Utility class to obtain residues list
  """
  @staticmethod
  def from_sequence(sequence: str) -> List[Residue]:
    """
    Gets a list of residues object from a one-letter code sequence of residues
    :param sequence: The one-letter code sequence
    :return: A list of Residues with the related code (without atoms)
    """
    return [Residue(res_id, code=c) for res_id, c in enumerate(sequence)]


class Chain:
  """
  Models a Chain in a protein.
  """
  def __init__(self, name: str = None, protein=None, residues: List[Residue] = None, device: str = 'cpu'):
    self.name = name
    self.protein = protein
    self.residues = residues if residues is not None else []
    for residue in self.residues:  # Linking this chain to residues
      residue.chain = self
    # Setting the same device of protein if given
    self.device = protein.device if protein is not None else device
    self.__coords = None  # Caching coords

  def sequence(self):
    """
    Gets the sequence of one-letter codes of AAs in the chain.
    :return: The sequence of AAs within the chain
    """
    return ''.join([res.get_one_letter_code() for res in self.residues])

  def coords(self):
    """
    Gets the coordinates of all atoms in the chain.
    :return: A torch.Tensor with coordinates of all atoms in the chain
    """
    if self.__coords is None:
      coords = []
      for residue in self.residues:
        for atom in residue.atoms:
          if atom.coords is not None:
            coords.append(atom.coords)
      self.__coords = torch.tensor(coords).to(self.device)
    return self.__coords

  def get_backbone_atoms(self) -> List[Atom]:
    """
    Get all backbone atoms of residues in this chain.
    :return: A list of all backbone atoms for each residue
    """
    return [atom for resi in self.residues for atom in resi.get_backbone_atoms()]


class Protein:
  """
  Models a Protein.
  """
  def __init__(self, name, filename: str = None,
               chains: List[Chain] = None, props: Dict[str, Any] = None, device: str = 'cpu'):
    self.name = name
    self.filename = filename  # Sequence or PDB file
    self.chains = chains if chains is not None else []
    self.props = props if props is not None else {}
    self.device = device
    self.__sequence = None  # Cache for sequence
    self.__coords = None  # Cache for coordinates
    self.__rog = None  # Cache for Radius of Gyration

  def sequence(self, separator: str = '') -> str:
    """
    Returns the whole sequence of one-letter code for all chains in the protein.
    :param separator: The separator string to split chains. Defaults to the empty string.
    :return: The sequence of one-letter codes of all AAs in all the protein's chains
    """
    if self.__sequence is None:
      self.__sequence = separator.join([chain.sequence() for chain in self.chains])
    return self.__sequence

  def coords(self) -> torch.Tensor:
    """
    Gets the coordinates of all atoms in the protein.
    :return: A torch.Tensor with all atom coordinates
    """
    if self.__coords is None and len(self.chains) > 0:
      # Moving chains to protein device
      for chain in self.chains:
        chain.device = self.device
      self.__coords = torch.cat([chain.coords() for chain in self.chains])
    return self.__coords

  def get_atom_symbols(self):
    atom_symbols = []
    for chain in self.chains:
      for residue in chain.residues:
        for atom in residue.atoms:
          atom_symbols.append(atom.symbol)
    return atom_symbols

  def has_chain(self, name: str) -> bool:
    for chain in self.chains:
      if chain.name == name:
        return True
    return False

  def get_chain(self, name: str) -> Chain:
    for chain in self.chains:
      if chain.name == name:
        return chain
    raise ValueError(f'chain {name} not found in protein {self.name}')

  def length(self):
    """
    Gets the length of all residues in the protein.
    :return: The number of AAs in the whole protein
    """
    return sum([len(chain.residues) for chain in self.chains])
