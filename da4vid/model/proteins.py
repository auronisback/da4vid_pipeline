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
    for atom in self.atoms:  # Adding residue to atoms
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


class Chains:
  """
  Utility class to instantiate chains.
  """
  @staticmethod
  def from_sequence(sequence: str, chain_separator: str = ':') -> List[Chain]:
    if not sequence:
      return []
    chains = []
    chain_id = 'A'
    for chain in sequence.split(chain_separator):
      chains.append(Chain(chain_id, residues=Residues.from_sequence(chain)))
      chain_id = chr(ord(chain_id) + 1)
    return chains


class Protein:
  """
  Models a Protein.
  """

  def __init__(self, name, filename: str = None,
               chains: List[Chain] = None, props: Dict[str, Any] = None, device: str = 'cpu'):
    self.name = name
    self.filename = filename  # Sequence or PDB file
    self.chains = chains if chains is not None else []
    # Linking chains to this protein
    for chain in self.chains:
      chain.protein = self
    self.props = NestedDict(props)
    self.device = device
    self.__sequence = None  # Cache for sequence
    self.__coords = None  # Cache for coordinates
    self.__rog = None  # Cache for Radius of Gyration (NOTE: is this needed?)
    self.__ca_coords = None  # Cache for coordinates of C-alpha atoms

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

  def get_atom_symbols(self) -> List[str]:
    """
    Gets the list of all atom symbols in this protein.
    :return: The list of atom symbols, in the order they
             appear in chains and residues
    """
    atom_symbols = []
    for chain in self.chains:
      for residue in chain.residues:
        for atom in residue.atoms:
          atom_symbols.append(atom.symbol)
    return atom_symbols

  def has_chain(self, name: str) -> bool:
    """
    Checks if the protein has a chain with the specified name.
    :param name: The name of the searched chain
    :return: True if this protein has a chain with the input
             name, False otherwise
    """
    for chain in self.chains:
      if chain.name == name:
        return True
    return False

  def get_chain(self, name: str) -> Chain:
    """
    Gets a chain with a specific name from the protein.
    :param name: The name of searched chain
    :return: The chain in the protein with the given name
    :raise ValueError: If no chain with the given name is
                       present in the protein
    """
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

  def ca_coords(self) -> torch.Tensor:
    """
    Returns the coordinate tensor of all C-alphas atoms in the
    whole protein.
    :return: A torch.Tensor with the coordinates of all C-alpha
             atoms
    """
    if self.__ca_coords is None:
      ca_coords = []
      for chain in self.chains:
        for resi in chain.residues:
          for atom in resi.atoms:
            if atom.code == 'CA':
              ca_coords.append([*atom.coords])
      self.__ca_coords = torch.tensor(ca_coords)
    return self.__ca_coords

  def add_prop(self, key: str, value: Any) -> None:
    """
    Adds a property to the protein, parsing its key according to dots.
    :param key: The name of the prop, in dot notation
    :param value: The value related to the property
    """
    self.props.add_value(key, value)

  def get_prop(self, key: str) -> Any | None:
    """
    Gets a property of the proteins using its dot separated key if it is present.
    :param key: The key of the prop in dot notation
    :return: The value for the key or None if the key is not present
    """
    return self.props.get_value(key)

  def has_prop(self, key: str) -> bool:
    """
    Check if the given property key is present in the props.
    :param key: The props key, in dot notation
    :return: True if the prop is present, false otherwise
    """
    return self.props.has_key(key)


class Proteins:
  """
  Class providing utilities for proteins.
  """

  @staticmethod
  def merge_sequence_with_structure(seq: Protein, struct: Protein) -> Protein:
    """
    Merges two proteins, taking the sequence from the first one and the
    structure of the second, merging also their properties.
    :param seq: The protein used to retain the sequence
    :param struct: The protein used to obtain atoms structure
    :return: The seq protein with atom coordinates and props
             linked from the struct protein
    :raise Error: If the chains and number of residues of the seq protein
                  does not match with the corresponding values of the
                  struct protein
    """
    Proteins.__check_seq_and_struct_coherence(seq, struct)
    # Changing protein file
    seq.filename = struct.filename
    # Adding props
    seq.props.merge(struct.props)
    # Adding atoms and coordinates
    for chain_seq in seq.chains:
      chain_struct = struct.get_chain(chain_seq.name)
      for resi_seq, resi_struct in zip(chain_seq.residues, chain_struct.residues):
        resi_seq.atoms = []
        for atom in resi_struct.atoms:
          atom.residue = resi_seq
          resi_seq.atoms.append(atom)
    return seq

  @staticmethod
  def __check_seq_and_struct_coherence(seq: Protein, struct: Protein) -> None:
    if not all([struct.has_chain(c.name) for c in seq.chains]):
      raise ValueError('chains mismatch between sequence and structure protein')
    for chain_seq in seq.chains:
      chain_struct = struct.get_chain(chain_seq.name)
      if len(chain_seq.residues) != len(chain_struct.residues):
        raise ValueError(f'chain {chain_seq.name} has a different number'
                         f' of residues between sequence and structure protein')

  @staticmethod
  def from_sequence(name: str, sequence: str, chain_separator: str = ':') -> Protein:
    return Protein(name, chains=Chains.from_sequence(sequence, chain_separator))


class NestedDict:
  def __init__(self, dictionary: Dict[str, Any] = None):
    """
    Created a new nested dictionary object, optionally specifying
    a default dictionary.
    :param dictionary: If given, the nested dictionary will be inited
                       with the given key-value pairs
    """
    self.dict = dictionary if dictionary else {}

  def add_value(self, key: str, value: Any) -> None:
    """
    Adds a value to this nested dictionary specifying its
    dot-separated key.
    :param key: The value's dot-separated key related
    :param value: The value which has to be added in the nested dictionary
    """
    tokens = key.split('.')
    actual_dict = self.dict
    for token in tokens[:-1]:
      # Adding dictionaries if nested properties is not present in actual dictionary for all but the last
      if token not in actual_dict.keys():
        actual_dict[token] = {}
      actual_dict = actual_dict[token]
    # Adding last token with value
    actual_dict[tokens[-1]] = value

  def has_key(self, key: str) -> bool:
    """
    Check if a key is present in this nested dictionary.
    :param key: The dot-separated key which needs to be searched
    :return: Whether the key is present or not in this object
    """
    return self.get_value(key) is not None

  def get_value(self, key: str) -> Any:
    """
    Gets the value related to the given key.
    :param key: The dot-separated key
    :return: The value related to the given key, or None if no
             key is present
    """
    tokens = key.split('.')
    actual_dict = self.dict
    for token in tokens[:-1]:
      if token not in actual_dict.keys():
        return None
      actual_dict = actual_dict[token]
    return actual_dict.get(tokens[-1], None)

  def merge(self, other) -> None:
    """
    Merges this dictionary with another nested dictionary object.
    :param other: The other nested dictionary
    """
    self.dict = self.__merge_rec(self.dict, other.dict)

  @staticmethod
  def __merge_rec(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    for key in d2:
      if key in d1:
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
          d1[key] = NestedDict.__merge_rec(d1[key], d2[key])
        elif d1[key] != d2[key]:
          raise ValueError(f'Conflict at key {key}')
      else:
        d1[key] = d2[key]
    return d1
