from typing import List, Dict, Any, Tuple

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

ONE_TO_THREE = {key: value.upper() for key, value in IUPAC_DATA}
THREE_TO_ONE = {value: key for key, value in ONE_TO_THREE.items()}


class Atom:
  def __init__(self, residue=None, number: int = None, code: str = None,
               coords: Tuple[float, float, float] = None, props: Dict[str, Any] = None):
    self.residue = residue
    self.number = number
    self.code = code
    self.coords = coords if coords is not None else [None, None, None]
    self.props = props if props is not None else {}  # Generic atom properties


class Residue:
  def __init__(self, residue_id: int, chain = None, number: int = None, code3: str = None, code1: str = None,
               atoms: List[Atom] = None, props: Dict[str, Any] = None):
    self.id = residue_id
    self.chain = chain
    self.number = number
    self.code3 = code3
    if code3 is not None and code3 in THREE_TO_ONE.keys():
      self.code1 = THREE_TO_ONE[code3]
    self.code1 = code1
    if code1 is not None and code1 in ONE_TO_THREE.keys():
      self.code3 = ONE_TO_THREE[code1]
    self.atoms = atoms if atoms is not None else []
    self.props = props if props is not None else {}  # Generic properties of this residue


class Chain:
  def __init__(self, name: str = None, protein=None, residues: List[Residue] = None):
    self.name = name
    self.protein = protein
    self.residues = residues if residues is not None else []

  def sequence(self):
    return ''.join([res.code1 for res in self.residues])


class Protein:
  def __init__(self, name, file: str = None,
               chains: List[Chain] = None, props: Dict[str, Any] = None):
    self.name = name
    self.file = file  # Sequence or PDB file
    self.chains = chains if chains is not None else []
    self.props = props if props is not None else {}

  def sequence(self) -> str:
    return ''.join([chain.sequence() for chain in self.chains])
