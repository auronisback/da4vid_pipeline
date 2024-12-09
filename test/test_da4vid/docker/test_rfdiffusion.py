import os.path
import shutil
import unittest
import warnings

import docker
import dotenv

from da4vid.docker.base import BaseContainer
from da4vid.docker.rfdiffusion import RFdiffusionContigMap, RFdiffusionPotentials, RFdiffusionContainer
from da4vid.gpus.cuda import CudaDeviceManager
from da4vid.model.proteins import Protein
from da4vid.io.pdb_io import read_from_pdb
from test.cfg import RESOURCES_ROOT, DOTENV_FILE
from test.test_da4vid.docker.helpers import duplicate_image, remove_duplicate_image


class RFdiffusionContigMapTest(unittest.TestCase):

  @staticmethod
  def __load_protein() -> Protein:
    return read_from_pdb(
      os.path.join(RESOURCES_ROOT, 'docker_test', 'rfdiffusion_test', 'rfdiffusion_test.pdb'))

  def test_add_random_length_contig_with_min_and_max(self):
    contig_map = RFdiffusionContigMap()
    contig_map.add_random_length_sequence(10, 15)
    self.assertEqual(1, len(contig_map.contigs), 'Invalid number of contigs')
    self.assertEqual(10, contig_map.contigs[0].min_length, 'Invalid min_length')
    self.assertEqual(15, contig_map.contigs[0].max_length, 'Invalid max_length')

  def test_add_random_length_contig_with_min_only(self):
    contig_map = RFdiffusionContigMap()
    contig_map.add_random_length_sequence(10)
    self.assertEqual(1, len(contig_map.contigs), 'Invalid number of contigs')
    self.assertEqual(10, contig_map.contigs[0].min_length, 'Invalid min_length')
    self.assertEqual(10, contig_map.contigs[0].max_length, 'max_length is not equal to min_length')

  def test_add_random_length_contig_with_max_lesser_than_min(self):
    contig_map = RFdiffusionContigMap()
    with self.assertRaises(ValueError):
      contig_map.add_random_length_sequence(10, 5)

  def test_add_fixed_sequence_contig_when_no_protein_given_raises_error(self):
    contig_map = RFdiffusionContigMap()
    with self.assertRaises(AttributeError):
      contig_map.add_fixed_sequence('A', 10, 15)

  def test_add_fixed_sequence_contig(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    contig_map.add_fixed_sequence('A', 10, 15)
    self.assertEqual(1, len(contig_map.contigs), 'More than one contig added')
    self.assertEqual('A', contig_map.contigs[0].chain.name, 'Invalid chain')
    self.assertEqual(10, contig_map.contigs[0].start, 'Invalid start')
    self.assertEqual(15, contig_map.contigs[0].end, 'Invalid end')

  def test_add_invalid_interval_for_fixed_sequence_contig(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    with self.assertRaises(ValueError):
      contig_map.add_fixed_sequence('A', 10, 5)

  def test_add_invalid_chain_for_fixed_sequence_contig(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    with self.assertRaises(ValueError):
      contig_map.add_fixed_sequence('Z', 10, 15)

  def test_add_chain_break_contig(self):
    contig_map = RFdiffusionContigMap()
    contig_map.add_chain_break()
    self.assertEqual(1, len(contig_map.contigs))

  def test_add_multiple_contigs(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    contig_map.add_random_length_sequence(10, 15)
    contig_map.add_fixed_sequence('A', 16, 25)
    contig_map.add_random_length_sequence(5)
    contig_map.add_chain_break()
    contig_map.add_fixed_sequence('A', 35, 45)
    self.assertEqual(5, len(contig_map.contigs), 'Invalid contigs length')
    self.assertEqual((10, 15),
                     (contig_map.contigs[0].min_length, contig_map.contigs[0].max_length),
                     'Invalid random sequence contig 0')
    self.assertEqual(('A', 16, 25),
                     (contig_map.contigs[1].chain.name, contig_map.contigs[1].start, contig_map.contigs[1].end),
                     'Invalid fixed sequence contig 1')
    self.assertEqual((5, 5),
                     (contig_map.contigs[2].min_length, contig_map.contigs[2].max_length),
                     'Invalid random sequence contig 2')
    self.assertEqual(('A', 35, 45),
                     (contig_map.contigs[4].chain.name, contig_map.contigs[4].start, contig_map.contigs[4].end),
                     'Invalid fixed sequence contig 4')

  def test_full_length_diffusion_contig(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    contig_map.full_diffusion()
    self.assertEqual(1, len(contig_map.contigs), 'Invalid contigs length')
    self.assertEqual(49, contig_map.contigs[0].min_length, 'Invalid min_length')
    self.assertEqual(49, contig_map.contigs[0].max_length, 'Invalid max_length')

  def test_full_length_diffusion_contig_raises_error_if_protein_not_set(self):
    contig_map = RFdiffusionContigMap()
    with self.assertRaises(AttributeError):
      contig_map.full_diffusion()

  def test_get_contigs_string_when_no_contigs_are_specified(self):
    contig_map = RFdiffusionContigMap()
    contig_string = contig_map.contigs_to_string()
    self.assertEqual('', contig_string, 'Invalid string for empty contigs')

  def test_get_contigs_string_with_only_one_random(self):
    contig_map = RFdiffusionContigMap()
    contig_map.add_random_length_sequence(10, 15)
    contig_string = contig_map.contigs_to_string()
    self.assertEqual('[10-15]', contig_string, 'Invalid contig string')

  def test_get_contigs_string_with_randoms_and_fixed(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    contig_map.add_random_length_sequence(10, 15)
    contig_map.add_fixed_sequence('A', 15, 22)
    contig_string = contig_map.contigs_to_string()
    self.assertEqual('[10-15/A15-22]', contig_string, 'Invalid contig string')

  def test_get_contigs_string_with_all_types_of_contigs(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    contig_map.add_random_length_sequence(10, 15)
    contig_map.add_fixed_sequence('A', 15, 22)
    contig_map.add_chain_break()
    contig_map.add_random_length_sequence(5, 10)
    contig_string = contig_map.contigs_to_string()
    self.assertEqual('[10-15/A15-22/0 5-10]', contig_string, 'Invalid contig string')

  def test_add_single_interval_to_provide_seq(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    contig_map.add_provide_seq(14, 24)
    self.assertTrue(contig_map.partial, 'Partial diffusion not correctly set')
    self.assertEqual(1, len(contig_map.provide_seq), 'Provide seq has not one element')
    self.assertEqual((13, 23), contig_map.provide_seq[0], 'Invalid added provide_seq')

  def test_add_invalid_interval_to_provide_seqs(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    with self.assertRaises(ValueError):
      contig_map.add_provide_seq(14, 4)

  def test_add_interval_greater_than_number_of_residues(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    with self.assertRaises(ValueError):
      contig_map.add_provide_seq(14, 49)

  def test_add_multiple_intervals_to_provide_seqs(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    contig_map.add_provide_seq(14, 24).add_provide_seq(38, 42)
    self.assertTrue(contig_map.partial, 'Partial diffusion not correctly set')
    self.assertEqual(2, len(contig_map.provide_seq), 'Provide seq has not one element')
    self.assertEqual((13, 23), contig_map.provide_seq[0], 'Invalid provide_seq at 0')
    self.assertEqual((37, 41), contig_map.provide_seq[1], 'Invalid provide_seq at 1')

  def test_get_provide_seq_string_when_provide_seq_is_empty(self):
    contig_map = RFdiffusionContigMap()
    ps_string = contig_map.provide_seq_to_string()
    self.assertEqual('', ps_string, 'Provide seq string is not empty')

  def test_get_provide_seq_string_with_single_seq(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    contig_map.add_provide_seq(14, 24)
    ps_string = contig_map.provide_seq_to_string()
    self.assertEqual('[13-23]', ps_string, 'Invalid provide_seq string')

  def test_get_provide_seq_string_with_multiple_seqs(self):
    contig_map = RFdiffusionContigMap(self.__load_protein())
    contig_map.add_provide_seq(14, 24).add_provide_seq(38, 42)
    ps_string = contig_map.provide_seq_to_string()
    self.assertEqual('[13-23,37-41]', ps_string, 'Invalid provide_seq string')


class RFdiffusionPotentialsTest(unittest.TestCase):
  def test_add_monomer_contacts_potential(self):
    potentials = RFdiffusionPotentials()
    potentials.add_monomer_contacts(5, 1.2)
    self.assertEqual(1, len(potentials.potentials), 'Invalid number of potentials')
    self.assertEqual('monomer_contacts', potentials.potentials[0]['type'], 'Invalid type')
    self.assertEqual(5, potentials.potentials[0]['r_0'], 'Invalid r_0 value')
    self.assertEqual(1.2, potentials.potentials[0]['weight'], 'Invalid weight value')

  def test_adding_invalid_monomer_contacts_r_0_should_raise_error(self):
    potentials = RFdiffusionPotentials()
    with self.assertRaises(ValueError):
      potentials.add_monomer_contacts(-5)

  def test_adding_invalid_monomer_contacts_weight_should_raise_error(self):
    potentials = RFdiffusionPotentials()
    with self.assertRaises(ValueError):
      potentials.add_monomer_contacts(5, -1)

  def test_add_monomer_rog_potential(self):
    potentials = RFdiffusionPotentials()
    potentials.add_rog(12.5, .7)
    self.assertEqual(1, len(potentials.potentials), 'Invalid number of potentials')
    self.assertEqual('monomer_ROG', potentials.potentials[0]['type'], 'Invalid type')
    self.assertEqual(12.5, potentials.potentials[0]['min_dist'], 'Invalid min_dist value')
    self.assertEqual(.7, potentials.potentials[0]['weight'], 'Invalid weight value')

  def test_adding_invalid_monomer_rog_min_dist_should_raise_error(self):
    potentials = RFdiffusionPotentials()
    with self.assertRaises(ValueError):
      potentials.add_rog(-1, .7)

  def test_adding_invalid_monomer_rog_weight_should_raise_error(self):
    potentials = RFdiffusionPotentials()
    with self.assertRaises(ValueError):
      potentials.add_rog(12.5, -1.7)

  def test_adding_multiple_potentials(self):
    potentials = RFdiffusionPotentials()
    potentials.add_rog(1.5, 2.7)
    potentials.add_monomer_contacts(7, 1.9)
    self.assertEqual(2, len(potentials.potentials), 'Invalid number of potentials')
    self.assertEqual('monomer_ROG', potentials.potentials[0]['type'], 'Invalid type')
    self.assertEqual(1.5, potentials.potentials[0]['min_dist'], 'Invalid min_dist value')
    self.assertEqual(2.7, potentials.potentials[0]['weight'], 'Invalid weight value')
    self.assertEqual('monomer_contacts', potentials.potentials[1]['type'], 'Invalid type')
    self.assertEqual(7, potentials.potentials[1]['r_0'], 'Invalid r_0 value')
    self.assertEqual(1.9, potentials.potentials[1]['weight'], 'Invalid weight value')

  def test_potentials_guide_decay_unspecified(self):
    potentials = RFdiffusionPotentials()
    self.assertEqual('linear', potentials.guide_decay)

  def test_potentials_guide_decay_constant(self):
    potentials = RFdiffusionPotentials()
    potentials.constant_decay()
    self.assertEqual('constant', potentials.guide_decay)

  def test_potentials_guide_decay_linear(self):
    potentials = RFdiffusionPotentials()
    potentials.constant_decay()  # Changing it before resetting
    potentials.linear_decay()
    self.assertEqual('linear', potentials.guide_decay)

  def test_potentials_guide_decay_quadratic(self):
    potentials = RFdiffusionPotentials()
    potentials.quadratic_decay()
    self.assertEqual('quadratic', potentials.guide_decay)

  def test_potentials_guide_decay_cubic(self):
    potentials = RFdiffusionPotentials()
    potentials.cubic_decay()
    self.assertEqual('cubic', potentials.guide_decay)

  def test_potentials_to_string_when_empty(self):
    potentials = RFdiffusionPotentials()
    pot_string = potentials.potentials_to_string()
    self.assertEqual('', pot_string)

  def test_potentials_to_string_with_monomer_rog(self):
    potentials = RFdiffusionPotentials()
    potentials.add_rog(11)
    pot_string = potentials.potentials_to_string()
    self.assertEqual('["type:monomer_ROG,min_dist:11,weight:1"]', pot_string)

  def test_potential_to_string_with_monomer_contacts(self):
    potentials = RFdiffusionPotentials()
    potentials.add_monomer_contacts(4)
    pot_string = potentials.potentials_to_string()
    self.assertEqual('["type:monomer_contacts,r_0:4,weight:1"]', pot_string)

  def test_potential_to_string_with_multiple_potentials(self):
    potentials = RFdiffusionPotentials()
    potentials.add_monomer_contacts(4).add_rog(11.3, 1.2)
    pot_string = potentials.potentials_to_string()
    self.assertEqual('["type:monomer_contacts,r_0:4,weight:1","type:monomer_ROG,min_dist:11.3,weight:1.2"]', pot_string)


class RFdiffusionTest(unittest.TestCase):

  def setUp(self):
    # Ignoring docker SDK warnings (still an unresolved issue in the SDK)
    warnings.simplefilter('ignore', ResourceWarning)
    self.resources_path = os.path.join(RESOURCES_ROOT, 'docker_test', 'rfdiffusion_test')
    self.input_dir = os.path.join(self.resources_path, 'inputs')
    os.makedirs(self.input_dir, exist_ok=True)
    self.output_dir = os.path.join(self.resources_path, 'outputs')
    os.makedirs(self.output_dir, exist_ok=True)
    self.input_pdb = os.path.join(self.resources_path, 'rfdiffusion_test.pdb')
    self.model_weights = dotenv.dotenv_values(DOTENV_FILE)['RFDIFFUSION_MODEL_FOLDER']
    self.client = docker.from_env()
    self.gpu_manager = CudaDeviceManager()
    duplicate_image(self.client, 'da4vid/rfdiffusion', 'rfdiff_duplicate')

  def tearDown(self):
    shutil.rmtree(self.input_dir)
    shutil.rmtree(self.output_dir)
    remove_duplicate_image(self.client, 'rfdiff_duplicate')
    self.client.close()

  def test_should_raise_error_if_invalid_image(self):
    rfdiff = RFdiffusionContainer(
      image='invalid_image',
      input_dir=self.input_dir,
      input_pdb=self.input_pdb,
      output_dir=self.output_dir,
      model_dir=self.model_weights,
      contig_map=RFdiffusionContigMap(),
      client=self.client,
      gpu_manager=self.gpu_manager
    )
    with self.assertRaises(BaseContainer.DockerImageNotFoundException):
      rfdiff.run()

  def test_should_correctly_execute_diffusion_with_default_image(self):
    protein = read_from_pdb(self.input_pdb)
    contigs = RFdiffusionContigMap(protein)
    contigs.add_random_length_sequence(10, 15)
    contigs.add_fixed_sequence('A', 16, 25)
    contigs.add_random_length_sequence(10, 15)
    res = RFdiffusionContainer(
      input_dir=self.input_dir,
      input_pdb=self.input_pdb,
      output_dir=self.output_dir,
      model_dir=self.model_weights,
      contig_map=contigs,
      num_designs=3,
      diffuser_T=15,
      client=self.client,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertTrue(res, 'RFdiffusion container stopped with errors!')
    diffused = [f for f in os.listdir(self.output_dir) if f.endswith('.pdb')]
    self.assertEqual(3, len(diffused))

  def test_should_correctly_execute_diffusion_with_specified_image(self):
    protein = read_from_pdb(self.input_pdb)
    contigs = RFdiffusionContigMap(protein)
    contigs.add_random_length_sequence(10, 15)
    contigs.add_fixed_sequence('A', 16, 25)
    contigs.add_random_length_sequence(10, 15)
    res = RFdiffusionContainer(
      image='rfdiff_duplicate',
      input_dir=self.input_dir,
      input_pdb=self.input_pdb,
      output_dir=self.output_dir,
      model_dir=self.model_weights,
      contig_map=contigs,
      num_designs=2,
      diffuser_T=15,
      client=self.client,
      gpu_manager=self.gpu_manager
    ).run()
    self.assertTrue(res, 'RFdiffusion container stopped with errors!')
    diffused = [f for f in os.listdir(self.output_dir) if f.endswith('.pdb')]
    self.assertEqual(2, len(diffused))


if __name__ == '__main__':
  unittest.main()
