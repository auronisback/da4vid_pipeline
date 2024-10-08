import unittest

import numpy as np
import torch

from da4vid.io import read_from_pdb, read_pdb_folder
from da4vid.model import Protein, Chain, Residue, Atom
from test.cfg import RESOURCES_ROOT, TEST_GPU

from da4vid.metrics.kabsch import kabsch, rmsd


class TestKabsch(unittest.TestCase):

  def test_kabsch_identical_tensors(self):
    A = torch.Tensor([[1.0, 2.0, 2.0],
                      [2.0, 1.0, -1.0],
                      [20.0, 10.0, 15.0]])
    R = torch.eye(3)
    t = torch.zeros(3)
    rmsd_val, R_opt, t_opt = kabsch(A, A)
    self.assertTrue(torch.allclose(R_opt, R, atol=1e-5, rtol=1e-8),
                    f'Rotation matrix is not identical: {R_opt}')
    self.assertTrue(torch.allclose(t_opt, t, atol=1e-5, rtol=1e-8),
                    f'Translation vector is not zero: {t_opt}')

  def test_kabsch_only_translation(self):
    A = torch.Tensor([[10.0, 10.0, 10.0],
                      [20.0, 10.0, 10.0],
                      [20.0, 10.0, 15.0]])
    R = torch.eye(3)
    t = torch.Tensor([4, 5, 6])
    _, R_opt, t_opt = kabsch(A, A + t)
    self.assertTrue(torch.allclose(R_opt, R, atol=1e-5, rtol=1e-8),
                    f'Rotation matrix is not identical: {R_opt}')
    self.assertTrue(torch.allclose(t_opt, t, atol=1e-5, rtol=1e-8),
                    f'Translation vector is not close enough:\nexpected: {t},\nactual: {t_opt}')

  def test_kabsch_only_rotation(self):
    A = torch.Tensor([[10.0, 10.0, 10.0],
                      [20.0, 10.0, 10.0],
                      [20.0, 10.0, 15.0]])
    alpha = beta = gamma = 45.0
    R = self.__get_rotation_matrix(alpha, beta, gamma)
    t = torch.zeros(3)
    _, R_opt, t_opt = kabsch(A, A.matmul(R.transpose(-2, -1)))
    self.assertTrue(torch.allclose(R_opt, R, atol=1e-5, rtol=1e-8),
                    f'Rotation matrix is not close enough:\nexpected: {R},\nactual:{R_opt}')
    self.assertTrue(torch.allclose(t_opt, t, atol=1e-5, rtol=1e-8),
                    f'Translation vector is not close enough:\nexpected: {t},\nactual: {t_opt}')

  def test_kabsch_both_rotation_and_translation(self):
    A = torch.Tensor([[10.0, 10.0, 10.0],
                      [20.0, 10.0, 10.0],
                      [20.0, 10.0, 15.0]])
    alpha = beta = gamma = 33
    R = self.__get_rotation_matrix(alpha, beta, gamma)
    t = torch.Tensor([4, -1, 3])
    _, R_opt, t_opt = kabsch(A, A.matmul(R.T) + t)
    self.assertTrue(torch.allclose(R_opt, R, atol=1e-5, rtol=1e-8),
                    f'Rotation matrix is not close enough:\nexpected: {R},\nactual:{R_opt}')
    self.assertTrue(torch.allclose(t_opt, t, atol=1e-5, rtol=1e-8),
                    f'Translation vector is not close enough: expected: {t},\nactual: {t_opt}')

  def test_kabsch_rmsd(self):
    A = torch.Tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [9.0, 8.0, 7.0]])
    alpha = 60
    beta = 30
    gamma = 90
    R = self.__get_rotation_matrix(alpha, beta, gamma)
    t = torch.Tensor([2, 11, -3])
    rmsd_val, R_opt, t_opt = kabsch(A, A.matmul(R.T) + t)
    self.assertTrue(torch.allclose(rmsd_val, torch.Tensor(0), atol=1e-5))

  def test_kabsch_batch(self):
    B = torch.Tensor([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [9.0, 8.0, 7.0]],
                      [[10.0, 10.0, 10.0],
                       [20.0, 10.0, 10.0],
                       [20.0, 10.0, 15.0]]
                      ])
    alpha = [30, 60]
    beta = [60, 10]
    gamma = [45, 0]
    R = torch.stack([self.__get_rotation_matrix(alpha[0], beta[0], gamma[0]),
                     self.__get_rotation_matrix(alpha[1], beta[1], gamma[1])])
    t = torch.Tensor([[4., 3., 2.], [10., 5., -7.]])
    BB = B.matmul(R.transpose(-2, -1)) + t.unsqueeze(1)
    rmsd_val, R_opt, t_opt = kabsch(B, BB)

    self.assertEqual(R_opt.shape, (2, 2, 3, 3), f'Rotation matrix shape is {R_opt.shape}, expected {2, 2, 3, 3}')
    self.assertEqual(t_opt.shape, (2, 2, 3), f'Translation tensor shape is {t_opt.shape}, expected {2, 2, 3}')
    self.assertEqual(rmsd_val.shape, (2, 2), f'RMSD shape is {rmsd_val.shape}, expected {2, 2}')
    # Checking only on diagonals
    self.assertTrue(torch.allclose(R_opt[0, 0], R[0], atol=1e-5, rtol=1e-8),
                    f'Rotation matrix is not close enough:\nexpected: {R[0]},\nactual:{R_opt[0, 0]}')
    self.assertTrue(torch.allclose(R_opt[1, 1], R[1], atol=1e-5, rtol=1e-8),
                    f'Rotation matrix is not close enough:\nexpected: {R[1]},\nactual:{R_opt[1, 1]}')
    self.assertTrue(torch.allclose(t_opt[0, 0], t[0], atol=1e-5, rtol=1e-8),
                    f'Translation vector is not close enough: expected: {t[0]},\nactual: {t_opt[0, 0]}')
    self.assertTrue(torch.allclose(t_opt[1, 1], t[1], atol=1e-5, rtol=1e-8),
                    f'Translation vector is not close enough: expected: {t[1]},\nactual: {t_opt[1, 1]}')
    self.assertTrue(torch.allclose(rmsd_val.diag(), torch.zeros(2), atol=1e-5),
                    f'RMSD is not close enough to zero: {rmsd_val}')

  def test_batch_on_cuda(self):
    self.assertTrue(torch.cuda.is_available(), f'CUDA is not available!')
    B = torch.Tensor([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [9.0, 8.0, 7.0]],
                      [[10.0, 10.0, 10.0],
                       [20.0, 10.0, 10.0],
                       [20.0, 10.0, 15.0]]
                      ]).to(TEST_GPU)
    alpha = [30, 60]
    beta = [60, 10]
    gamma = [45, 0]
    R = torch.stack([self.__get_rotation_matrix(alpha[0], beta[0], gamma[0]),
                     self.__get_rotation_matrix(alpha[1], beta[1], gamma[1])]).to(TEST_GPU)
    t = torch.Tensor([[4., 3., 2.],
                      [10., 5., -7.]]).to(TEST_GPU)
    BB = B.matmul(R.transpose(-2, -1)) + t.unsqueeze(1)
    rmsd_val, R_opt, t_opt = kabsch(B, BB, device=TEST_GPU)

    self.assertEqual(R_opt.shape, (2, 2, 3, 3),
                     f'Rotation matrix shape is {R_opt.shape}, expected {2, 2, 3, 3}')
    self.assertEqual(t_opt.shape, (2, 2, 3),
                     f'Translation tensor shape is {t_opt.shape}, expected {2, 2, 3}')
    self.assertEqual(rmsd_val.shape, (2, 2),
                     f'RMSD shape is {rmsd_val.shape}, expected {2, 2}')
    # Checking only on diagonals
    self.assertTrue(torch.allclose(R_opt[0, 0], R[0], atol=1e-3, rtol=1e-8),
                    f'Rotation matrix is not close enough:\nexpected: {R[0]},\nactual:{R_opt[0, 0]}')
    self.assertTrue(torch.allclose(R_opt[1, 1], R[1], atol=1e-3, rtol=1e-8),
                    f'Rotation matrix is not close enough:\nexpected: {R[1]},\nactual:{R_opt[1, 1]}')
    self.assertTrue(torch.allclose(t_opt[0, 0], t[0], atol=5e-3, rtol=1e-8),
                    f'Translation vector is not close enough: expected: {t[0]},\nactual: {t_opt[0, 0]}')
    self.assertTrue(torch.allclose(t_opt[1, 1], t[1], atol=5e-3, rtol=1e-8),
                    f'Translation vector is not close enough: expected: {t[1]},\nactual: {t_opt[1, 1]}')
    self.assertTrue(torch.allclose(rmsd_val.diag(), torch.zeros(2, device=TEST_GPU), atol=5e-3),
                    f'RMSD is not close enough to zero: {rmsd_val}')

  def test_kabsch_uneven_batches(self):
    B = torch.Tensor([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [9.0, 8.0, 7.0]],
                      [[10.0, 10.0, 10.0],
                       [20.0, 10.0, 10.0],
                       [20.0, 10.0, 15.0]],
                      [[-1., -2., -3.],
                       [-3., -5., -6.],
                       [-7., -.8, -.9]]
                      ]).to(TEST_GPU)
    alpha = [30, 60]
    beta = [60, 10]
    gamma = [45, 0]
    R = torch.stack([self.__get_rotation_matrix(alpha[0], beta[0], gamma[0]),
                     self.__get_rotation_matrix(alpha[1], beta[1], gamma[1])]).to(TEST_GPU)
    t = torch.Tensor([[4., 3., 2.],
                      [10., 5., -7.]]).to(TEST_GPU)
    BB = B[0::2].matmul(R.transpose(-2, -1)) + t.unsqueeze(1)

    rmsd_val, R_opt, t_opt = kabsch(B, BB, device=TEST_GPU)
    self.assertEqual(R_opt.shape, (3, 2, 3, 3), f'Rotation matrix shape is {R_opt.shape}, expected {3, 2, 3, 3}')
    self.assertEqual(t_opt.shape, (3, 2, 3), f'Translation tensor shape is {t_opt.shape}, expected {3, 2, 3}')
    self.assertEqual(rmsd_val.shape, (3, 2), f'RMSD shape is {rmsd_val.shape}, expected {3, 2}')
    # Checking only on diagonals
    self.assertTrue(torch.allclose(R_opt[0, 0], R[0], atol=1e-3, rtol=1e-8),
                    f'Rotation matrix is not close enough:\nexpected: {R[0]},\nactual:{R_opt[0, 0]}')
    self.assertTrue(torch.allclose(R_opt[2, 1], R[1], atol=1e-3, rtol=1e-8),
                    f'Rotation matrix is not close enough:\nexpected: {R[1]},\nactual:{R_opt[2, 1]}')
    self.assertTrue(torch.allclose(t_opt[0, 0], t[0], atol=5e-3, rtol=1e-8),
                    f'Translation vector is not close enough: expected: {t[0]},\nactual: {t_opt[0, 0]}')
    self.assertTrue(torch.allclose(t_opt[2, 1], t[1], atol=5e-3, rtol=1e-8),
                    f'Translation vector is not close enough: expected: {t[1]},\nactual: {t_opt[2, 1]}')
    self.assertTrue(torch.allclose(rmsd_val[0, 0], torch.zeros(1, device=TEST_GPU), atol=5e-3),
                    f'RMSD is not close enough to zero: {rmsd_val}')
    self.assertTrue(torch.allclose(rmsd_val[2, 1], torch.zeros(1, device=TEST_GPU), atol=5e-3),
                    f'RMSD is not close enough to zero: {rmsd_val}')

  @staticmethod
  def __get_rotation_matrix(alpha, beta, gamma):
    R_x = torch.Tensor([[1.0, 0.0, 0.0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha), np.cos(alpha)]])
    R_y = torch.Tensor([[np.cos(beta), 0, np.sin(beta)],
                        [0, 1, 0],
                        [- np.sin(beta), 0, np.cos(beta)]])
    R_z = torch.Tensor([[np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]])
    return R_x @ R_y @ R_z


class TestRMSD(unittest.TestCase):

  def test_rmsd_should_raise_error_if_shapes_are_different(self):
    first = Protein('first', chains=[
      Chain('A', residues=[
        Residue(1, code='G', atoms=[Atom(code='CA', symbol='C', coords=(1., 2., 2.))]),
        Residue(2, code='A', atoms=[Atom(code='CA', symbol='C', coords=(2., 1., -1.))]),
        Residue(3, code='A', atoms=[Atom(code='CA', symbol='C', coords=(-1., 1., -1.))]),
        Residue(4, code='A', atoms=[Atom(code='CA', symbol='C', coords=(-2., 2., 3.))])
      ])
    ])
    second = Protein('first', chains=[
      Chain('A', residues=[
        Residue(1, code='G', atoms=[Atom(code='CA', symbol='C', coords=(1., 2., 2.))]),
        Residue(2, code='A', atoms=[Atom(code='CA', symbol='C', coords=(2., 1., -1.))]),
        Residue(4, code='A', atoms=[Atom(code='CA', symbol='C', coords=(-2., 2., 3.))])
      ])
    ])
    with self.assertRaises(ValueError):
      rmsd(first, second)

  def test_rmsd_one_vs_one_on_same_protein(self):
    first = Protein('first', chains=[
      Chain('A', residues=[
        Residue(1, code='G', atoms=[Atom(code='CA', symbol='C', coords=(1., 2., 2.))]),
        Residue(2, code='A', atoms=[Atom(code='CA', symbol='C', coords=(2., 1., -1.))]),
        Residue(3, code='A', atoms=[Atom(code='CA', symbol='C', coords=(-1., 1., -1.))]),
        Residue(4, code='A', atoms=[Atom(code='CA', symbol='C', coords=(-2., 2., 3.))])
      ])
    ])
    rmsd_val, R, t = rmsd(first, first, device=TEST_GPU)
    torch.testing.assert_close(torch.tensor([0.]).squeeze().to(TEST_GPU), rmsd_val, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(torch.eye(3).to(TEST_GPU), R)
    torch.testing.assert_close(torch.zeros(3).to(TEST_GPU), t)

  def test_rmsd_shape_for_one_vs_one(self):
    first_pdb = f'{RESOURCES_ROOT}/rmsd_test/orig.pdb'
    second_pdb = f'{RESOURCES_ROOT}/rmsd_test/sample_1000.pdb'
    first = read_from_pdb(first_pdb)
    second = read_from_pdb(second_pdb)
    rmsd_val, R, t = rmsd(first, second, device=TEST_GPU)
    self.assertEqual(0, rmsd_val.squeeze().ndim, 'Returned RMSD is not scalar')
    self.assertEqual(torch.Size([3, 3]), R.shape, 'Rotation matrix shape does not match')
    self.assertEqual(torch.Size([3]), t.shape, 'Translation vector shape does not match')

  def test_rmsd_shape_for_one_vs_all(self):
    first_pdb = f'{RESOURCES_ROOT}/rmsd_test/orig.pdb'
    second_list_pdb = f'{RESOURCES_ROOT}/rmsd_test'
    first = read_from_pdb(first_pdb)
    second = read_pdb_folder(second_list_pdb)
    rmsd_val, R, t = rmsd(first, second, device=TEST_GPU)
    self.assertEqual(torch.Size([len(second)]), rmsd_val.shape, 'Invalid RMSD shape')
    self.assertEqual(torch.Size([len(second), 3, 3]), R.shape, 'Invalid Rotation matrix shape')
    self.assertEqual(torch.Size([len(second), 3]), t.shape, 'Invalid translation vector shape')

  def test_rmsd_shape_for_all_vs_one(self):
    first_pdb = f'{RESOURCES_ROOT}/rmsd_test/orig.pdb'
    second_list_pdb = f'{RESOURCES_ROOT}/rmsd_test'
    first = read_from_pdb(first_pdb)
    second = read_pdb_folder(second_list_pdb)
    rmsd_val, R, t = rmsd(second, first, device=TEST_GPU)
    self.assertEqual(torch.Size([len(second)]), rmsd_val.shape, 'Invalid RMSD shape')
    self.assertEqual(torch.Size([len(second), 3, 3]), R.shape, 'Invalid Rotation matrix shape')
    self.assertEqual(torch.Size([len(second), 3]), t.shape, 'Invalid translation vector shape')

  def test_rmsd_shape_for_all_vs_all(self):
    first = [read_from_pdb(f'{RESOURCES_ROOT}/rmsd_test/orig.pdb'),
             read_from_pdb(f'{RESOURCES_ROOT}/rmsd_test/sample_1008.pdb')]
    second = read_pdb_folder(f'{RESOURCES_ROOT}/rmsd_test')
    rmsd_val, R, t = rmsd(first, second, device=TEST_GPU)
    self.assertEqual(torch.Size([len(first), len(second)]), rmsd_val.shape,
                     'Invalid RMSD shape')
    self.assertEqual(torch.Size([len(first), len(second), 3, 3]), R.shape,
                     'Invalid Rotation matrix shape')
    self.assertEqual(torch.Size([len(first), len(second), 3]), t.shape,
                     'Invalid translation vector shape')


if __name__ == '__main__':
  unittest.main()
