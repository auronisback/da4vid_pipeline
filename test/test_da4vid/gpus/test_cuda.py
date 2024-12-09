import unittest

import torch.cuda

from da4vid.gpus.cuda import CudaDevice, CudaDeviceManager


class CudaDeviceTest(unittest.TestCase):

  @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
  @unittest.skipIf(torch.cuda.device_count() < 1, 'no available CUDA devices')
  def test_device_instantiation_raise_error_if_invalid_index(self):
    import torch
    count = torch.cuda.device_count()
    with self.assertRaises(ValueError):
      CudaDevice(count + 2)

  @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
  @unittest.skipIf(torch.cuda.device_count() < 1, 'no available CUDA devices')
  def test_gpu_device_created_correctly(self):
    CudaDevice(0)


class CudaDeviceManagerTest(unittest.TestCase):

  @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
  def test_cuda_device_manager_creation_with_default_policy(self):
    manager = CudaDeviceManager()
    self.assertEqual(torch.cuda.device_count(), len(manager.devices))
    # Asserting circularity
    self.assertEqual(manager.devices[0], manager.next_device())
    self.assertEqual(manager.devices[1], manager.next_device())
    self.assertEqual(manager.devices[0], manager.next_device())
    self.assertEqual(manager.devices[1], manager.next_device())

  @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
  def test_cuda_device_manager_creation_with_explicit_round_robin_policy(self):
    manager = CudaDeviceManager('round_robin')
    self.assertEqual(torch.cuda.device_count(), len(manager.devices))
    # Asserting circularity
    self.assertEqual(manager.devices[0], manager.next_device())
    self.assertEqual(manager.devices[1], manager.next_device())
    self.assertEqual(manager.devices[0], manager.next_device())
    self.assertEqual(manager.devices[1], manager.next_device())

  @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
  def test_cuda_device_manager_creation_with_most_free_device_policy(self):
    manager = CudaDeviceManager('most_free')
    self.assertEqual(torch.cuda.device_count(), len(manager.devices))
    # By now, let's just ensure that the same device is returned if nothing
    # is allocated on the GPU
    self.assertEqual(manager.next_device(), manager.next_device())

  @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
  def test_cuda_device_manager_returns_specified_number_of_devices(self):
    manager = CudaDeviceManager()
    devices = manager.next_devices(len(manager.devices))
    self.assertEqual(len(manager.devices), len(devices))

  @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
  def test_cuda_device_manager_returns_at_max_all_devices(self):
    manager = CudaDeviceManager()
    devices = manager.next_devices(len(manager.devices) + 1)
    self.assertEqual(len(manager.devices), len(devices))
