import unittest

import torch.cuda

from da4vid.gpus.cuda import CudaDevice


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
    cd = CudaDevice(0)
    print(cd.get_total_memory(), cd.get_free_memory(), cd.get_free_memory_ratio())
    print(cd.free_memory_str('G'))
    print(cd.free_memory_str('M'))
    print(cd.free_memory_str('k'))
    print(cd.free_memory_str('K'))
    print(cd.free_memory_str('B'))
    print(cd.free_memory_str('b'))


