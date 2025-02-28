import abc
from typing import List

import torch


class CudaDevice:
  """
  Abstracts details of a CUDA gpu device.
  """
  def __init__(self, index: int):
    """
    Creates a new CUDA device by its index.
    :param index: The index of the device in the available GPU list
    :raise RuntimeError: if cuda is not available in torch
    :raise ValueError: if the index is not related to a valid CUDA device
    """
    if not torch.cuda.is_available():
      raise RuntimeError('cuda devices not available')
    if index > torch.cuda.device_count():
      raise ValueError(f'device with index {index} not available (device count: {torch.cuda.device_count()})')
    self.index = index
    self._props = torch.cuda.get_device_properties(index)
    self.name = f'cuda:{index}'
    self.gpu_name = self._props.name

  def get_total_memory(self) -> int:
    """
    Gets the total memory of this device, in bytes.
    :return: The total memory of the device
    """
    return torch.cuda.mem_get_info(self.index)[1]

  def get_free_memory(self) -> int:
    """
    Gets the amount of free memory on this device, in bytes.
    :return: The free available memory of this device
    """
    return torch.cuda.mem_get_info(self.index)[0]

  def get_free_memory_ratio(self) -> float:
    """
    Gets the ratio between free and total memory of this device.
    :return: The ratio between the free memory and the total memory of the device
    """
    free, total = torch.cuda.mem_get_info(self.index)
    return free / total

  def total_memory_str(self, unit: str = 'G') -> str:
    """
    Returns a string representing the total memory of the device, in the
    specified unit of measure.
    :param unit: The unit of measure for the memory, which can be:
                 - b: bits (yeah, but why?!)
                 - B: bytes
                 - k or K: kilobytes
                 - M: megabytes
                 - G: gigabytes
                 Defaults to gigabytes (G)
    :return: A string representing the total memory in the given unit of
             measure. If the unit is M or G, then it is rounded to the
             2nd decimal digit
    :raise ValueError: if the unit of measure is invalid
    """
    return self.__mem_to_str(self.get_total_memory(), unit)

  def free_memory_str(self, unit: str = 'G'):
    """
        Returns a string representing the free memory of the device, in the
        specified unit of measure.
        :param unit: The unit of measure for the memory, which can be:
                     - b: bits (again, why should someone need this?!)
                     - B: bytes
                     - k or K: kilobytes
                     - M: megabytes
                     - G: gigabytes
                     Defaults to gigabytes (G)
        :return: A string representing the free memory in the given unit of
                 measure. If the unit is M or G, then it is rounded to the
                 2nd decimal digit
        :raise ValueError: if the unit of measure is invalid
        """
    return self.__mem_to_str(self.get_free_memory(), unit)

  @staticmethod
  def __mem_to_str(mem: float, unit: str) -> str:
    match unit:
      case 'b':  # Bits?
        return f'{mem * 8}b'
      case 'B':  # Bytes
        return f'{mem}B'
      case 'K' | 'k':  # Kilobytes
        return f'{mem / (1 << 10)}kB'
      case 'M':
        return f'{round(mem / (1 << 20), 2)}MB'
      case 'G':
        return f'{round(mem / (1 << 30), 2)}GB'
      case _:
        raise ValueError(f'invalid unit of measure: {unit}')


class CudaDeviceManager:
  """
  Class managing CUDA devices in the pipeline.
  """
  class __Policy(abc.ABC):
    """
    Abstract class for different selection policies in the used devices.
    """

    @abc.abstractmethod
    def next_device(self) -> CudaDevice:
      """
      Returns the next device to use, according to the specific policy.
      :return: The CudaDevice object of the next device which should be used
               for pipeline operations
      """
      pass

  class __RoundRobin(__Policy):
    """
    Defines a round-robin policy to assign devices.
    """
    def __init__(self, manager: 'CudaDeviceManager'):
      self.manager = manager
      self.actual = 0

    def next_device(self) -> CudaDevice:
      """
      Gets the next device, according to a round-robin policy.
      :return: The next unused device
      """
      if self.actual == len(self.manager.devices):
        self.actual = 0
      act = self.actual
      self.actual += 1
      return self.manager.devices[act]

  class __MostFreeDevice(__Policy):
    """
    Defines a policy which each time returns the device with the most free
    memory absolute value.
    """
    def __init__(self, manager):
      self.manager = manager

    def next_device(self) -> CudaDevice:
      """
      Gets the device with the most available memory at calling time.
      :return: The device which has the more absolute value of free memory
      """
      most_free_device = None
      max_free_memory = 0
      for device in self.manager.devices:
        free_memory = device.get_free_memory()
        if free_memory > max_free_memory:
          max_free_memory = free_memory
          most_free_device = device
      return most_free_device

  def __init__(self, policy: str = 'round_robin'):
    """
    Creates a new device manager with the specified policy.
    :param policy: The used policy, which can be:
                   - round_robin: a round-robin policy. Default option.
                   - most_free: uses a most-free device policy
    :raise RuntimeError: if cuda is not available
    """
    if not torch.cuda.is_available():
      raise RuntimeError('CUDA is not available')
    self.devices = [CudaDevice(i) for i in range(torch.cuda.device_count())]
    self.__policy: CudaDeviceManager.__Policy
    match policy:
      case 'round_robin':
        self.__policy = CudaDeviceManager.__RoundRobin(self)
      case 'most_free':
        self.__policy = CudaDeviceManager.__MostFreeDevice(self)
      case _:
        raise ValueError(f'invalid device policy: {policy}')

  def next_devices(self, num: int = 1) -> CudaDevice | List[CudaDevice]:
    """
    Gets the next devices to use, according to the specified policy.
    :param num: The number of devices needed, according to the policy. Defaults to 1.
    :return: A list with num of CUDA devices. If num parameter is greater than the
             maximum number of CUDA devices, all devices will be returned
    """
    if len(self.devices) < num:
      num = len(self.devices)
    return [self.__policy.next_device() for _ in range(num)]

  def next_device(self) -> CudaDevice:
    """
    Gets the next device according to the specified policy.
    :return: The CudaDevice object to be used for the next pipeline operation
    """
    return self.next_devices(1)[0]
