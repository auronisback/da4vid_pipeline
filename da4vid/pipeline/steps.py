import abc
from typing import List

from da4vid.model.samples import SampleSet


class PipelineStep(abc.ABC):
  @abc.abstractmethod
  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Abstract method executing the concrete step.
    :param sample_set: The set of samples on which execute the method
    :return: Anything useful the concrete method wishes to return
    """
    pass


class PipelineIteration(PipelineStep):
  """
  Class composing one or more pipeline steps.
  """

  def __init__(self, name: str, steps: List[PipelineStep] = None):
    """
    Creates an iteration step.
    :param name: The name of the iteration
    :param steps: The list of steps included in the iteration. Defaults to
                  no steps
    """
    self.name = name
    self.steps = steps if steps else []

  def execute(self, sample_set) -> SampleSet:
    """
    Executes all steps in the run.
    :return: The sample set obtained by performing all steps in the iteration
    """
    for step in self.steps:
      sample_set = step.execute(sample_set)
    return sample_set

  def add_steps(self, *steps: List[PipelineStep]) -> None:
    """
    Adds one or more step to the pipeline.
    :param steps: The steps to add
    """
    self.steps += steps
