import abc
import os.path

import docker
from pathvalidate import sanitize_filename
from typing import List

from da4vid.model.proteins import Epitope
from da4vid.model.samples import SampleSet, Sample


class PipelineStep(abc.ABC):

  def __init__(self, name: str, parent=None, folder: str | None = None):
    """
    Creates an abstract step.
    :param name: The name of the step
    :param parent: The parent step
    :param folder: The folder (relative or absolute) in which the step is executed
    """
    self.name = name
    self.parent = parent
    self.folder = folder
    self.__ctx_folder = self.folder if self.folder and os.path.isabs(self.folder) else None
    self.__identifier = None  # Caching the identifier

  @abc.abstractmethod
  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Abstract method executing the concrete step.
    :param sample_set: The set of samples on which execute the method
    :return: Anything useful the concrete method wishes to return
    """
    pass

  def get_context_folder(self):
    """
    Gets the folder in which inner steps will be executed.
    :return: The folder used for I/O operations of inner steps
    """
    if not self.__ctx_folder:
      self.__ctx_folder = self._sanitize_filename(self.folder if self.folder else self.name)
      if self.parent:
        self.__ctx_folder = os.path.join(self.parent.get_context_folder(), self.__ctx_folder)
    return self.__ctx_folder

  def get_step_identifier(self) -> str:
    """
    Gets all the path from the pipeline root to this step, dot-separated.
    :return: The name of all ancestor steps and the name of this object,
             dot-separated
    """
    if self.__identifier:
      return self.__identifier
    self.__identifier = f'{self.parent.get_step_identifier() if self.parent else ""}.{self.name}'
    return self.__identifier

  @staticmethod
  def _sanitize_filename(filename: str) -> str:
    return sanitize_filename(filename.replace(' ', '_'))


class CompositeStep(PipelineStep):
  def __init__(self, steps: List[PipelineStep] = None, **kwargs):
    """
    Creates a composite step, composed of a collection of iteration steps.
    :param name: The name of the iteration
    :param parent: The Composite step which is parent of this step. If none, this
                   step is the root step
    :param folder: The optional folder in which execute file I/O operations of
                   inner steps. If it is a relative path, it is relative to the
                   parent step. If not given, the name of the step will be used
                   as the relative path (sanitized as well to be a valid filename)
    :param steps: The list of steps included in the iteration. Defaults to
                  no steps
    """
    super().__init__(**kwargs)
    self.steps = steps if steps else []

  def add_step(self, steps: PipelineStep | List[PipelineStep]) -> None:
    """
    Adds one or more pipeline steps at the end of the alredy inserted
    steps.
    :param steps: The single step or a list of steps which will be added
    """
    if isinstance(steps, PipelineStep):
      steps = [steps]
    self.steps += steps
    # Setting parent
    for step in self.steps:
      step.parent = self

  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes all steps in the collection.
    :return: The sample set obtained by performing all steps sequentially
    """
    for step in self.steps:
      sample_set = step.execute(sample_set)
    return sample_set


class PipelineRootStep(CompositeStep):
  def __init__(self, name: str, antigen: Sample, epitope: Epitope, folder: str):
    if folder is None:
      raise ValueError(f'Root pipeline folder needed')
    if os.path.isfile(folder):
      raise FileExistsError(f'Root folder is a regular file: {self.folder}')
    super().__init__(name=name, parent=None, folder=folder)
    self.antigen = antigen
    self.epitope = epitope


class DockerStep(PipelineStep, abc.ABC):
  def __init__(self, client: docker.DockerClient, image: str, **kwargs):
    super().__init__(**kwargs)
    self.client = client
    self.image = image
