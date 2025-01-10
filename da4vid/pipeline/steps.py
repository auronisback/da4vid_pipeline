import abc
import os.path
from typing import List

import docker
from pathvalidate import sanitize_filename

from da4vid.model.proteins import Epitope
from da4vid.model.samples import SampleSet, Sample


class PipelineStep(abc.ABC):
  """
  Abstracts a generic step in the pipeline.
  """

  def __init__(self, name: str, parent=None, folder: str | None = None, log_on_file: bool = True):
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
    self.out_logfile = os.path.join(self.get_context_folder(), 'stdout.log') if log_on_file else None
    self.err_logfile = os.path.join(self.get_context_folder(), 'stderr.log') if log_on_file else None

  @abc.abstractmethod
  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Abstract method executing the concrete step.
    :param sample_set: The set of samples on which execute the method
    :return: Anything useful the concrete method wishes to return
    """
    pass

  @abc.abstractmethod
  def output_folder(self) -> str:
    """
    Gets the output folder of the concrete step.
    :return: The string with the absolute path to the output folder
    """
    pass

  @abc.abstractmethod
  def input_folder(self) -> str:
    """
    Gets the input folder of the concrete step.
    :return: The string with the absolute path to the input folder
    """
    pass

  def get_context_folder(self):
    """
    Gets the folder in which inner steps will be executed.
    :return: The folder used for I/O operations of inner steps
    """
    if not self.__ctx_folder:
      self.__ctx_folder = self._sanitize_name(self.folder if self.folder else self.name)
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
  def _sanitize_name(name: str) -> str:
    """
    Sanitize a string in order to be used as file or folder name
    :param name: The name which will be sanitized
    :return: A sanitized version of name which can be used as pathname
    """
    return sanitize_filename(name.replace(' ', '_'))


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
    Adds one or more pipeline steps at the end of the already inserted
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

  def input_folder(self) -> str:
    """
    Gets the input folder of the composite step, which refers to the input
    folder of the first sub-step in this composite step.
    :return: The input folder of the 1st sub-step
    :raise AttributeError: if no steps have been set in this object
    """
    if self.steps:
      return self.steps[0].input_folder()
    raise AttributeError('No steps set in this composite step')

  def output_folder(self) -> str:
    """
    Gets the output folder of the composite step, which refers to the output
    folder of the last sub-step in this composite step.
    :return: The output folder of the last sub-step
    :raise AttributeError: if no steps have been set in this object
    """
    if self.steps:
      return self.steps[-1].output_folder()
    raise AttributeError('No steps set in this composite step')


class PipelineRootStep(CompositeStep):
  """
  Abstracts the root element of the pipeline.
  """
  def __init__(self, name: str, antigen: Sample, epitope: Epitope, folder: str):
    """
    Defines the root step of the pipeline, a composite step which includes every
    other step.
    :param name: The name of the pipeline
    :param antigen: The antigen sample used to start the pipeline
    :param epitope: The epitope around which scaffold
    :param folder: The context folder in which pipeline steps will be
                   executed
    """
    if folder is None:
      raise ValueError(f'Root pipeline folder needed')
    if os.path.isfile(folder):
      raise FileExistsError(f'Root folder is a regular file: {self.folder}')
    super().__init__(name=name, parent=None, folder=folder)
    self.antigen = antigen
    self.epitope = epitope

  def execute(self, sample_set: SampleSet = None) -> SampleSet:
    """
    Executes the pipeline. If sample set is not given, it will be automatically
    inferred by the antigen and the epitope parameters.
    :param sample_set: The sample set on which start the pipeline. If None, it
                       will be obtained by the antigen and epitope attributes
    :return: The sample set with produced de-novo antigens
    """
    if sample_set is None:
      sample_set = SampleSet()
      sample_set.add_samples(self.antigen)
    return super().execute(sample_set)


class DockerStep(PipelineStep, abc.ABC):
  """
  Abstracts a step in the pipeline which is executed as a Docker container.
  """
  def __init__(self, client: docker.DockerClient, image: str, **kwargs):
    """
    Initializes parameters common to steps executing operations in Docker containers.
    :param client: The client instance used to instantiate containers
    :param image: The image used by the containers in the step
    :param kwargs: Other common arguments to pipeline steps, such as
                   name and folder
    """
    super().__init__(**kwargs)
    self.client = client
    self.image = image
