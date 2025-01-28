import abc
import os.path
from typing import List, TypeVar, Callable, Any

import docker
from pathvalidate import sanitize_filename

from da4vid.model.proteins import Epitope
from da4vid.model.samples import SampleSet, Sample

# Generic type for subclasses of specific steps
T = TypeVar('T', bound='PipelineStep')


class PipelineException(Exception):
  """
  Class abstracting all pipeline exceptions.
  """
  def __init__(self, message: str = ''):
    super().__init__()
    self.message = message


class PipelineStep(abc.ABC):
  """
  Abstracts a generic step in the pipeline.
  """

  def __init__(self, name: str, parent=None, folder: str | None = None, log_on_file: bool = True,
               pre_step_fn: List[Callable[[T, Any], None]] | Callable[[T, Any], None] = None,
               post_step_fn: List[Callable[[T, Any], None]] | Callable[[T, Any], None] = None,
               failed_step_fn: List[Callable[[T, PipelineException, Any], None]] |
                               Callable[[T, PipelineException, Any], None] = None,
               **callable_kwargs):
    """
    Creates an abstract step, specifying its name, its optional parent, the folder in
    which IO operations of the step should be executed, a flag indicating whether to
    print logs on file rather than stdout/stderr, and three functions (or list of functions)
    which will be called back at specific step events. The first two functions will
    be called respectively before and after, while the third is fired when errors in the
    pipeline occur and accepts the pipeline exception object as second parameters.
    :param name: The name of the step
    :param parent: The parent step
    :param folder: The folder (relative or absolute) in which the step is executed
    :param pre_step_fn: A function or a list of functions which needs to be executed
                        before the execution of the concrete step
    :param pre_step_fn: A function or a list of functions which needs to be executed
                        after the execution of the concrete step
    :param callable_kwargs: The kwargs dictionary to pass to callable functions
    """
    self.name = name
    self.parent = parent
    self.folder = folder
    self.__ctx_folder = self.folder if self.folder and os.path.isabs(self.folder) else None
    self.__identifier = None  # Caching the identifier
    self.out_logfile = os.path.join(self.get_context_folder(), 'stdout.log') if log_on_file else None
    self.err_logfile = os.path.join(self.get_context_folder(), 'stderr.log') if log_on_file else None
    self.__pre_step_fn = [] if not pre_step_fn else (
      pre_step_fn if isinstance(pre_step_fn, List) else [pre_step_fn])
    self.__post_step_fn = [] if not post_step_fn else (
      post_step_fn if isinstance(post_step_fn, List) else [post_step_fn])
    self.__failed_step_fn = [] if not failed_step_fn else (
      failed_step_fn if isinstance(failed_step_fn, List) else [failed_step_fn])
    self.__callable_kwargs = callable_kwargs | {}

  def execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes the step. It will execute any pre-step functions registered in the step,
    and will then execute the actual step logic. If the step ends successfully,
    each registered post-step function will be executed, otherwise failure functions
    will be fired and then the PipelineException will be rethrown.
    :param sample_set: The input sample set
    :return: The sample set after the step evaluation
    """
    try:
      for fn in self.__pre_step_fn:
        fn(self, **self.__callable_kwargs)
      result_set = self._execute(sample_set)
      for fn in self.__post_step_fn:
        fn(self, **self.__callable_kwargs)
      return result_set
    except PipelineException as e:
      for fn in self.__failed_step_fn:
        fn(self, e, **self.__callable_kwargs)
      raise e

  @abc.abstractmethod
  def _execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Abstract method executing the concrete step.
    :param sample_set: The set of samples on which execute the method
    :return: Anything useful the concrete method wishes to return
    """
    try:
      for fn in self.__pre_step_fn:
        fn(self, **self.__callable_kwargs)
      result_set = self._execute(sample_set)
      for fn in self.__post_step_fn:
        fn(self, **self.__callable_kwargs)
      return result_set
    except PipelineException as e:
      for fn in self.__failed_step_fn:
        fn(self, e, **self.__callable_kwargs)
      raise e

  def resume(self, sample_set) -> SampleSet:
    """
    Resumes the last execution of this step. Pre-, post- and failure-step callbacks will be
    fired according to the logic in the *execution* method.
    :param sample_set: The input sample set
    :return: The sample set after the step resuming
    """
    try:
      for fn in self.__pre_step_fn:
        fn(self, **self.__callable_kwargs)
      result_set = self._resume(sample_set)
      for fn in self.__post_step_fn:
        fn(self, **self.__callable_kwargs)
      return result_set
    except PipelineException as e:
      for fn in self.__failed_step_fn:
        fn(self, e, **self.__callable_kwargs)
      raise e

  @abc.abstractmethod
  def _resume(self, sample_set: SampleSet) -> SampleSet:
    """
    Method used to recover data from a previously executed step. This method
    will be called when a pipeline evaluation has been interrupted for some
    reason, and should be resumed. It should differ from the *execute* method
    as it should not execute heavy computations and instead rely on output
    files obtained in a previous execution.
    :param sample_set: The input sample set
    :return: The sample set object as if this step was executed again
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

  def register_pre_step_fn(self, pre_step_fn: List[Callable[[T, Any], None]] | Callable[[T, Any], None]) -> None:
    """
    Adds a new pre-step function (or a list of pre-step functions).
    :param pre_step_fn: The function or the list of functions which will
                        be run before the step execution
    """
    self.__pre_step_fn = self.__register_fn(self.__pre_step_fn, pre_step_fn)

  def register_post_step_fn(self, post_step_fn: List[Callable[[T, Any], None]] | Callable[[T, Any], None]) -> None:
    """
    Adds a new pre-step function (or a list of pre-step functions).
    :param post_step_fn: The function or the list of functions which will
                         be run after a successful step execution
    """
    self.__post_step_fn = self.__register_fn(self.__post_step_fn, post_step_fn)

  def register_failed_step_fn(self, failed_step_fn:
                              List[Callable[[T, PipelineException, Any], None]] |
                              Callable[[T, PipelineException, Any], None]) -> None:
    """
    Adds a new pre-step function (or a list of pre-step functions).
    :param failed_step_fn: The function or the list of functions which will
                           be fired if the step execution results in errors
    """
    self.__failed_step_fn = self.__register_fn(self.__failed_step_fn, failed_step_fn)

  @staticmethod
  def __register_fn(registered_fn, new_fn):
    return registered_fn + (new_fn if isinstance(new_fn, List) else [new_fn])


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

  def _execute(self, sample_set: SampleSet) -> SampleSet:
    """
    Executes all steps in the collection.
    :return: The sample set obtained by performing all steps sequentially
    """
    for step in self.steps:
      sample_set = step._execute(sample_set)
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

  def _resume(self, sample_set: SampleSet) -> SampleSet:
    for step in self.steps:
      sample_set = step._resume(sample_set)
    return sample_set


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

  def _execute(self, sample_set: SampleSet = None) -> SampleSet:
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
    return super()._execute(sample_set)

  def _resume(self, sample_set: SampleSet = None) -> SampleSet:
    if sample_set is None:
      sample_set = SampleSet()
      sample_set.add_samples(self.antigen)
    return super()._resume(sample_set)


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


class FoldCollectionStep(PipelineStep):
  """
  Utility Step to collect all folds for a specific model in a sample set into a new sample set.
  """

  def __init__(self, name: str, model: str, **kwargs):
    """
    Creates the step which will collect folds for the given model.
    :param name: The name of this step in the pipeline
    :param model: The model whose folds will be collected into a new sample set
    :param kwargs: Other arguments
    """
    super().__init__(name, **kwargs)
    self.model = model

  def _execute(self, sample_set: SampleSet) -> SampleSet:
    return sample_set.folded_sample_set(self.model)

  def _resume(self, sample_set: SampleSet) -> SampleSet:
    # It is equivalent to the execute step
    return self._execute(sample_set)

  def output_folder(self) -> str:
    return ''

  def input_folder(self) -> str:
    return ''
