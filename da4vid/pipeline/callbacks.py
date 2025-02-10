import logging
import os.path
import time
from datetime import datetime

from da4vid.pipeline.steps import PipelineStep, CompositeStep


class ProgressManager:
  """
  Class used to save progresses in pipeline steps. This class provides methods
  intended to be used as post- and failure callbacks for pipeline steps, in order
  to save the progression in a file. Each line of the file will store the name of
  a completed step. The name is fully-qualified, meaning it is the concatenation
  of all names of the particular step and its parents, dot-separated.
  The file can be used to resume a pipeline which has been interrupted for any reason.
  """

  def __init__(self, progression_file: str):
    self.progress = []
    # Checking that the progression file is not a directory
    if os.path.isdir(progression_file):
      raise FileExistsError(f'Progression file is a directory: {progression_file}')
    self.progression_file = progression_file
    # Creating the progression file if not exists
    if not os.path.isfile(progression_file):
      f = open(progression_file, 'w')
      f.close()
    else:
      # Progress file already existing: loading progress
      with open(progression_file) as f:
        for line in f:
          self.progress.append(line.strip())

  def register(self, step: PipelineStep) -> None:
    """
    Register callbacks in this object to the given step. If the step is
    composite, then callbacks will be registered to all of its sub-steps.
    :param step: A pipeline step
    """
    step.register_post_step_fn(self.save_completed_step)
    if isinstance(step, CompositeStep):
      for sub_step in step.steps:
        self.register(sub_step)

  def save_completed_step(self, step: PipelineStep, **kwargs) -> None:
    """
    Saves the step completion in the file specified when this object has
    been created.
    :param step: The step which has been completed
    :param kwargs: Ignored
    """
    with open(self.progression_file, 'a') as f:
      step_name = step.full_name()
      self.progress.append(step_name)
      f.write(f'{step_name}\n')
      f.flush()

  def has_been_completed(self, step: PipelineStep) -> bool:
    return step.full_name() in self.progress


class ElapsedTimeSaver:
  """
  Class abstracting callbacks to save the time elapsed between steps, in CSV format.
  """

  COLUMNS = ['step', 'elapsed (s)']

  def __init__(self, time_file: str, force_rewrite: bool = False, delimiter: str = ';'):
    if os.path.isdir(time_file):
      raise FileExistsError(f'Time file is a directory: {time_file}')
    if os.path.exists(time_file) and not force_rewrite:
      raise FileExistsError(f'Time file already exists and force_rewrite has not been specified')
    self.time_file = time_file
    self.delimiter = delimiter
    self.__stack = []  # Stack for saving names and starting time in case of composite steps
    with open(time_file, 'w') as f:
      f.write(delimiter.join(self.COLUMNS) + "\n")
      f.flush()

  def register(self, step: PipelineStep) -> None:
    """
    Register callbacks in this object to the given step. If the step is
    composite, then callbacks will be registered to all of its sub-steps.
    :param step: A pipeline step
    """
    step.register_pre_step_fn(self.on_step_start)
    step.register_post_step_fn(self.on_step_end)
    if isinstance(step, CompositeStep):
      for sub_step in step.steps:
        self.register(sub_step)

  def on_step_start(self, step: PipelineStep, **kwargs) -> None:
    self.__stack.append((step.full_name(), datetime.fromtimestamp(time.time())))

  def on_step_end(self, step: PipelineStep, **kwargs) -> None:
    last_step, start = self.__stack.pop()
    if last_step != step.full_name():
      logging.warning(f'Invalid step: expecting {last_step}, seen {step.full_name()}')
    else:
      end = datetime.fromtimestamp(time.time())
      elapsed = end - start
      with open(self.time_file, 'a') as f:
        f.write(self.delimiter.join([step.full_name(), str(elapsed.total_seconds())]) + "\n")
