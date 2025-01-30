import os.path

from da4vid.pipeline.steps import PipelineStep


class ProgressSaver:
  """
  Class used to save progresses in pipeline steps. This class provides methods
  intended to be used as post- and failure callbacks for pipeline steps, in order
  to save the progression in a file. Each line of the file will store the name of
  a completed step. The name is fully-qualified, meaning it is the concatenation
  of all names of the particular step and its parents, dot-separated.
  The file can be used to resume a pipeline which has been interrupted for any reason.
  """

  def __init__(self, progression_file: str):
    # Checking that the progression file is not a directory
    if os.path.isdir(progression_file):
      raise FileExistsError(f'Progression file is a directory: {progression_file}')
    self.progression_file = progression_file
    # Creating the progression file if not exists
    if not os.path.isfile(progression_file):
      f = open(progression_file, 'w')
      f.close()

  def save_completed_step(self, step: PipelineStep) -> None:
    """
    Saves the step completion in the file specified when this object has
    been created.
    :param step: The step which has been completed
    """
    with open(self.progression_file, 'a') as f:
      f.write(f'{step.full_name()}\n')
      f.flush()
