"""
Script used to run the pipeline.

:author Francesco Altiero <francesco.altiero@unina.it>
"""
import os.path
import click

from da4vid.model.samples import SampleSet
from da4vid.pipeline.callbacks import ProgressManager, ElapsedTimeSaver
from da4vid.pipeline.config import PipelineCreator, PipelinePrinter
from da4vid.pipeline.steps import PipelineStep, PipelineRootStep, CompositeStep


@click.group()
def cli():
  pass


@click.command(name='execute', short_help='Executes a DA4VID pipeline.')
@click.argument('configuration', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--json', '-j', help='Specifies that configuration files is in JSON format.',
              type=bool, default=False)
@click.option('--show-pipeline', '-s', type=bool, help='Shows pipeline configuration before resuming.',
              default=True, is_flag=True)
@click.option('--save-progress', '-p', type=click.Path(exists=False, dir_okay=False, writable=True),
              help='Specify the file in which pipeline progress will be saved. Defaults to configuration name.')
@click.option('--save-time', '-t', type=bool, default=False, is_flag=True,
              help='Specify if times for various pipeline steps should be saved, in CSV format.')
def execute(configuration, json: bool, show_pipeline: bool, save_progress: str, save_time: bool) -> None:
  """
  Executes the DA4VID pipeline specified by a configuration file, in yml or json format.
  \f

  :param configuration: Path to configuration file
  :param json: Flag indicating if the configuration is in JSON format
  :param show_pipeline: Flag indicating whether show the pipeline before resuming
  :param save_progress: File in which pipeline progress will be saved, in case of future resuming.
                        If not given, a progress file with the same name of the configuration in
                        the same folder will be used
  :param save_time: Flag indicating if pipeline step execution times should be recorded
  """
  if json:
    raise NotImplementedError("JSON configurations have not been implemented yet! :'(")
  pipeline = PipelineCreator().from_yml(configuration)
  if show_pipeline:
    PipelinePrinter().print(pipeline)
  if save_progress is None:
    save_progress = __progress_file_from_configuration_path(configuration)
  progress_saver = ProgressManager(save_progress)
  progress_saver.register(pipeline)
  click.echo(f'Saving progress to {save_progress}')
  if save_time:
    time_file = __elapsed_time_from_configuration_path(configuration)
    click.echo(f'Saving execution time to {time_file}')
    time_saver = ElapsedTimeSaver(time_file)
    time_saver.register(pipeline)
  pipeline.execute()


@click.command(name='resume', short_help='Resumes a previously ran pipeline.')
@click.argument('configuration', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--json', '-j', help='Specifies that configuration files is in JSON format',
              type=bool, default=False)
@click.option('--show-pipeline', '-s', type=bool, help='Shows pipeline configuration before resuming.', default=True)
@click.option('--progress-file', type=click.Path(exists=True, dir_okay=False, readable=True),
              help="Progress file for the pipeline")
def resume(configuration: str, json: bool, show_pipeline: bool, progress_file: str) -> None:
  """
  Resumes a previously ran pipeline given a progress file.
  \f

  :param configuration: Path to configuration file
  :param json: Flag indicating if the configuration is in JSON format
  :param show_pipeline: Flag indicating whether show the pipeline before resuming
  :param progress_file: The file with progress used to resume the pipeline. Defaults to
                        the configuration file (with a .progress extension)
  """
  if json:
    raise NotImplementedError("JSON configurations have not been implemented yet! :'(")
  pipeline = PipelineCreator().from_yml(configuration)
  if show_pipeline:
    PipelinePrinter().print(pipeline)
  if not progress_file:
    progress_file = __progress_file_from_configuration_path(configuration)
  progress_manager = ProgressManager(progress_file)
  __resume_step(pipeline, progress_manager)


def __resume_step(step: PipelineStep, progress_manager: ProgressManager, sample_set: SampleSet = None) -> SampleSet:
  if progress_manager.has_been_completed(step):
    return step.resume(sample_set)
  else:
    if isinstance(step, PipelineRootStep):
      sample_set = SampleSet()
      sample_set.add_samples(step.antigen)
    if isinstance(step, CompositeStep):
      for sub_step in step.steps:
        sample_set = __resume_step(sub_step, progress_manager, sample_set)
      return sample_set
    else:
      return step.execute(sample_set)


@click.command(name='show', short_help='Prints a pipeline configuration.')
@click.argument('configuration', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--json', '-j', help='Specifies that configuration files is in JSON format',
              type=bool, default=False)
def show(configuration, json: bool) -> None:
  """
  Prints the configuration file passed in input as a tree string.
  \f

  :param configuration: Path to configuration file
  :param json: Flag indicating if the configuration is in JSON format
  """
  if json:
    raise NotImplementedError("JSON configurations have not been implemented yet! :'(")
  pipeline = PipelineCreator().from_yml(configuration)
  PipelinePrinter().print(pipeline)


def __progress_file_from_configuration_path(configuration: str) -> str:
  return os.path.join(os.path.dirname(configuration),
                      '.'.join(configuration.split('.')[:-1]) + '.progress')


def __elapsed_time_from_configuration_path(configuration: str) -> str:
  return os.path.join(os.path.dirname(configuration),
                      '.'.join(configuration.split('.')[:-1]) + '.time.csv')


if __name__ == '__main__':
  cli.add_command(execute)
  cli.add_command(resume)
  cli.add_command(show)
  cli()
