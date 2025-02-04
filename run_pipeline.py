"""
Script used to run the pipeline.

:author Francesco Altiero <francesco.altiero@unina.it>
"""
import os.path
import click

from da4vid.pipeline.callbacks import ProgressSaver
from da4vid.pipeline.config import PipelineCreator, PipelinePrinter


@click.group()
def cli():
  pass


@click.command(name='execute', short_help='Executes a DA4VID pipeline.')
@click.argument('configuration', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--json', '-j', help='Specifies that configuration files is in JSON format',
              type=bool, default=False)
@click.option('--show-pipeline', '-s', type=bool, help='Shows pipeline configuration before resuming.',
              default=True)
@click.option('--save-progress', '-p', type=click.Path(exists=False, dir_okay=False, writable=True),
              help='Specify the file in which pipeline progress will be saved. Defaults to configuration name')
def execute(configuration, json: bool, show_pipeline: bool, save_progress: str) -> None:
  """
  Executes the DA4VID pipeline specified by a configuration file, in yml or json format.
  \f

  :param configuration: Path to configuration file
  :param json: Flag indicating if the configuration is in JSON format
  :param show_pipeline: Flag indicating whether show the pipeline before resuming
  :param save_progress: File in which pipeline progress will be saved, in case of future resuming.
                        If not given, a progress file with the same name of the configuration in
                        the same folder will be used
  """
  if json:
    raise NotImplementedError("JSON configurations have not been implemented yet! :'(")
  pipeline = PipelineCreator().from_yml(configuration)
  if show_pipeline:
    PipelinePrinter().print(pipeline)
  if save_progress is None:
    save_progress = os.path.join(os.path.dirname(configuration), '.'.join(configuration.split('.')[:-1] + '.progress'))
  click.echo(f'Saving progress to {save_progress}')
  progress_saver = ProgressSaver(save_progress)
  progress_saver.register(pipeline)
  pipeline.execute()


@click.command(name='resume', short_help='Resumes a previously ran pipeline.')
@click.argument('configuration', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--json', '-j', help='Specifies that configuration files is in JSON format',
              type=bool, default=False)
@click.option('--show-pipeline', '-s', type=bool, help='Shows pipeline configuration before resuming.', default=True)
def resume(configuration: str, json: bool, show_pipeline: bool) -> None:
  """
  Resumes a previously ran pipeline.
  \f

  :param configuration: Path to configuration file
  :param json: Flag indicating if the configuration is in JSON format
  :param show_pipeline: Flag indicating whether show the pipeline before resuming
  """
  if json:
    raise NotImplementedError("JSON configurations have not been implemented yet! :'(")
  pipeline = PipelineCreator().from_yml(configuration)
  if show_pipeline:
    PipelinePrinter().print(pipeline)
  print('resuming')


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


if __name__ == '__main__':
  cli.add_command(execute)
  cli.add_command(resume)
  cli.add_command(show)
  cli()
