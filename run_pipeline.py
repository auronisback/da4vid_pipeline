"""
Script used to urn the pipeline.

:author Francesco Altiero <francesco.altiero@unina.it>
"""
import click

from da4vid.pipeline.config import PipelineCreator, PipelinePrinter


@click.group()
def cli():
  pass


@click.command(name='execute', short_help='Executes a DA4VID pipeline.')
@click.argument('configuration', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--json', '-j', help='Specifies that configuration files is in JSON format',
              type=bool, default=False)
@click.option('--show-pipeline', '-s', type=bool, help='Shows pipeline configuration before resuming.', default=True)
def execute(configuration, json: bool, show_pipeline: bool) -> None:
  """
  Executes the DA4VID pipeline specified by a configuration file, in yml or json format.
  \f

  :param configuration: Path to configuration file
  :param json: Flag indicating if the configuration is in JSON format
  :param show_pipeline: Flag indicating whether show the pipeline before resuming
  """
  if json:
    raise NotImplementedError("JSON configurations have not been implemented yet! :'(")
  else:
    pipeline = PipelineCreator().from_yml(configuration)
  if show_pipeline:
    PipelinePrinter().print(pipeline)
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
  else:
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
  pipeline = PipelineCreator().from_yml(configuration)
  PipelinePrinter().print(pipeline)


if __name__ == '__main__':
  cli.add_command(execute)
  cli.add_command(resume)
  cli.add_command(show)
  cli()
