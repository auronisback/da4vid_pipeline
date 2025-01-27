"""
Script used to urn the pipeline.

:author Francesco Altiero <francesco.altiero@unina.it>
"""
import os.path
import sys

import click

from da4vid.pipeline.config import PipelineCreator, PipelinePrinter


@click.group()
def cli():
  pass


@click.command(name='execute', short_help='Executes a DA4VID pipeline.')
@click.argument('configuration', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--json', '-j', help='Specifies that configuration files is in JSON format',
              type=bool, default=False)
def execute(configuration, json: bool) -> None:
  """
  Executes the DA4VID pipeline specified by a configuration file, in yml or json format.
  \f

  :param configuration: Path to configuration file
  :param json: Flag indicating if the configuration is in JSON format
  """
  click.echo('executing')
  click.echo(configuration)
  click.echo(json)


@click.command(name='resume', short_help='Resumes a previously ran pipeline.')
@click.argument('configuration', type=click.Path(exists=True, dir_okay=False, readable=True))
def resume(configuration: str, json: bool) -> None:
  """
  Resumes a previously ran pipeline.
  \f

  :param configuration: Path to configuration file
  :param json: Flag indicating if the configuration is in JSON format
  """
  click.echo('resuming')


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


def main():
  if len(sys.argv) != 2:
    print(f'Usage: python {sys.argv[0]} <pipeline configuration YAML file>', file=sys.stderr)
    exit(1)
  cfg_file = sys.argv[1]
  if not os.path.isfile(cfg_file):
    print(f'Configuration file not exists or it is not a regular file: {cfg_file}', file=sys.stderr)
    exit(2)
  pipeline = PipelineCreator().from_yml(cfg_file)
  PipelinePrinter().print(pipeline)
  # pipeline.execute()


if __name__ == '__main__':
  cli.add_command(execute)
  cli.add_command(resume)
  cli.add_command(show)
  cli()
