"""
Script used to check if all pipeline configuration is as intended.

:author Francesco Altiero <francesco.altiero@unina.it>
"""
import os.path
import sys

from da4vid.pipeline.config import PipelineCreator, PipelinePrinter


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


if __name__ == '__main__':
  main()
