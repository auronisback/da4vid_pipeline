"""
Workaround to trick Colabfold in having already calculated
Multi-Sequence Alignments.

Author: Francesco Altiero <francesco.altiero@unina.it>
"""

import os
import shutil
import sys


def read_fasta(fasta_file: str):
  seqs = []
  with open(fasta_file) as f:
    line = f.readline()
    while line:
      name = line.strip()[1:]
      f.readline()  # Skipping a line
      seqs.append(name)
      line = f.readline()
  return seqs


def main():
  if len(sys.argv) < 4:
    print(f'Usage: python {sys.argv[0]} <fasta_file> <output_folder> <sample_num>')
    exit(1)
  fasta_file = sys.argv[1]
  sample_id = int(sys.argv[3])
  output_folder = sys.argv[2]
  base_name = '.'.join(os.path.basename(fasta_file).split('.')[:-1])
  orig_a3m = os.path.join(output_folder, f'{base_name}_{sample_id}.a3m')
  orig_env = os.path.join(output_folder, f'{base_name}_{sample_id}_env')
  seqs = read_fasta(fasta_file)
  for name in seqs:
    this_a3m = os.path.join(output_folder, f'{name}.a3m')
    this_env = os.path.join(output_folder, f'{name}_env')
    if this_a3m == orig_a3m:
      continue
    if not os.path.exists(this_a3m):
      shutil.copy2(orig_a3m, this_a3m)
    if not os.path.exists(this_env):
      shutil.copytree(orig_env, this_env)


if __name__ == '__main__':
  main()
