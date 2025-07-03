#!/bin/bash

DATA_PREPARE_SCRIPT=/masif/data/masif_site/data_prepare_one.sh

if [ $# -ne 2 ]; then
    echo "Usage: $0 <list with pdb files and name_chains> <folder_with inputs>"
    exit 1
fi

list_input=$1
input_folder=$2

if [ ! -f "${list_input}" ]; then
    echo "List file ${list_input} does not exists or is not a regular file."
    exit 2
fi

if [ ! -e "${input_folder}" ]; then
    echo "Input folder ${input_folder} does not exists."
    exit 3
fi

i=1
while read -r p; do
    input_file=$input_folder/$(echo $p | cut -d' ' -f1)
    ppi_id=$(echo $p | cut -d' ' -f2)
    echo "$i - Preparing ${p}"
    ${DATA_PREPARE_SCRIPT} --file "${input_file}" "${ppi_id}"
    i=$((i+1))
done < "${list_input}"