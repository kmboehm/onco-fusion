#!/bin/bash
source /gpfs/mskmind_ess/limr/mambaforge/etc/profile.d/conda.sh
conda activate transformer
for entry in "qupath/data/slides"/*.svs; do
        temp=`basename ${entry}`
        temp="${temp/.svs/.tsv}"
        temp="qupath/data/results/${temp}"
        echo ${temp}
        if test -f ${temp}; then
            echo "${temp} exists"
        else
            echo sh _run_gpu_qupath_stardist_singleSlide.sh `basename $entry`
            sh _run_gpu_qupath_stardist_singleSlide.sh `basename $entry`
        fi
done
