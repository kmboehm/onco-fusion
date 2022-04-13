singularity run --env TF_FORCE_GPU_ALLOW_GROWTH=true,PER_PROCESS_GPU_MEMORY_FRACTION=0.8 -B /fscratch/docker/qupath/data:/data,/fscratch/docker/qupath/models:/models,/fscratch/docker/qupath/scripts:/scripts --nv /fscratch/docker/qupath/qupath-stardist_latest.sif java -Djava.awt.headless=true \
-Djava.library.path=/qupath-gpu/build/dist/QuPath-0.2.3/lib/app \
-jar /qupath-gpu/build/dist/QuPath-0.2.3/lib/app/qupath-0.2.3.jar \
script --image /data/slides/${1} /scripts/stardist_nuclei_and_lymphocytes.groovy
