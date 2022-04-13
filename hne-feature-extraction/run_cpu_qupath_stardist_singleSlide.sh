singularity run -B /fscratch/docker/qupath/data:/data,/fscratch/docker/qupath/models:/models,/fscratch/docker/qupath/scripts:/scripts --nv /fscratch/docker/qupath/qupath-stardist_latest.sif java -Djava.awt.headless=true \
-Djava.library.path=/qupath-cpu/build/dist/QuPath-0.2.3/lib/app \
-jar /qupath-cpu/build/dist/QuPath-0.2.3/lib/app/qupath-0.2.3.jar \
script --image /data/slides/${1} /scripts/stardist_nuclei_and_lymphocytes.groovy
