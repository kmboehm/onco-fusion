export MYUID := $(shell id -u)
export MYGID := $(shell id -g)
export SINGULARITY_CACHEDIR := $(PWD)/.singularity/cache
export SINGULARITY_TMPDIR := $(PWD)/.singularity/tmp
export SINGULARITY_LOCALCACHEDIR := $(PWD)/.singularity/lcache
IMAGE = docker://druvpatel/qupath-stardist:latest

.PHONY: all help build clean test run
.DEFAULT_GOAL := help

# help menu
define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


build:      ## build docker image
	docker build -t qupath/latest .


init:
	./init_singularity_env.sh


clean:      ## cleanup
	docker system prune 


clean-images:  ## remove qupath-stardist docker and singularity images. WARNING: Will affect other users!
	docker rmi -f $(IMAGE)
	singularity cache clean -f
	rm qupath-stardist_latest.sif


run-cpu:     ## run a script within the star_dist service. This run uses cpus.
	docker-compose -f docker-compose.yml run \
           star_dist \
              java -Djava.awt.headless=true -Djava.library.path=/qupath-cpu/build/dist/QuPath-0.2.3/lib/app \
 		-jar /qupath-cpu/build/dist/QuPath-0.2.3/lib/app/qupath-0.2.3.jar \
 		  script --image /$(image) /$(script)

run-gpu:     ## run a script within the star_dist service. This run uses gpus.
	docker-compose -f docker-compose.yml run \
           -e TF_FORCE_GPU_ALLOW_GROWTH=true -e PER_PROCESS_GPU_MEMORY_FRACTION=0.8 \
              star_dist \
                 java -Djava.awt.headless=true -Djava.library.path=/qupath-gpu/build/dist/QuPath-0.2.3/lib/app \
                    -jar /qupath-gpu/build/dist/QuPath-0.2.3/lib/app/qupath-0.2.3.jar \
                       script --image /$(image) /$(script)


build-singularity: init  ## pulls qupath-stardist image and converts it to a singularity image
	singularity pull --force  $(IMAGE)
	singularity cache list -v


run-singularity-gpu: ## runs the qupath-stardist image in gpu mode
	singularity run --env TF_FORCE_GPU_ALLOW_GROWTH=true,PER_PROCESS_GPU_MEMORY_FRACTION=0.8  -B $(PWD)/data:/data,$(PWD)/detections:/detections,$(PWD)/models:/models,$(PWD)/scripts:/scripts  --nv $(PWD)/qupath-stardist_latest.sif java -Djava.awt.headless=true \
	-Djava.library.path=/qupath-gpu/build/dist/QuPath-0.2.3/lib/app  \
	-jar /qupath-gpu/build/dist/QuPath-0.2.3/lib/app/qupath-0.2.3.jar  \
	script --image /$(image) /$(script)

run-singularity-cpu: ## runs the qupath-stardist image in cpu mode
	singularity run -B $(PWD)/data:/data,$(PWD)/detections:/detections,$(PWD)/models:/models,$(PWD)/scripts:/scripts  $(PWD)/qupath-stardist_latest.sif java -Djava.awt.headless=true \
	-Djava.library.path=/qupath-cpu/build/dist/QuPath-0.2.3/lib/app  \
	-jar /qupath-cpu/build/dist/QuPath-0.2.3/lib/app/qupath-0.2.3.jar  \
	script --image /$(image) /$(script)



