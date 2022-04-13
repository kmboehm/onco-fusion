#!/bin/bash

mkdir -p .singularity/cache
mkdir -p .singularity/tmp
mkdir -p .singularity/lcache

export SINGULARITY_CACHEDIR=$PWD/.singularity/cache
export SINGULARITY_TMPDIR=$PWD/.singularity/tmp
export SINGULARITY_LOCALCACHEDIR=$PWD/.singularity/lcache

echo using SINGULARITY_CACHEDIR = $SINGULARITY_CACHEDIR
echo using SINGULARITY_TMPDIR = $SINGULARITY_TMPDIR
echo using SINGULARITY_LOCALCACHEDIR= $SINGULARITY_LOCALCACHEDIR
