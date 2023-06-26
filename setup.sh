#!/bin/bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Script to be sourced on launch of the Gradient Notebook
EXIT_CODE=0
echo "Graphcore setup - Starting notebook setup"
DETECTED_NUMBER_OF_IPUS=$(python .gradient/available_ipus.py)
if [[ "$1" == "test" ]]; then
    IPU_ARG="${DETECTED_NUMBER_OF_IPUS}"
else
    IPU_ARG=${1:-"${DETECTED_NUMBER_OF_IPUS}"}
fi
echo "Graphcore setup - Detected ${DETECTED_NUMBER_OF_IPUS} IPUs"
if [[ "${DETECTED_NUMBER_OF_IPUS}" == "0" ]]; then
    echo "=============================================================================="
    echo "                         ERROR  DETECTED"
    echo "=============================================================================="
    echo "Connection to IPUs timed-out. This error indicates a problem with the "
    echo "hardware you are running on. Please contact Paperspace Support at "
    echo " https://docs.paperspace.com/contact-support/ "
    echo " referencing the Notebook ID: ${PAPERSPACE_METRIC_WORKLOAD_ID:-unknown}"
    echo "=============================================================================="
    exit 255
fi
# Check the state of the partition
GC_INFO_OUTPUT=$(timeout 5 gc-info -l 2>&1)
if [[ "$(echo ${GC_INFO_OUTPUT} | grep 'Partition.* \[active\]')" ]]
then
    echo "Graphcore setup - Partition check - passed"
elif [[ "$(echo ${GC_INFO_OUTPUT} | grep 'partition is not ACTIVE')" ]]
then
    echo "=============================================================================="
    echo "                         IPU ERROR  DETECTED"
    echo "=============================================================================="
    echo " IPU Partition is not active. This error indicates a problem with the "
    echo "hardware you are running on. Please contact Paperspace Support at "
    echo " https://docs.paperspace.com/contact-support/ "
    echo " referencing the Notebook ID: ${PAPERSPACE_METRIC_WORKLOAD_ID:-unknown}"
    echo "=============================================================================="
    gc-info -l
    exit 254
else
    echo "[WARNING] IPU Partition in an unrecognised state - Notebook will start normally but"
    echo "[WARNING] you may encounter hardware related errors. Get in touch with Paperspace and/or"
    echo "[WARNING] Graphcore support if you encounter unexpected behaviours or errors."
    echo "[WARNING] Output was: ${GC_INFO_OUTPUT}"
    EXIT_CODE=253
fi

export NUM_AVAILABLE_IPU=${IPU_ARG}
export GRAPHCORE_POD_TYPE="pod${IPU_ARG}"
export POPLAR_EXECUTABLE_CACHE_DIR="/tmp/exe_cache"
export DATASETS_DIR="/tmp/dataset_cache"
export CHECKPOINT_DIR="/tmp/checkpoints"


# mounted public dataset directory (path in the container)
# in the Paperspace environment this would be ="/datasets"
export PUBLIC_DATASETS_DIR="/datasets"

export TF_POPLAR_FLAGS='--executable_cache_path='${POPLAR_EXECUTABLE_CACHE_DIR}''

# Environment variables customised for OGB notebooks
export OGB_DATASETS_DIR="${DATASETS_DIR}/ogb_lsc_pcqm4mv2/datasets"
export OGB_CHECKPOINT_DIR="${CHECKPOINT_DIR}/ogb_lsc_pcqm4mv2/checkpoints"
export OGB_SUBMISSION_CODE="./pcqm4mv2_submission"

# Logger specific vars
export TIER_TYPE=$(python .gradient/check_tier.py)
export FIREHOSE_STREAM_NAME="paperspacenotebook_production"
export GCLOGGER_CONFIG="${PUBLIC_DATASETS_DIR}/gcl"
export REPO_FRAMEWORK="Tensorflow 2"

echo "Graphcore setup - Spawning dataset preparation process"
nohup /notebooks/.gradient/prepare-datasets.sh ${@} & tail -f nohup.out &

echo "Graphcore setup - Starting Jupyter kernel"
export PIP_DISABLE_PIP_VERSION_CHECK=1 CACHE_DIR=/tmp
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True \
            --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True \
            --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True

exit $EXIT_CODE