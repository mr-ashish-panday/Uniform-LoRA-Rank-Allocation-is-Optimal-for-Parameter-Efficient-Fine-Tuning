#!/usr/bin/env bash
mkdir -p reproducibility/{code,env,logs,outputs}
cp fine_tune.py reproducibility/code/
cp -r utils reproducibility/code/
conda env export > reproducibility/env/conda_env.yaml
cp -r logs reproducibility/logs
cp -r outputs reproducibility/outputs
tar czf reproducibility.tar.gz reproducibility
