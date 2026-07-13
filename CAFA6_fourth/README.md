# CAFA5 Protein Function Prediction - 2nd Place Solution Reproduction

> **Code Source Note**：The code in this repository is primarily derived from the second-place solution of the CAFA5 Protein Function Prediction Competition. Original repository address:
> [https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place](https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place)

---

Here are the instructions to reproduce the CAFA5 2nd solution using given code

# CONTENTS

* `nn_solution`                 : scripts for training Neural Network base models
* `protlib`                     : utils and code to train Py-Boost and LogReg models, data preprocessing and efficient metric computation
* `protnn`                      : utils and code to train GCN stacker model
* `CAFA5PIpeline.ipynb`         : CAFA5PIpeline.ipynb - notebook contains all the scripts calls and detailed explanation of each step. Also, contains directory structure (shoul be considered as both `directory_structure.txt` and `entry_points.md`)
* `config.yaml`                 : config used to execute training and inference. 
* `create-pytorch-env.sh`       : install all the requirements to run all deep learning parts
* `create-rapids-env.sh`        : install all the requirements to run processing and ML steps


# HARDWARE 

We used the following setup to train:

* 24 CPUs
* 512 GB RAM
* 2 x Tesla V100 32 GB

Minimal required hardware:
    
* 8 CPUs
* 64 GB RAM
* 1 x Tesla V100 32 GB    
* 300 GB disk space
    
# SOFTWARE

* Ubuntu 18.04
* Nvidia driver version 450 
* `python>=3.8` to run `CAFA5PIpeline.ipynb` and `Download.ipynb` notebooks. This `python` will not be used to train the models, it only launches the execution notebooks. Only requred libraries are `pyyaml` to read `config.yaml` and `kaggle` to obtain the original dataset via API
* `conda>=23.5.2`. We need one of the latest version to use Mamba solver. Otherwise, setup the environments will take hours

Other required tools will be installed via `create-pytorch-env.sh` and `create-rapids-env.sh` scripts.

* `pytorch-env` is the environment to train DL models. It will install pytorch, cupy, and some extra bio libraries
* `rapids-env` is the enviromnent to do preprocessing and train ML models. It uses NVIDIA RAPIDS toolkit (cudf) and cupy libraries to make the efficient dataprocessing, metric computation (including custom CUDA kernels for graph manipulation) and custom ML algorithms implementations.



# DOCKER REPRODUCTION

This section provides a Docker-based one-click reproduction solution, eliminating the need to manually configure dual conda environments (`pytorch-env` + `rapids-env`) and ensuring consistent operation across all hardware platforms.

```bash
docker load -i cafa6_4th_image.tar

docker run -it --gpus all \
  -v /home/xxx/helpers:/app/helpers \
  -v /home/xxx/models:/app/models \
  -v /home/xxx/pytorch-env:/app/pytorch-env \
  -v /home/xxx/rapids-env:/app/rapids-env \
  -v /home/xxx/CAFA5A/sub:/app/sub \
  -v /home/xxx/temporal:/app/temporal \
  cafa6_4th bash

source /opt/conda/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc

python /app/run_pipeline.py
```

After the pipeline runs to completion, a result file ready for Kaggle submission will be generated and stored in ./sub/.