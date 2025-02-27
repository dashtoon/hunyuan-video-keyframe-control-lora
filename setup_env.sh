#!/user/bin/env bash
conda create -n hunyuan_control_env python=3.10 -y && conda activate hunyuan_control_env
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit cuda -y

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install ninja
python -m pip install --verbose --upgrade git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

wget "https://raw.githubusercontent.com/kohya-ss/musubi-tuner/refs/heads/main/requirements.txt" -O requirements.txt
python -m pip install -r requirements.txt && rm requirements.txt

python -m pip install accelerate==1.2.1 transformers==4.46.3 bitsandbytes==0.45.2 decord==0.6.0 deepspeed==0.16.3 opencv-python==4.10.0.84 pandas==2.2.3 peft==0.14.0 mosaicml-streaming==0.11.0 pyrallis==0.3.1 torch-optimi==0.2.1
python -m pip install huggingface-hub hf_transfer
python -m pip install --upgrade git+https://github.com/huggingface/diffusers@81440fd47493b9f9e817411ca0499d0bf06fde95
python -m pip install icecream pre-commit
