# **HunyuanVideo Keyframe Control Lora**

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our experiments on HunyuanVideo Keyframe Control Lora.

## 🔥🔥🔥 News!!

- 27 February 2025: We release the training code of HunyuanVideo Keyframe Control Lora and [Blog]().
- 24 February 2025: We release the inference code and model weights of HunyuanVideo Keyframe Control Lora . [Download](https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora/tree/main).

## Abstract

HunyuanVideo Keyframe Control Lora is an adapter for HunyuanVideo T2V model for keyframe-based video generation. ​Our architecture builds upon existing models, introducing key enhancements to optimize keyframe-based video generation:​

- We modify the input patch embedding projection layer to effectively incorporate keyframe information. By adjusting the convolutional input parameters, we enable the model to process image inputs within the Diffusion Transformer (DiT) framework.​
- We apply Low-Rank Adaptation (LoRA) across all linear layers and the convolutional input layer. This approach facilitates efficient fine-tuning by introducing low-rank matrices that approximate the weight updates, thereby preserving the base model's foundational capabilities while reducing the number of trainable parameters.
- The model is conditioned on user-defined keyframes, allowing precise control over the generated video's start and end frames. This conditioning ensures that the generated content aligns seamlessly with the specified keyframes, enhancing the coherence and narrative flow of the video.​

## 🎥 Demo

**Click on the first column images to view the generated videos**

| Generated Video                                                                                                                                                                                                                             | Image 1                                                                                                                     | Image 2                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| <a href="https://content.dashtoon.ai/stability-images/14b7dd1a-1f46-4c4c-b4ec-9d0f948712af.mp4"><img src="https://content.dashtoon.ai/stability-images/41aeca63-064a-4003-8c8b-bfe2cc80d275.png" width="200" alt="Click to view video"></a> | <img src="https://content.dashtoon.ai/stability-images/41aeca63-064a-4003-8c8b-bfe2cc80d275.png" width="200" alt="Image 1"> | <img src="https://content.dashtoon.ai/stability-images/28956177-3455-4b56-bb6c-73eacef323ca.png" width="200" alt="Image 2"> |
| <a href="https://content.dashtoon.ai/stability-images/b00ba193-b3b7-41a1-9bc1-9fdaceba6efa.mp4"><img src="https://content.dashtoon.ai/stability-images/ddabbf2f-4218-497b-8239-b7b882d93000.png" width="200" alt="Click to view video"></a> | <img src="https://content.dashtoon.ai/stability-images/ddabbf2f-4218-497b-8239-b7b882d93000.png" width="200" alt="Image 1"> | <img src="https://content.dashtoon.ai/stability-images/b603acba-40a4-44ba-aa26-ed79403df580.png" width="200" alt="Image 2"> |
| <a href="https://content.dashtoon.ai/stability-images/0cb84780-4fdf-4ecc-ab48-12e7e1055a39.mp4"><img src="https://content.dashtoon.ai/stability-images/5298cf0c-0955-4568-935a-2fb66045f21d.png" width="200" alt="Click to view video"></a> | <img src="https://content.dashtoon.ai/stability-images/5298cf0c-0955-4568-935a-2fb66045f21d.png" width="200" alt="Image 1"> | <img src="https://content.dashtoon.ai/stability-images/722a4ea7-7092-4323-8e83-3f627e8fd7f8.png" width="200" alt="Image 2"> |
| <a href="https://content.dashtoon.ai/stability-images/ce12156f-0ac2-4d16-b489-37e85c61b5b2.mp4"><img src="https://content.dashtoon.ai/stability-images/69d9a49f-95c0-4e85-bd49-14a039373c8b.png" width="250" alt="Click to view video"></a> | <img src="https://content.dashtoon.ai/stability-images/69d9a49f-95c0-4e85-bd49-14a039373c8b.png" width="250" alt="Image 1"> | <img src="https://content.dashtoon.ai/stability-images/0cef7fa9-e15a-48ec-9bd3-c61921181802.png" width="250" alt="Image 2"> |

## 📜 Recommeded Settings

1. The model works best on human subjects. Single subject images work slightly better.
2. It is recommended to use the following image generation resolutions `720x1280`, `544x960`, `1280x720`, `960x544`.
3. It is recommended to set frames from 33 upto 97. Can go upto 121 frames as well (but not tested much).
4. Prompting helps a lot but works even without. The prompt can be as simple as just the name of the object you want to generate or can be detailed.
5. `num_inference_steps` is recommended to be 50, but for fast results you can use 30 as well. Anything less than 30 is not recommended.

## 🛠️ Dependencies and Installation

Begin by cloning the repository:

```shell
git clone https://github.com/dashtoon/hunyuan-video-keyframe-control-lora.git
cd hunyuan-video-keyframe-control-lora
```

### Installation Guide for Linux

We recommend CUDA versions 12.4

Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

```shell
bash setup_env.sh
```

## Inference

The model weights can be downloaded from [Huggingface](https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora)

You can run inference using the provided script. The script uses `flash_attn` but can also be modified to use `sage_attn`. Running the below command will output a video that is saved in `output.mp4`

- An NVIDIA GPU with CUDA support is required.
  - The model is tested on a single 80G GPU.
  - **Minimum**: The minimum GPU memory required is ~60GB for 720px1280px129f and ~45G for 544px960px129f.
  - **Recommended**: We recommend using a GPU with 80GB of memory for better generation quality.
- Tested operating system: Linux

```shell
export BASE_MODEL = "hunyuanvideo-community/HunyuanVideo"
export LORA_PATH = "<PATH TO DOWNLOADED CONTROL LORA>"
export IMAGE_1 = "<PATH TO THE FIRST FRAME>"
export IMAGE_2 = "<PATH TO THE LAST FRAME>"
export PROMPT = "<A BEAUTIFUL PROMPT>"
export HEIGHT = 960
export WIDHT = 544
export n_FRAMES = 33

python hv_control_lora_inference.py \
    --model $BASE_MODEL \
    --lora $LORA_PATH \
    --frame1 $IMAGE_1 --frame2 $IMAGE_2 --prompt $PROMPT --frames $n_FRAMES \
    --height $HEIGHT --width $WIDTH \
    --steps 50 \
    --guidance 6.0 \
    --seed 123143153 \
    --output output.mp4
```

## Training

### Dataset Preparation

It is recommended to have atleast 1 GPU with 80GB of VRAM. We use mosaic-ml streaming for caching our data. We expect our original data in the following format. Running the tree command, you should see:

```
dataset
├── metadata.csv
├── videos
    ├── 00000.mp4
    ├── 00001.mp4
    ├── ...
```

The csv can contain any number of columns, but due to limited support at the moment, we only make use of prompt and video columns. The CSV should look like this:

```
caption,video_file,other_column1,other_column2
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.,videos/00000.mp4,...,...
```

For the above format you would run the following command for starting to cache the dataset:

```shell
python scripts/hv_cache_dataset.py \
    --csv "dataset/metadata.csv" \
    --base_dir "dataset" \
    --video_column video_file \
    --caption_column "caption" \
    --output_dir "dataset/mds_cache" \
    --bucket_reso \
        "1280x720x33" "1280x720x65" "1280x720x97" "960x544x33" "960x544x65" "960x544x97" \
        "720x1280x33" "720x1280x65" "720x1280x97" "544x960x33" "544x960x65" "544x960x97" \
    --min_bucket_count 100 \
    --head_frame 0
```

- `bucket_reso` : this specifies the bucket resolutions to train on in the format of WxHxF.
- `head_frame`: the intial frame from where to start extracting from a video

**NOTE:** It is recommened to first convert your video into separate scenes and ensure there is continuity between scenes. [This](https://github.com/aigc-apps/EasyAnimate/tree/main/easyanimate/video_caption) is a good starting point for video dataset preparation.

The next commanded will start caching the LLM embeds and the VAE states.

```shell
NUM_GPUS=8
MIXED_PRECISION="bf16"
accelerate launch --num_processes=$NUM_GPUS --mixed_precision=$MIXED_PRECISION --main_process_port=12345 \
    scripts/hv_precompute_latents_dist.py \
        --pretrained_model_name_or_path="hunyuanvideo-community/HunyuanVideo" \
        --mds_data_path "dataset/mds_cache" \
        --output_dir "dataset/mds_cache_latents" \
        --recursive
```

Now you need to add the path to all the mds latent folders in `./configs/config_defaults.yaml` config file under `data.local` as a list. The latent_cache should be stored unfer `--output_dir` folder as `1280x720x33_00` folders. Where `1280` is the width of the video, `720` is the height of the video and `33` is the framerate of the video and `00` is the gpu id.
Now we are ready to start training!

### Starting a traning run

```shell
NUM_GPUS=8
MIXED_PRECISION="bf16"
EXPERIMENT_NAME="my_first_run"
OUTPUT_DIR="outputs/"
CONFIG_PATH="./configs/config_defaults.yaml"
NUM_EPOCHS=1

accelerate launch --num_processes=$NUM_GPUS --mixed_precision=$MIXED_PRECISION --main_process_port=12345 \
    hv_train_control_lora.py \
        --config_path $CONFIG_PATH \
        --experiment.run_id=$EXPERIMENT_NAME \
        --experiment.output_dirpath=$OUTPUT_DIR \
        --network.train_norm_layers=False \
        --network.lora_dropout=0.05 \
        --hparams.ema.use_ema=False \
        --hparams.num_train_epochs=1
```

## Acknowledgements

- We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.

- We build on top of a body of great open-source libraries: transformers, accelerate, peft, diffusers, bitsandbytes, torchao, deepspeed, mosaicml-streaming -- to name a few.
