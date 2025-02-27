# **HunyuanVideo Keyframe Control Lora**

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper exploring HunyuanVideo.

## 🔥🔥🔥 News!!

- 27 February 2025: We release the training code of HunyuanVideo Keyframe Control Lora and [Blog]().
- 24 February 2025: We release the inference code and model weights of HunyuanVideo Keyframe Control Lora . [Download](https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora/tree/main).

## Abstract

HunyuanVideo Keyframe Control Lora is an adapter for HunyuanVideo T2V model for keyframe-based video generation. ​Our architecture builds upon existing models, introducing key enhancements to optimize keyframe-based video generation:​

- We modify the input patch embedding projection layer to effectively incorporate keyframe information. By adjusting the convolutional input parameters, we enable the model to process image inputs within the Diffusion Transformer (DiT) framework.​
- We apply Low-Rank Adaptation (LoRA) across all linear layers and the convolutional input layer. This approach facilitates efficient fine-tuning by introducing low-rank matrices that approximate the weight updates, thereby preserving the base model's foundational capabilities while reducing the number of trainable parameters.
- The model is conditioned on user-defined keyframes, allowing precise control over the generated video's start and end frames. This conditioning ensures that the generated content aligns seamlessly with the specified keyframes, enhancing the coherence and narrative flow of the video.​

## 🎥 Demo

| Image 1                                                                                           | Image 2                                                                                           | Generated Video                                                                                                               |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| ![Image 1](https://content.dashtoon.ai/stability-images/41aeca63-064a-4003-8c8b-bfe2cc80d275.png) | ![Image 2](https://content.dashtoon.ai/stability-images/28956177-3455-4b56-bb6c-73eacef323ca.png) | <video controls autoplay src="https://content.dashtoon.ai/stability-images/14b7dd1a-1f46-4c4c-b4ec-9d0f948712af.mp4"></video> |
| ![Image 1](https://content.dashtoon.ai/stability-images/ddabbf2f-4218-497b-8239-b7b882d93000.png) | ![Image 2](https://content.dashtoon.ai/stability-images/b603acba-40a4-44ba-aa26-ed79403df580.png) | <video controls autoplay src="https://content.dashtoon.ai/stability-images/b00ba193-b3b7-41a1-9bc1-9fdaceba6efa.mp4"></video> |
| ![Image 1](https://content.dashtoon.ai/stability-images/5298cf0c-0955-4568-935a-2fb66045f21d.png) | ![Image 2](https://content.dashtoon.ai/stability-images/722a4ea7-7092-4323-8e83-3f627e8fd7f8.png) | <video controls autoplay src="https://content.dashtoon.ai/stability-images/0cb84780-4fdf-4ecc-ab48-12e7e1055a39.mp4"></video> |
| ![Image 1](https://content.dashtoon.ai/stability-images/69d9a49f-95c0-4e85-bd49-14a039373c8b.png) | ![Image 2](https://content.dashtoon.ai/stability-images/0cef7fa9-e15a-48ec-9bd3-c61921181802.png) | <video controls autoplay src="https://content.dashtoon.ai/stability-images/ce12156f-0ac2-4d16-b489-37e85c61b5b2.mp4"></video> |

## 📜 Requirements

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

## Acknowledgements

- We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.

- We build on top of a body of great open-source libraries: transformers, accelerate, peft, diffusers, bitsandbytes, torchao, deepspeed, mosaicml-streaming -- to name a few.
