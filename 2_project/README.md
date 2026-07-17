<!-- ## Requirments :
#### !pip install torch torchvision torchaudio
#### !pip install diffusers transformers accelerate peft
#### !pip install   pillow numpy opencv-python datasets
#### !pip install git+https://github.com/huggingface/diffusers
#### !pip install streamlit
#### !pip install gradio
#### !pip install open_clip_torch
#### !pip install lpips
#### !pip install --upgrade certifi



## DataSet:
#### Wild-Heart/Disney-VideoGeneration-Dataset
#### pick from hugging face git repo.


## Load the data and clean it:

#### first loead the video from directory frames by frames.
#### check whether image is corrupted or not.
#### for every frame check the success factor.
#### scale each and every frame between 0 and 1.
#### and permute the order of input: (frames,height,width,channels) -> (frames,channels,height,width)
#### resize the image to 256 x 256.
#### make the image centric oriented.
#### return the video in form of  "pixel_values": frames,
####            "caption": self.captions[idx]
#### because we feed the data to network in the form of videos and captions.



## Pipeline 
#### StableDiffusionPipeline (powerful image-to-video generation model that can generate 2-4 second high resolution  videos conditioned on an input image.)

### Workflow of StableDiffusionPipeline:
#### Input Image Encoding: The input image is converted into a latent representation using a Variational Autoencoder (VAE) encoder.
#### Noise Augmentation: The encoded image is slightly augmented with noise (noise_aug_strength) to introduce stochasticity, allowing the model to generate variation.
#### Denoising (UNet): The 3D UNet (now with temporal layers) predicts the noise in the video frames, iteratively denoising the latent representation over a series of steps.
#### Temporal Consistency: The model ensures that objects do not change shape or color unexpectedly between frames.
#### Decoding: The denoised latents are passed through the VAE decoder to create the final video frames.


#### Compoents of pipeline:  VAE,UNet,CLIP image encoder , Scheduler,Image processor

####  Configure Pipeline scheduler with DDPMScheduler
#### DDPM(Denoising Diffusion Probabilistic Models)  refers to the discrete denoising scheduler from the paper as well as pipeline.iteratively removes noise from a Gaussian noise tensor over a set number of steps, guided by a UNet model, to reconstruct a clean image from random noise.


## Lora configuration for Unet:
#### rank=4
#### lora_alpha=8.
#### initial_lora_weights: guassian weights


### Apply lora to unet.
#### unet is a convolutional neural netowrk(CNN) architecture designed for fast,precise image segmentation.
#### it utilizes U-shaped structure consisting of a contracting path to capture context and a symmetric exapnding path for  precise localization connect by skip connections that transfer high-resolution features.

#### takes existing UNet.
#### injects Lora layers into specified modules.
#### freezes original weights
#### makes only lora weights trainable.


#### data loader is used to batch and shuffle dataset samples.
#### Auto tokenizer converts text captions into token IDS
#### accelerate simplifies mixed precision and device handling.


## Optimizer
#### AdamW optimizer is used for transformer based models.
#### lr = learning rate.
#### use unet parameters

## VAE
#### vae -> variational encoder are probabilistic generative deep learning models that encode input data into a continous , structured latent space to reconstruct , denoise, or generate new similiar data.


#### Get Image Embeddings (SVD requires this instead of text)
#### Resize for CLIP Vision model (224x224)
#### CLIP image encoder expects 224×224 images.
#### diffusion models operate in latent space instead of pixel space.


### Encode the full video frames → video latents (what the model must denoise)
### Encode the first frame → conditioning latents (what guides the video)

#### SVD UNet takes 8 channels: 4 (noisy video latents) + 4 (clean first-frame latents)


# Forward Pass -> calculate loss -> backpropogation -> update the weights

# Save the unet.


#     User Prompt
#       ↓
#   Stable Diffusion (t2i_pipe)
#       ↓
#   Initial Image
#       ↓
#    Stable Video Diffusion (pipe)
#       ↓
#   Generated Video



 -->


# Video Generation using Stable Video Diffusion with LoRA Fine-Tuning

## 1. Environment Setup
#### Expected Python version == 3.11
#### Install the required dependencies before running the project.

```bash

Create Virtual Environment:

python3 -m venv env_name
source venv/bin/activate
pip install --upgrade pip

and then run the following command to install dependencies


pip install -r requirements.txt

or 

pip install torch torchvision torchaudio
pip install diffusers transformers accelerate peft
pip install pillow numpy opencv-python datasets
pip install git+https://github.com/huggingface/diffusers
pip install streamlit
pip install gradio
pip install open_clip_torch
pip install lpips
pip install --upgrade certifi

```

---

# 2. Dataset

Dataset used for training:

**Disney Video Generation Dataset**

Source:  
Hugging Face Repository  
`Wild-Heart/Disney-VideoGeneration-Dataset`

URL: `https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset/tree/316b85f5d7c263260cc526a25eece96cc30e0c06`

This dataset contains short Disney-style videos paired with captions describing the visual content of the video.

---

# 3. Data Loading and Preprocessing

The dataset is processed to convert raw videos into a format suitable for training the diffusion model.

## Steps

### 1. Frame Extraction
- Load each video from the dataset directory.
- Extract frames sequentially.

### 2. Corruption Check
- Verify that each frame is readable and not corrupted.

### 3. Frame Validation
- Ensure frames meet the required processing conditions.

### 4. Normalization
- Scale pixel values of each frame between **0 and 1**.

### 5. Dimension Reordering
Reorder frame dimensions from:

```
(frames, height, width, channels)
```

to

```
(frames, channels, height, width)
```

### 6. Image Resizing
Resize every frame to:

```
256 × 256
```

### 7. Center Cropping
Apply center cropping to ensure the subject remains properly aligned.

### 8. Final Output Format

The processed data is returned in the following format:

```python
{
    "pixel_values": frames,
    "caption": self.captions[idx]
}
```

This format allows the model to receive both the **video frames** and their **corresponding captions**.

---

# 4. Stable Diffusion Pipeline

The project utilizes a **Stable Diffusion Pipeline** designed for **image-to-video generation**.

The pipeline generates **2–4 second high-resolution videos** conditioned on an input image.

---

# 5. Workflow of Stable Diffusion Pipeline

### 1. Input Image Encoding
The input image is converted into a **latent representation** using a **Variational Autoencoder (VAE) encoder**.

### 2. Noise Augmentation
Noise is added to the encoded image using a **noise augmentation strength parameter**.  
This introduces stochasticity and allows the model to generate variations.

### 3. Denoising using UNet
A **3D UNet with temporal layers** predicts and removes noise iteratively from the latent representation.

### 4. Temporal Consistency
Temporal layers ensure that objects maintain **consistent shape, position, and color across frames**.

### 5. Decoding
The denoised latent representations are decoded using the **VAE decoder** to generate the final video frames.

---

# 6. Components of the Pipeline

The pipeline consists of the following key components:

- **VAE (Variational Autoencoder)**
- **UNet (Denoising Network)**
- **CLIP Image Encoder**
- **Scheduler**
- **Image Processor**

---

# 7. Diffusion Scheduler

The pipeline uses the **DDPMScheduler**.

### DDPM (Denoising Diffusion Probabilistic Models)

DDPM iteratively removes noise from a Gaussian noise tensor over a fixed number of steps.  
The process is guided by a **UNet model** to reconstruct clean images from random noise.

---

# 8. LoRA Configuration for UNet

Low-Rank Adaptation (LoRA) is applied to fine-tune the UNet efficiently.

### Configuration

- **Rank:** 4
- **LoRA Alpha:** 8
- **Initialization:** Gaussian weights

---

# 9. Applying LoRA to UNet

The UNet architecture is a **Convolutional Neural Network (CNN)** designed for precise image processing.

### UNet Architecture

- **Contracting Path:** Captures contextual information.
- **Expanding Path:** Enables precise localization.
- **Skip Connections:** Preserve high-resolution spatial information.

### LoRA Integration

LoRA fine-tuning works as follows:

- Takes an existing **pretrained UNet**
- Injects **LoRA layers** into specific modules
- **Freezes original model weights**
- **Only LoRA parameters are trainable**

This allows efficient fine-tuning with significantly fewer parameters.

---

# 10. Data Loading

A **DataLoader** is used to:

- Batch dataset samples
- Shuffle the dataset during training
- Efficiently feed data to the model

---

# 11. Tokenization

An **AutoTokenizer** converts text captions into **token IDs** that can be processed by the model.

---

# 12. Accelerate Library

The **Accelerate** library simplifies:

- Device placement (CPU / GPU)
- Mixed precision training
- Distributed training

---

# 13. Optimizer

The **AdamW optimizer** is used for training.

### Configuration

- **Optimizer:** AdamW
- **Learning Rate:** Defined during training
- **Parameters:** UNet LoRA parameters

AdamW is well suited for **transformer and diffusion models**.

---

# 14. Variational Autoencoder (VAE)

A **Variational Autoencoder (VAE)** encodes images into a structured **latent space**.

### Purpose

- Compress images into latent representations
- Enable efficient diffusion in latent space
- Reconstruct images from latent vectors

Diffusion models operate in **latent space rather than pixel space**, making training more efficient.

---

# 15. CLIP Image Encoder

The **CLIP Vision Encoder** is used to obtain image embeddings.

### Requirements

CLIP expects images of size:

```
224 × 224
```

Frames are resized accordingly before being passed into the encoder.

---

# 16. Latent Representations

### Video Latents
All video frames are encoded into latent representations.

```
video frames → VAE encoder → video latents
```

These are the latents that the diffusion model learns to denoise.

### Conditioning Latents

The **first frame of the video** is encoded separately.

```
first frame → conditioning latent
```

This latent representation guides the video generation.

---

# 17. SVD UNet Input

The Stable Video Diffusion UNet receives **8 channels**:

```
4 channels → noisy video latents
4 channels → clean first-frame latents
```

---

# 18. Training Procedure

Training follows the standard deep learning workflow:

```
Forward Pass
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Weight Update
```

Only **LoRA parameters of the UNet** are updated during training.

---

# 19. Model Saving

After training, the **fine-tuned UNet with LoRA weights** is saved for inference.

---

# 20. Video Generation Workflow

The complete generation pipeline works as follows:

```
User Prompt
      ↓
Stable Diffusion (Text → Image)
      ↓
Initial Image
      ↓
Stable Video Diffusion
      ↓
Generated Video
```

The text prompt is first converted into an image, which is then used to generate a short video.

---

# 21. Output

The final output is a **2–4 second generated video** aligned with the user prompt.




##  You can run the cell in which we load the svd_video_lora parameters to generate a video.



