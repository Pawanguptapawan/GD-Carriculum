# Diffusion Models — Study Notes

Based on the DDPM (Denoising Diffusion Probabilistic Model) implementation in
[diffusion_models.ipynb](diffusion_models.ipynb) and
[diffusion_utilities.ipynb](diffusion_utilities.ipynb) — a context-conditioned
U-Net trained to generate 16x16 sprite images.

## 1. Core Idea

A diffusion model learns to generate data by reversing a gradual noising
process:

- **Forward process (fixed, not learned):** start from a real image `x_0` and
  add a little Gaussian noise at each of `T` timesteps until, at `x_T`, the
  image is indistinguishable from pure noise.
- **Reverse process (learned):** train a neural network to predict the noise
  that was added at each step, so that starting from pure noise `x_T` and
  repeatedly subtracting the predicted noise reconstructs a realistic image
  `x_0`.

The network never directly predicts the image — it predicts the **noise
component** `ε` at a given noisy image `x_t` and timestep `t`. This is easier
to learn than predicting the clean image directly.

## 2. Forward Process & Noise Schedule

Controlled by a variance schedule `β_t` that grows linearly from `β1` to `β2`
over `T` timesteps (`timesteps = 500`, `beta1 = 1e-4`, `beta2 = 0.02` in the
notebook):

```python
b_t  = (beta2 - beta1) * linspace(0, 1, T+1) + beta1     # β_t
a_t  = 1 - b_t                                            # α_t
ab_t = cumprod(a_t)                                       # ᾱ_t (cumulative product)
```

- `β_t` — how much noise is injected at step `t`.
- `α_t = 1 − β_t` — how much of the signal survives step `t`.
- `ᾱ_t = Π α_i` — cumulative signal retained from step 0 to `t`. This lets you
  jump directly from `x_0` to a noisy `x_t` in one step instead of iterating:

```
x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε,      ε ~ N(0, I)
```

This is exactly the `perturb_input` helper:

```python
def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
```

- Small `t` → mostly signal, a little noise.
- Large `t` → mostly noise, almost no signal.

**Training** samples a random image, a random timestep `t`, and random noise
`ε`, computes the noisy `x_t` via `perturb_input`, and trains the network to
predict `ε` from `(x_t, t)` — typically with a simple MSE loss between
predicted and true noise (`||ε − ε_θ(x_t, t)||²`).

## 3. Reverse (Denoising) Process

At sampling time, the true `x_0` is unknown, so the model iterates backward
from pure noise, at each step removing its *predicted* noise and adding a
small amount of fresh Gaussian noise back in (except at the very last step):

```python
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise
```

Full sampling loop (`sample_ddpm`): start with `x_T ~ N(0, I)`, then for
`t = T ... 1`:
1. Predict the noise: `eps = nn_model(x_t, t)`.
2. Compute the mean of `x_{t-1}` by subtracting a scaled version of `eps`.
3. Add back a small amount of fresh noise `z` (unless `t == 1`).

**Key pitfall demonstrated in the notebook** (`sample_ddpm_incorrect`): if you
skip re-adding noise at every step (`z = 0` always), sampling collapses into
blurry/washed-out, low-diversity outputs. Re-injecting noise at each reverse
step is essential for the reverse process to correctly model the stochastic
distribution — it's not a bug to be optimized away.

## 4. Network Architecture — `ContextUnet`

A U-Net with skip connections, additionally conditioned on **timestep** and
an optional **context label** (e.g. sprite category: human / non-human / food
/ spell / side-facing).

```
x → init_conv (ResidualConvBlock)
  → down1 (UnetDown)  → down2 (UnetDown) → to_vec (AvgPool + GELU) = hiddenvec
                                                     │
                          embed t, c at two resolutions (EmbedFC)
                                                     │
  hiddenvec → up0 (ConvTranspose, upsample to bottleneck size)
  up0, scaled by (cemb1, temb1) → up1 (UnetUp, + down2 skip connection)
  up1, scaled by (cemb2, temb2) → up2 (UnetUp, + down1 skip connection)
  up2, concat with x → out (Conv layers) → predicted noise ε
```

Building blocks (`diffusion_utilities.ipynb`):

| Block | Role |
|---|---|
| `ResidualConvBlock` | Conv → BatchNorm → GELU, twice, with a residual (skip) add when `is_res=True`. Standard conv building block used everywhere in the U-Net. |
| `UnetDown` | Two `ResidualConvBlock`s + `MaxPool2d(2)` — halves spatial resolution, extracts features (the encoder path). |
| `UnetUp` | `ConvTranspose2d` (upsample) + two `ResidualConvBlock`s. Concatenates the corresponding encoder skip connection before convolving — this is the classic U-Net trick that preserves fine spatial detail lost during downsampling. |
| `EmbedFC` | A tiny 2-layer MLP (`Linear → GELU → Linear`) that projects a scalar/vector (timestep or context label) into an embedding vector matching a feature map's channel count. |

**Why timestep and context are embedded and injected, not concatenated as
extra channels:** the embeddings are reshaped to `(batch, channels, 1, 1)`
and combined with feature maps as `cemb * up + temb` — multiplying by the
context embedding and adding the time embedding at *two different
resolutions* in the decoder. This lets the network use the same weights to
process images at very different noise levels and (optionally) different
class conditions, since `t` and `c` modulate the features at every stage
rather than being fixed at the input.

**Context masking / classifier-free-style conditioning:** if no context `c`
is passed, it's zeroed out (`c = torch.zeros(...)`), so the same network can
run both conditionally and unconditionally — the basis for classifier-free
guidance (blending conditional and unconditional predictions at sampling
time to control how strongly generation follows the label), even though this
notebook's sampling functions don't yet unpack that step.

## 5. Data Pipeline

- `CustomDataset` loads pre-rendered 16x16 sprite images (`.npy` array) and
  one-hot category labels, applying `ToTensor()` then
  `Normalize((0.5,), (0.5,))` to map pixel values from `[0,1]` to `[-1,1]`
  (diffusion models are typically trained/sampled in a symmetric `[-1, 1]`
  range, matching the Gaussian noise they're trained against).
- `null_context=True` zeroes out labels, useful for training the
  unconditional branch needed for classifier-free guidance.

## 6. Hyperparameters Used

| Name | Value | Meaning |
|---|---|---|
| `timesteps` | 500 | Number of forward/reverse diffusion steps `T` |
| `beta1`, `beta2` | 1e-4, 0.02 | Start/end of the linear noise variance schedule |
| `n_feat` | 64 | Base channel width of the U-Net |
| `n_cfeat` | 5 | Number of context/class categories |
| `height` | 16 | Image is 16x16 (must be divisible by 4 for two down-sampling stages) |

## 7. Glossary

- **DDPM** — Denoising Diffusion Probabilistic Model; the specific
  forward/reverse formulation used here (Ho et al., 2020).
- **`ε` (epsilon) / noise prediction** — the network's output; the model is
  an "ε-predictor," not a direct image generator.
- **Noise schedule** — the sequence `β_1...β_T` controlling how aggressively
  noise is added per step; linear here, though cosine schedules are also
  common in later work.
- **Context conditioning** — supplying auxiliary information (class label,
  text embedding, etc.) so generation can be steered rather than purely
  unconditional.
- **U-Net** — encoder-decoder CNN with skip connections between matching
  resolutions; the near-universal backbone for image diffusion models.

## 8. Beyond This Notebook (not implemented here, but worth knowing)

- **Classifier-free guidance:** sample both with and without context, then
  extrapolate: `ε_guided = ε_uncond + w · (ε_cond − ε_uncond)` to trade off
  fidelity to the prompt vs. sample diversity.
- **DDIM sampling:** a deterministic, non-Markovian reverse process that
  reaches similar quality in far fewer steps (e.g. 50 instead of 500) by
  skipping timesteps — much faster than the full DDPM loop shown here.
- **Latent diffusion (e.g. Stable Diffusion):** run the whole forward/reverse
  process in a compressed VAE latent space instead of pixel space, making
  high-resolution image generation tractable.




```
    https://claude.ai/code/artifact/5a42b5c6-1fa9-47f2-b78b-977e4d29dbe1?via=auto_preview

```

```
    https://claude.ai/code/artifact/740ee28a-d419-4e59-afc7-bd7746505ce2?via=auto_preview
    
```


