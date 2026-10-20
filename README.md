# ComfyUI-MOSS-GGUF — GGUF + BF16

**Modular MOSS-Image GGUF and BF16 loading for ComfyUI — by [Rebel AI](https://huggingface.co/realrebelai).**

Run quantized **OpenMOSS MOSS-Image** diffusion transformers through ComfyUI with separate, dropdown-based loaders for the GGUF transformer, Qwen3.5 text encoder, and VAE. Supports both **MOSS-Image-SFT** and **MOSS-Image-Base** checkpoints when converted to the compatible MOSS GGUF format.

> **Status: experimental.** End-to-end inference has worked with a locally generated Base Q4_K_S GGUF. The latest attention adapter has been checked with synthetic numerical tests, but complete equivalence to upstream BF16 inference, all quantization tiers, and SFT image quality have **not** been established. This is a community integration, not an official OpenMOSS release.

## Features

- **GGUF and BF16 transformer selection in one node:** Choose a quantized `.gguf` or an unquantized `.safetensors` from the same dropdown. BF16 uses the official OpenMOSS model-state loader, not GGUF ops.\n- **MOSS-specific GGUF transformer loader:** Uses the official MOSS model architecture with the quantized operations from `city96/ComfyUI-GGUF`.
- **Separate text encoder loader:** Loads the MOSS Qwen3.5 encoder through the official OpenMOSS loader.
- **Separate VAE loader:** Loads the MOSS-compatible VAE.
- **Dropdown discovery:** Scans standard ComfyUI model directories and subfolders, with no absolute file paths required in the workflow.
- **Model inventory:** Reports which compatible files the node package can discover.
- **Base and SFT:** The same loader handles compatible quantizations of either checkpoint; choose the corresponding GGUF in the dropdown.
- **Local attention dtype adaptation:** Handles GGUF integer-weight storage and mixed-precision attention without editing upstream MOSS or ComfyUI-GGUF source files.

The runtime consists of `__init__.py`, `discovery.py`, `runtime.py`, and `bf16.py`. **No audit scripts or development utilities are required to generate images.**

## Requirements

1. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with a compatible PyTorch installation.
2. The **official OpenMOSS MOSS-Image ComfyUI integration**, including its MOSS model and encoder loaders: [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image/tree/main/comfyui).
3. [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).
4. A compatible MOSS-Image **DiT GGUF**, a **Qwen3.5-9B text encoder**, and a **MOSS / FLUX.2-compatible VAE**.

**Dependency folder names matter:** The current runtime discovers upstream modules by their loaded file locations. Keep the upstream MOSS integration directory named `moss-image` (or `moss_image`) and the city96 plugin directory containing `ComfyUI-GGUF`. **This repository's own folder may be renamed.** Do not rename the dependency directories arbitrarily.

## Installation

### Manual installation

Place the three runtime files together in:

```text
ComfyUI/
└── custom_nodes/
    ├── ComfyUI-MOSS-GGUF/
    │   ├── __init__.py
    │   ├── discovery.py
    │   └── runtime.py
    ├── ComfyUI-GGUF/
    └── moss-image/
```

Install the official MOSS and ComfyUI-GGUF integrations following **their** installation instructions and install their required Python dependencies. This bridge does not replace either upstream integration.

Restart ComfyUI after installing or updating the nodes.

### Git installation (after this repository is published)

Clone **your published repository URL** into `ComfyUI/custom_nodes/` and restart ComfyUI. The repository name can be `ComfyUI-MOSS-GGUF`; no hardcoded local `D:` drive path is needed for the node package.

## Model files

The GGUF is the **diffusion transformer only**. The encoder and VAE are separate components.

| Component | Recommended location | Supported selection |
| --- | --- | --- |
| MOSS-Image GGUF or BF16 (SFT or Base) | `ComfyUI/models/diffusion_models/` or `ComfyUI/models/unet/` | `.gguf`, `.safetensors` (BF16 only) |
| Qwen3.5-9B text encoder | `ComfyUI/models/text_encoders/` or `ComfyUI/models/clip/` | `.safetensors`, `.pt`, `.pth` |
| MOSS / FLUX.2-compatible VAE | `ComfyUI/models/vae/` | `.safetensors`, `.pt`, `.pth` |

Subfolders are supported. The dropdown displays a category-prefixed relative path such as `diffusion_models/MOSS/MOSS-Image-SFT-Q4_K_S.gguf`.

**Use SFT for your primary image-generation tests.** Base is also supported and may be useful for training-oriented work or comparisons. The loader does not transform Base weights into SFT weights.

## Nodes

All four nodes appear under **`MOSS-Image/Rebel (GGUF)`** (category retained for existing workflows). The transformer's input socket is still named `gguf` for backward compatibility, but the dropdown now also accepts BF16 safetensors.

| Node as displayed in ComfyUI | Inputs | Output | Purpose |
| --- | --- | --- | --- |
| **MOSS-Image GGUF / BF16 Transformer (Rebel)** | `gguf`, `compute_precision` | `MODEL` | Loads a MOSS-format DiT GGUF or BF16 safetensors through the appropriate backend. |
| **MOSS-Image Qwen3.5 Encoder (Rebel)** | `encoder`, `device` | `CLIP` | Loads the MOSS text encoder through the official OpenMOSS loader. |
| **MOSS-Image VAE (Rebel)** | `vae` | `VAE` | Loads the VAE for decoding. |
| **MOSS Model Inventory (Rebel)** | None | `STRING` and UI report | Lists model search paths and discovered files. |

**Transformer compute precision:** `bf16` (default) or `fp32`. The encoder offers `default` and `cpu` placement. These are the options implemented by the current nodes.

### Basic workflow

```text
MOSS-Image GGUF Transformer (Rebel)
  → ModelSamplingSD3 (shift 5)
  → SkipLayerGuidanceDiTSimple (single_layers "22", start 0, end 1)
  → KSampler [MODEL]

MOSS-Image Qwen3.5 Encoder (Rebel)
  → CLIPTextEncode (positive prompt) → KSampler [positive]
  → CLIPTextEncode (empty negative)  → KSampler [negative]

EmptyFlux2LatentImage (1024 × 1024) → KSampler [latent_image]

KSampler [samples] → VAEDecode → SaveImage
MOSS-Image VAE (Rebel) → VAEDecode [vae]
```

### Recommended first-test settings

These are the recommended OpenMOSS starting settings, **not a benchmark result for every quantization**.

| Setting | Value |
| --- | --- |
| Transformer | **MOSS-Image-SFT GGUF** (start with Q4_K_S if available) |
| Resolution / latent | `EmptyFlux2LatentImage`, `1024 × 1024` |
| Steps | `50` |
| CFG | `3.0` |
| Sampler | `euler` |
| Scheduler | `sgm_uniform` |
| `ModelSamplingSD3` shift | `5.0` |
| `SkipLayerGuidanceDiTSimple` | `single_layers: "22"`, `start_percent: 0`, `end_percent: 1` |
| Denoise | `1.0` |
| Negative prompt | Empty |
| Compute precision | `bf16` (or `fp32` if debugging precision behavior) |

Use a fixed seed when comparing SFT vs. Base or Q4 vs. Q5. Changing the model variant does not require changing the nodes.

## What this loader does — and does not do

**GGUF path:** The bridge reads a GGUF with the MOSS architecture marker, checks essential tensor names, calls the official MOSS model-state loader with city96 GGUF operations, wraps the result in the GGUF-aware model patcher, and applies a per-instance compatibility adapter for integer-weight dtype handling and attention.

**Not a generic GGUF loader:** Ordinary GGUF LLM applications such as LM Studio and llama.cpp text inference do not run this image-generation DiT. Generic ComfyUI GGUF loaders may also lack MOSS architecture support.

**BF16 path:** Original `moss_image_sft_bf16.safetensors`, `moss_image_base_bf16.safetensors`, and complete LoRA-merged BF16 checkpoints can be selected. The loader maps fused `layers.N.proj.qkv.weight` tensors to the official separate q/k/v projection layout using zero-copy views. The actual BF16 model and the Qwen encoder are still subject to ComfyUI's memory management; BF16 inference has not yet been GPU-validated on this Rebel node.\n\n**Not a W4A8 loader:** W4A8 ConvRot `.safetensors` is a different quantization format. **Do not select W4A8 files in the GGUF transformer dropdown.** Use a separately verified compatible safetensors quantization path.

**No model weights included:** Download the model, encoder, and VAE separately. This GitHub repository is for the node integration only.

## Troubleshooting

**No GGUF / BF16 / encoder / VAE in the dropdown**

- Check the corresponding `models/` directory above; subfolders are supported.
- Add **MOSS Model Inventory (Rebel)** to see exactly where the package searches.
- Restart ComfyUI after adding or moving model files.

**Missing plugin module (`moss` or `gguf`)**

Install and enable the official OpenMOSS integration and city96 ComfyUI-GGUF. Confirm their folder names match the dependency-folder expectations above, then restart.

**`Expected MOSS GGUF (moss_image)`**

The file was not converted with the expected MOSS GGUF architecture metadata. A GGUF extension alone is not sufficient.

**`Official MOSS source changed ... Refusing an incomplete math patch`**

The upstream model code differs from the version this adapter expects. The loader deliberately stops instead of applying an unverified partial patch. Check version compatibility.

**Attention Q/K/V dtype error**

Confirm you are using the corrected runtime and the official dependency plugins. The bridge is designed to align attention-kernel input dtypes; it does not guarantee compatibility with every upstream update.

**Images are dark, soft, or badly composed**

Confirm the official workflow settings, the FLUX.2 latent node, the correct VAE, and the correct SFT source checkpoint. Low-bit quantization can degrade detail. Successful execution does not guarantee BF16-equivalent image quality.

**CUDA out of memory / unexpectedly slow**

Model file size is not peak VRAM usage. The encoder, VAE, attention activations, resolution, and CPU offloading also consume resources. Try your normal ComfyUI low-VRAM settings or CPU encoder placement as appropriate.

## Compatibility and release status

- The Base Q4_K_S-v2 GGUF has completed local image-generation smoke tests on an RTX 3070-class GPU.\n- BF16 checkpoint loading, QKV mapping, and attention adaptation are newly added and have **not** been end-to-end verified on a local RTX 3070. The earlier `Cannot copy out of meta tensor; no data!` error was observed in the **negative prompt/text encoder** path, so enabling BF16 transformer loading does not establish a fix for that separate problem.
- The current adapter's attention behavior has undergone synthetic numerical checks; **that is not a full GPU end-to-end correctness certification**.
- SFT, other quantization tiers, upstream dependency updates, and W4A8 paths must not be represented as comprehensively validated.
- No benchmark FPS, VRAM guarantee, or image-quality parity claim is made here.

If reporting a bug, include the exact GGUF filename, ComfyUI version, OpenMOSS integration version, city96 plugin version, GPU, and traceback.

## Credits and licensing

- **MOSS-Image architecture, model weights, and official ComfyUI implementation:** [OpenMOSS Team](https://github.com/OpenMOSS/MOSS-Image)
- **GGUF backend:** [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- **Community integration and quantizations:** [Rebel AI](https://huggingface.co/realrebelai) · [X / @realrebelai](https://x.com/realrebelai)

This is an **unofficial** project. Keep the applicable upstream license and attribution notices. Before adding a license file to this GitHub repository, choose a license for **your original node code** and ensure it is compatible with the dependencies; the original MOSS model license does not automatically license unrelated original source code.
