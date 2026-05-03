# Zero-Shot Object Localization with SAM + CLIP

Locate any object in an image using a plain text description; no training, no labels, no fine-tuning required.

Given a prompt like *"Pick the screwdriver"*, the pipeline segments the image into candidate regions using **SAM**, ranks them by semantic similarity using **CLIP**, and returns a bounding box around the best match.

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14LxTFpd5VV08mCazm3ZQhklOSjL-gEy3?usp=sharing)

---

## Demo

[![Watch the Demo](https://img.shields.io/badge/Watch-Video%20Demo-blue?logo=google-drive)](https://drive.google.com/file/d/108iFDh1PYlA7dnyt4QUayU_V7o_BpJvc/view?usp=sharing)

> *"Pick the screwdriver"* → SAM proposes regions → CLIP ranks by text similarity → bounding box returned

---

## How It Works


```
Image + Text Prompt
       │
       ▼
┌─────────────┐     class-agnostic      ┌──────────────┐
│     SAM     │  ──── region masks ───▶ │    Crops     │
│  (ViT-B)   │                          │  (N regions) │
└─────────────┘                         └──────┬───────┘
                                               │
                                      CLIP image encoder
                                               │
                                               ▼
                                    ┌──────────────────┐
  Text Prompt ──── CLIP text ──────▶│ Cosine Similarity│
                   encoder          │   Ranking (top-k)│
                                    └────────┬─────────┘
                                             │
                                             ▼
                                      Bounding Box(es)
```

1. **SAM** generates up to 100 class-agnostic region proposals via a grid of prompt points
2. Each region is cropped from the original image
3. **OpenCLIP** encodes both the crops and the text prompt into a shared embedding space
4. Regions are ranked by cosine similarity, the top-k matches are returned with bounding boxes

---

## Models

| Model | Variant | Source |
|---|---|---|
| Segment Anything (SAM) | ViT-B | Meta AI |
| OpenCLIP | ViT-B/32, LAION-2B | mlfoundations |

---

## Quick Start

### Run in Colab (no setup required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14LxTFpd5VV08mCazm3ZQhklOSjL-gEy3?usp=sharing)

1. Open the notebook via the badge above
2. Run all cells in order (Runtime → Run all)
3. When prompted, choose:
   - **Option 1** → uses the sample image from this repo automatically
   - **Option 2** → upload your own image from your device
4. Enter a text prompt describing the object you want to locate
5. The result image with bounding box is displayed and saved

### Run locally

```bash
git clone https://github.com/KomisD/Zero-Shot-Object-Localization.git
cd Zero-Shot-Object-Localization
pip install open_clip_torch git+https://github.com/facebookresearch/segment-anything.git opencv-python Pillow tqdm
jupyter notebook notebook.ipynb
```

---

## Example Prompts

```
"Pick the screwdriver"
"Take the red cup"
"Find the keyboard"
"Grab the bottle on the left"
```

---

## Key Design Choices

**Why mask background before CLIP encoding?**
Setting `apply_mask_to_crop=False` keeps the surrounding context visible to CLIP. For scene-level and relational prompts (*"the cup on the left"*), context improves matching. Set to `True` for isolated object queries.

**Why SAM ViT-B and not ViT-H?**
ViT-B runs on free-tier Colab GPU within reasonable time (~30–60s per image). ViT-H gives better boundary quality but requires more memory. Swap the checkpoint path and registry key to upgrade.

---

## Limitations

- Accuracy depends on SAM's segmentation coverage — heavily occluded objects may not get a clean proposal
- Ambiguous or abstract prompts benefit from a larger CLIP backbone (ViT-L/14)
- `input()` prompts require Colab — for scripted use, replace with hardcoded values

## Future Extensions

- Upgrade to **SAM 2** for improved boundary quality
- Swap to **ViT-L/14 CLIP** for finer semantic precision
- Add confidence thresholding to suppress low-similarity matches
- Extend to video with temporal consistency across frames

---

## Project Structure

```
Zero-Shot-Object-Localization/
├── notebook.ipynb          # Main notebook
├── test_image.png          # Sample image (auto-loaded in Option 1)
├── examples/
│   └── result.jpg          # Output preview
└── README.md
```

---

> **License:** MIT
> **Contact:** [komdimos@gmail.com](mailto:komdimos@gmail.com)
