# SketchNet-PortraitNet

Two small artwork classifiers built with transfer learning: **SketchNet** distinguishes paintings from sketches, while **PortraitNet** distinguishes portraits from general paintings.

This is a side project created as a useful tool to help my sister [ **:-)** ].

## Results

Each test set contains only artists absent from training and validation.

| Model | Test accuracy | Balanced accuracy | Test images |
|---|---:|---:|---:|
| SketchNet | **92.47%** | 89.56% | 664 |
| PortraitNet | **96.27%** | 96.19% | 617 |

See the [SketchNet](docs/sketchnet-evaluation.md) and [PortraitNet](docs/portraitnet-evaluation.md) evaluation notes for per-class results and confidence intervals.

## Quick start

Install [uv](https://docs.astral.sh/uv/), clone this repository, and create the locked environment from the project directory:

```bash
uv sync --frozen
```

The trained `.pt` model files are not stored in the source repository. Download `sketchnet.pt` and `portraitnet.pt` from the repository's **Releases** page, then place both files in a local `models` directory. With the [GitHub CLI](https://cli.github.com/) installed, this can be done from the cloned repository with:

```bash
mkdir -p models
gh release download --pattern 'sketchnet.pt' --pattern 'portraitnet.pt' --dir models
```

Run either classifier on one image:

```bash
uv run sketchnet predict path/to/image.jpg --checkpoint models/sketchnet.pt
uv run portraitnet predict path/to/image.jpg --checkpoint models/portraitnet.pt
```

Or run the two-stage classifier, which returns `sketch`, `portrait`, or `general_painting`:

```bash
uv run sketchnet classify path/to/image.jpg \
  --sketch-checkpoint models/sketchnet.pt \
  --portrait-checkpoint models/portraitnet.pt
```

To sort a folder with SketchNet, provide the folder containing the images:

```bash
uv run sketchnet sort path/to/images --checkpoint models/sketchnet.pt
```

The source folder is left unchanged. Images are copied recursively into two sibling folders:

```text
images_painting/
images_sketch/
```

PortraitNet works in the same way. For example, sort the paintings produced above with:

```bash
uv run portraitnet sort path/to/images_painting --checkpoint models/portraitnet.pt
```

This creates `images_painting_general_painting/` and `images_painting_portrait/`. Output folders must not already exist, which prevents accidental overwriting.

The training images are private and are not required for prediction. The model files are kept in GitHub Releases rather than in Git so the source repository remains lightweight.

## Architecture

Both classifiers start from a ResNet-50 pretrained on ImageNet. The final residual block and a new two-class output head are fine-tuned with cross-entropy loss and Adam (`lr=10^-4`, batch size 32). The checkpoint with the best validation balanced accuracy is retained.

## Train and evaluate

Reproducing the training runs requires a compatible image collection under `data/private/sketchnet` and `data/private/portraitnet`. The commands use Apple MPS below; omit `--device mps` to let the program choose an available device.

```bash
uv run sketchnet prepare
uv run sketchnet train --device mps
uv run sketchnet evaluate --device mps

uv run portraitnet prepare
uv run portraitnet train --device mps
uv run portraitnet evaluate --device mps

uv run pytest
```

## License

The source code and released model checkpoints are provided under the [MIT License](LICENSE). The private training images are not distributed and are not covered by this license.
