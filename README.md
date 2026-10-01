# Semantic Segmentation of Mechanical Parts

A PyTorch semantic-segmentation project for identifying six types of mechanical parts at pixel level.

## Results

| Pipeline | Validation mean Dice |
|---|---:|
| U-Net | **0.9721** |
| U-Net + TTA + connected-component cleanup | **0.9740** |

The final validation result was obtained on a held-out set of **301 images** from the 2,000 labelled training images.

## Problem

Each 384×384 RGB image may contain several mechanical parts on different backgrounds. Every pixel is classified as:

- Background
- `hex_nut`
- `washer`
- `bolt`
- `ball_bearing`
- `spring`
- `o_ring`

The competition evaluates predictions using mean Dice across image/class pairs.

## Approach

### Data validation
The notebook verifies the competition's RLE convention, checks RLE round-trips, correctly parses class names containing underscores, and confirms that PNG masks agree with masks reconstructed from the training CSV.

### Train/validation split
The 2,000 labelled images are split into **1,699 training** and **301 validation** images while approximately preserving background-material and scene-clutter distributions.

### Data augmentation
Training images and masks receive synchronized random horizontal flips, vertical flips, and 90° rotations.

### U-Net
A U-Net is implemented from scratch in PyTorch with four encoder stages, a bottleneck, four decoder stages, convolution/BatchNorm/ReLU blocks, max pooling, transposed-convolution upsampling, and skip connections.

### Loss
Training uses an equal-weight combination of **Cross-Entropy Loss** and **Soft Dice Loss**.

### Training
- Epochs: 25
- Batch size: 16
- Optimizer: Adam
- Initial learning rate: `1e-3`
- Scheduler: cosine annealing
- Best checkpoint selected by validation mean Dice

### Test-Time Augmentation
Six orientations are evaluated at inference time: original, horizontal flip, vertical flip, and 90°/180°/270° rotations. Predictions are transformed back and averaged.

### Post-processing
Connected components smaller than 15 pixels are removed from foreground classes to suppress isolated prediction noise.

### Submission
Final masks are converted to the competition's column-major RLE format and written to `submission.csv`.

## Repository structure

```text
semantic-segmentation/
├── README.md
├── semantic_segmentation_portfolio.ipynb
├── requirements.txt
├── .gitignore
└── results/
    └── validation_prediction.png
```

## Dataset setup

The dataset is intentionally **not included** in this repository.

Use:

```text
data/
├── metadata.csv
├── train/
│   ├── train.csv
│   ├── images/
│   └── masks/
└── test/
    └── images/
```

The notebook defaults to `data/`. For another location:

```bash
export SEGMENTATION_DATA_DIR=/path/to/Dataset
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `semantic_segmentation_portfolio.ipynb`.

## Reproducibility

The notebook contains the executed training/validation results from the original experiment. Re-running training requires the competition dataset and a suitable PyTorch environment; GPU execution is recommended.

The trained model weights are intentionally not included because they are generated artifacts.

## Skills demonstrated

- PyTorch
- Semantic segmentation
- U-Net
- Custom Dataset/DataLoader
- Image augmentation
- Multi-class pixel classification
- Dice-based evaluation
- Custom differentiable loss
- Learning-rate scheduling
- Test-time augmentation
- Connected-component post-processing
- RLE submission generation
- Validation and sanity checks
