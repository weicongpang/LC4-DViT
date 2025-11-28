# LC4-DViT: Land-cover Creation for Land-cover Classification with Deformable Vision Transformer

## Project Introduction

Land-cover underpins ecosystem services, hydrologic regulation, disaster-risk reduction, and evidence-based land planning; timely, accurate land-cover maps are therefore critical for environmental stewardship. Remote sensing-based land-cover classification offers a scalable route to such maps but is hindered by scarce and imbalanced annotations and by geometric distortions in high-resolution scenes. We propose LC4-DViT (Land-cover Creation for Land-cover Classification with Deformable Vision Transformer), a framework that combines generative data creation with a deformation-aware Vision Transformer. A text-guided diffusion pipeline uses GPT-4o–generated scene descriptions and super-resolved exemplars to synthesize class-balanced, high-fidelity training images, while DViT couples a DCNv4 deformable convolutional backbone with a Vision Transformer encoder to jointly capture fine-scale geometry and global context. On eight classes from the Aerial Image Dataset (AID)—Beach, Bridge, Desert, Forest, Mountain, Pond, Port, and River—DViT achieves 0.9572 overall accuracy, 0.9576 macro F1-score, and 0.9510 Cohen’s Kappa, improving over a vanilla ViT baseline (0.9274 OA, 0.9300 macro F1, 0.9169 Kappa) and outperforming ResNet50, MobileNetV2, and FlashInternImage. Cross-dataset experiments on a three-class SIRI-WHU subset (Harbor, Pond, River) yield 0.9333 overall accuracy, 0.9316 macro F1, and 0.8989 Kappa, indicating good transferability. An LLM-based judge using GPT-4o to score Grad-CAM heatmaps further shows that DViT’s attention aligns best with hydrologically meaningful structures. These results suggest that description-driven generative augmentation combined with deformation-aware transformers is a promising approach for high-resolution land-cover mapping.

## Project Structure

```
LC4-DViT/
├── config.py                          # Global configuration file
├── train.py                           # Training script
├── test.py                            # Testing script
├── atten_heatmap.py                   # Attention Heatmap Visualization
├── classification_to_name.json        # Class name mapping file
├── models/                            # Model definition directory
│   ├── build.py                       # Model Parameters Definition
│   ├── flash_intern_image.py          # FlashInternImage model
│   ├── intern_image.py                # InternImage model
│   ├── vit_dcnv4.py                   # DViT model
│   ├── mobilenetv2.py                 # MobileNetV2 model
│   ├── vit.py                         # ViT Model
│   └── DCNv4/                         # DCNv4 operation module
├── DCNv4_op/                          # Functions of DCNv4
├── ops_dcnv3/                         # Functions of DCNv3
├── environment.yaml                   # Project's environment setup
```


## Supported Models

1. **Vision Transformer**
2. **FlashInternImage**
3. **DViT (Deformable Vision Transformer)**
4. **MobileNetV2**
5. **Resnet50**


## Running Environment

- Python 3.12+
- PyTorch 2.5.1+cu124
- GPU: RTX 4090(24GB) * 1    
- CPU: 16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz

## Installation Steps

1. **Clone the project and initiate conda environment**
```bash
git clone https://github.com/weicongpang/LC4-DViT.git
cd LC4-DViT
conda create -n lcv2dvit python=3.12
conda activate lcv2dvit
```

2. **Install relevant dependencies**

```bash
pip install -r requirements.txt
```

For installing DCNv4, please go to the path of DCNv4_op, and run the command
```bash
python setup.py build install
```

3. **Train and Test** 
Open ```models/build.py```, modify this code ```model_type = 'flash_intern_image' ``` based on the model you want to train.
You can also modify the training parameters in ```models/build.py```.

Modify the following configurations based on the path of your dataset and the path of your weights file.
You can specify the directories that you want to save your results. 
```
TEST_DATASET_DIR = '/path/to/test/file'
WEIGHTS_PATH = '/path/to/trained_model.pth'
CLASS_MAP_PATH = '/path/to/classification_to_name.json'
RESULTS_DIR = '/path/to/train_result_directory'

CM_FILENAME = 'confusion_matrix_dvit.png'
RESULTS_FILENAME = 'results.txt'
```

Run ```python train.py``` for training.

During training, the script will:
- Automatically create timestamped result directories
- Save the best model weights
- Periodically save checkpoints
- Record training logs
- Generate training curve plots


Run ```python test.py``` for testing. 

The testing script will:
- Load trained model weights
- Evaluate performance on test set
- Generate confusion matrices
- Calculate various classification metrics
- Save detailed result reports

## Processed Dataset
We also provided our processed dataset, which is available at *[Processed Data](https://huggingface.co/datasets/williampang/AIDGTFGEN/tree/main)*.

## Pretrained Models
We also provided our proposed pretrained checkpoint DViT model, which is available at *[Checkpoint Model](https://huggingface.co/williampang/LC4-dvit_all_models/blob/main/final_trained_dvit.pth)*.

## Performance Metrics

The testing script calculates the following metrics:
- **Overall Accuracy (OA)**: Overall accuracy
- **Mean Accuracy (mAcc)**: Mean class accuracy
- **Cohen-Kappa**: Kappa coefficient
- **Precision/Recall/F1-score**: Precision/Recall/F1-score, all in macro
- **Per-class Accuracy/Precision/Kappa/F1-score**: Accuracy for each class






