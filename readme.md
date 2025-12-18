# SoFlow: Solution Flow Models for One-Step Generative Modeling

This is the official PyTorch implementation for the paper **SoFlow: Solution Flow Models for One-Step Generative Modeling**.

Arxiv: https://arxiv.org/abs/2512.15657

![](demo.png)
## Code Structure

The code structure of this repository is straightforward:

* `latent_dataset.py`: Contains the DDP code for VAE latent pre-extraction for ImageNet-256x256 conditional generation using [SD-VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse).  
* `dit.py`: Contains the Diffusion Transformer implementation from the [DiT repository](https://github.com/facebookresearch/DiT) with slight modifications to adapt to our model.  
* `augmentation.py` and `unet.py`: Contain the data augmentation and U-Net for unconditional CIFAR-10 generation used in the [EDM repository](https://github.com/NVlabs/edm), with some slight modifications to adapt to our model.  
* `models.py`: Contains our model's implementation including loss computation.  
* `train.py` and `inference.py`: Contain our DDP training and inference code.  
* `evaluator.py`: Contains the standard evaluation code provided in the [ADM repository](https://github.com/openai/guided-diffusion/tree/main/evaluations), modified slightly for more convenient usage.

Our checkpoints are available at: https://huggingface.co/zlab-princeton/SoFlow.

## 1. Setup Conda Environments

Training and Inference Environment
```bash
conda create -n soflow python=3.10 -y  
conda activate soflow  
pip install -r requirements_soflow.txt
```
Evaluation Environment
```bash
conda create -n soflow_eval python=3.10 -y
conda activate soflow_eval  
pip install -r requirements_soflow_eval.txt
```
## 2. Training SoFlow Models
**Dataset Preparation:**

For **ImageNet** training, latent_dataset.py processes the raw ImageNet dataset from --data-path and saves it to --save-path using DDP. Specifically, different processes will process the dataset into several HDF5 files and finally merge them into a single file named imagenet_latent.hdf5.

```bash
conda activate soflow  
torchrun --nproc-per-node=8 latent_dataset.py --data-path ./imagenet --save-path ./imagenet_latent --image-size 256 --device-batch-size 256 --num-workers 4 --seed 42
```

For **CIFAR-10** training, no data processing is required; you can run the training process directly.

To ensure ease of use, we have integrated all hyperparameters into a YAML file. We have provided detailed explanations for the hyperparameters inside `imagenet.yaml` and `cifar.yaml`. To launch a new training process, you can simply run:
```bash
conda activate soflow  
torchrun --nproc-per-node=8 train.py --config your_config.yaml
```
Some implementation details:

* **Directory Management:** If `working_dir` does not exist, the training command will create it and copy the training config YAML file into it. After training, `working_dir` will contain a `config.yaml` file, a `log.txt` file, and three folders: `ckpts`, `evals`, and `figs`.  
* **Resuming Training:** If the training command is run with an existing `working_dir`, the program will automatically load the latest checkpoints from `working_dir/ckpts` to continue the training process.  
* **Visualization:** For every `eval_demo_steps` steps configured, the program will automatically generate an image mesh with shape `eval_demo_shape` using `eval_NFE` inference steps for visualization.  
* **Checkpoints & Evaluation:** For every eval_step steps configured, the program will automatically save checkpoints to `ckpts` and save 50,000 inference images (with `eval_NFE` inference steps) to a .npz file in evals for evaluation.

Some Training Tips:
The schedule function $r(k,K)$ influences convergence speed. If you plan to train a model for only a few steps, you can choose a smaller `l_init_ratio` or simply use a constant schedule.
Additionally, when using a non-constant schedule, the defined **total steps** will affect performance at the current step. This is because the schedule decays more slowly as the total steps increase. For example, comparing two models both trained for 200k iterations, the one scheduled for 400k total steps will outperform the one scheduled for 800k total steps.


## 3. Inference with Pretrained Checkpoints and Evaluation

You can download our checkpoints and their training configs, along with the CIFAR-10 and ImageNet 256x256 reference files, by cloning the repository:

```bash
git clone https://huggingface.co/zlab-princeton/SoFlow
```
* The reference file for the **ImageNet 256x256** dataset is downloaded from the [ADM repository](https://github.com/openai/guided-diffusion/tree/main/evaluations).  
* The reference file for the **CIFAR-10** dataset is the training set containing 50,000 images, following previous works.

### ImageNet 256x256

For ImageNet checkpoints, using the commands below will achieve a 1-NFE / 2-NFE FID-50K of 2.9617 / 2.6606.

```bash
conda activate soflow  

torchrun --nproc-per-node=8 inference.py --config ./SoFlow/XL-2-cond/config.yaml --ckpt-steps 1200000 --eval-NFE 1 --eval-batch-size 125 --seed 42

torchrun --nproc-per-node=8 inference.py --config ./SoFlow/XL-2-cond/config.yaml --ckpt-steps 1200000 --eval-NFE 2 --eval-batch-size 125 --seed 42

conda activate soflow_eval  

python evaluator.py --ref_batch ./SoFlow/Ref/VIRTUAL_imagenet256_labeled.npz --sample_batch_dir ./SoFlow/XL-2-cond/evals
```

**Performance for other ImageNet 256x256 models:**

| Models | Train Epochs | 1-NFE FID-50K | 2-NFE FID-50K |
| :---- | :---- | :---- | :---- |
| SoFlow-B/4 (uncond) | 80 | 58.5646 | 58.2240 |
| SoFlow-B/4 (cond) | 80 | 11.5897 | 8.2212 |
| SoFlow-B/2 (cond) | 240 | 4.8491 | 4.2435 |
| SoFlow-M/2 (cond) | 240 | 3.7329 | 3.4229 |
| SoFlow-L/2 (cond) | 240 | 3.2007 | 2.8995 |
| SoFlow-XL/2 (cond) | 240 | **2.9617** | **2.6606** |

The results in this table are achieved by standard DiT architectures, only with slight modification by adding another time embedding.

### CIFAR-10

For CIFAR-10, using the commands below will achieve a 1-NFE / 2-NFE FID-50K of 2.8600 / 2.2827.

```bash
conda activate soflow 

torchrun --nproc-per-node=8 inference.py --config ./SoFlow/UNet-uncond/config.yaml --ckpt-steps 800000 --eval-NFE 1 --eval-batch-size 125 --seed 42

torchrun --nproc-per-node=8 inference.py --config ./SoFlow/UNet-uncond/config.yaml --ckpt-steps 800000 --eval-NFE 2 --eval-batch-size 125 --seed 42

conda activate soflow_eval  

python evaluator.py --ref_batch ./SoFlow/Ref/cifar10.npz --sample_batch_dir ./SoFlow/UNet-uncond/evals  
```

Our evaluator can automatically evaluate all npz files in --sample_batch_dir and save the results into `eval_results.txt`.
