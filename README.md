Below is a suggested README.md you can place in your GitHub repository. This README is designed to be clear, attractive, and highlight the key steps you took, the files changed, and how to reproduce your results.

Vision Transformer (ViT) with Knowledge Distillation & Fine-Tuning on CIFAR-10

Author: Israel Oladeji, Harshita Parsup
Forked from: Google Research Vision Transformer repository

Introduction

This repository shows how to fine-tune a Vision Transformer (ViT) model on CIFAR-10 using JAX/Flax, integrate knowledge distillation from a stronger teacher model, and evaluate performance on the test set. The steps and changes described here are based on modifying the original vit_jax code provided by Google Research.

Key Papers:
	•	An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
	•	MLP-Mixer: An all-MLP Architecture for Vision
	•	How to train your ViT?
	•	When Vision Transformers Outperform ResNets without Pretraining

Changes and Enhancements
	1.	Knowledge Distillation (KD):
We introduced KD by loading a stronger teacher model (e.g., ViT-L_32) and guiding the student model (ViT-B_32) during training.
	2.	Pre-Preparing Datasets & Drive Integration:
To avoid TFDS preparation issues in the Colab TPU environment, we stored datasets in Google Drive and reused pre-processed data to reduce dataset preparation errors.
	3.	Fine-Tuning Hyperparameters:
We adjusted the learning rate, alpha (distillation weight), and temperature to improve KD performance.
	4.	Faster Training:
Reduced batch size and accum_steps for efficient TPU/GPU training. After caching datasets and downloading pre-trained weights, training took around 4 minutes for 100 steps.

Edited Files
	•	vit_jax/train.py:
Added teacher_apply_fn, teacher_params, alpha, and temperature arguments to make_update_fn and modified loss_fn to compute a KD loss (KL-divergence between teacher and student logits).
	•	vit_jax/configs/common.py and vit_jax/configs/models.py:
Used as-is but we reloaded configurations to choose ViT-B_32 as student and ViT-L_32 as teacher.
	•	vit_jax/input_pipeline.py:
We didn’t directly modify the logic here. Instead, we ensured we used a stable tfds_data_dir on Google Drive and removed dataset slicing (like train[:98%]) to prevent environment issues.

Note: The original vit_jax.ipynb was updated to integrate these changes. The colab now:
	•	Mounts Google Drive for persistent storage.
	•	Prepares and loads CIFAR-10 from the cached directory.
	•	Loads and fine-tunes ViT-B_32 with KD from ViT-L_32.
	•	Evaluates performance and plots training loss curves.

Steps to Reproduce
	1.	Clone the Repository:

git clone https://github.com/yourusername/vision_transformer.git
cd vision_transformer


	2.	Install Dependencies:
Make sure you have JAX, Flax, TFDS, and other packages installed:

pip install -r vit_jax/requirements.txt


	3.	Prepare the Dataset:
In a CPU/GPU environment:
	•	Mount your Google Drive.
	•	Set config.tfds_data_dir = '/gdrive/My Drive/my_tfds_data'.
	•	Load CIFAR-10 once to cache it locally:

ds = input_pipeline.get_data_from_tfds(config=config, mode='train')
ds_test = input_pipeline.get_data_from_tfds(config=config, mode='test')


	4.	Switch to TPU Runtime (if using Colab):
Change runtime type to TPU in Colab, re-mount Drive, and run the notebook again. The dataset will load from the pre-processed cache, avoiding TFDS preparation errors.
	5.	Run Fine-Tuning with KD:
The modified make_update_fn in train.py now accepts teacher_params and teacher_apply_fn. Just run:

# In vit_jax.ipynb
update_fn_repl = train.make_update_fn(
    apply_fn=model.apply,
    teacher_apply_fn=teacher_model.apply,
    teacher_params=teacher_params,
    alpha=0.27,      # Tune if needed
    temperature=1.0, # Tune if needed
    accum_steps=accum_steps,
    tx=tx
)

Then start the training loop. After training, evaluate the accuracy:

accuracy = get_accuracy(params_repl)
print("Test Accuracy:", accuracy)



Results & Notes
	•	Initial random accuracy ~10%.
	•	After fine-tuning with KD for ~100 steps, accuracy ~97%.
	•	Small variations (+/- 0.3%) are normal. Consider hyperparameter tuning, longer training, or different augmentations for further improvements.

Inference

We have a separate inference section that:
	•	Loads a pre-trained model (e.g., ViT-B_32 fine-tuned on ImageNet).
	•	Runs predictions on a sample image.
	•	Shows the top-10 predicted labels.

This step is optional, depending on your project requirements.

Happy experimenting!
