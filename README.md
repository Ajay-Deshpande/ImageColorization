# ImageColorization

**Image Colorization (Implementation of Pix2Pix Paper)**

---
**Summary:**

This project implements the Pix2Pix model for image colorization, leveraging the powerful architecture of U-Net with encoder and decoder blocks. Pix2Pix is a generative adversarial network (GAN) architecture designed for image-to-image translation tasks. In this implementation, grayscale images are provided as input, and the model learns to predict corresponding colorized images.

---
![alt text](https://github.com/Ajay-Deshpande/ImageColorization/blob/master/outputs/colorization_after_epoch_90.png)
---
**Files:**

1. **data_downloader.py:** This script downloads the COCO 2017 dataset for training the model. It saves the dataset into the specified directory.

2. **train.py:** This script contains the training pipeline for the Pix2Pix model. It loads the dataset, defines the model architecture, trains the model, and saves the trained model weights at every epoch.

3. **requirements.txt:** This file lists the required Python packages and their versions for running the project. Install these dependencies using 
    ```
   pip install -r requirements.txt
    ```

5. **utils/data_loaders.py:** This file defines a custom dataset class (`ColorizationDataset`) for loading and preprocessing images for colorization.

6. **utils/nn_helpers.py:** This file contains custom PyTorch modules and loss functions used for building and training neural networks, including encoder and decoder blocks, a U-Net generator, a patch discriminator, and a GAN loss function.

7. **models/model_architecture.py:** This file defines the architecture for the image colorization model. It includes initialization functions, the `ImageColorizationModel` class representing the Pix2Pix model, and methods for training and optimization.

---

**Usage:**

1. **Setup:**
   - Clone the repository to your local machine:
     ```
     git clone [repository_url]
     ```
   - Install the required dependencies listed in the `requirements.txt` file:
     ```
     pip install -r requirements.txt
     ```

2. **Data Preparation:**
   - Run `data_downloader.py` to download and prepare the training dataset.

3. **Training:**
   - Customize training parameters in `train.py` if needed.
   - Run `train.py` to train the Pix2Pix model on the prepared dataset.

4. **Evaluation:**
   - Evaluate the trained model using validation data to assess its performance.

5. **Deployment:**
   - Deploy the trained model for colorization tasks in real-world scenarios.

---
Note: The training takes 20 minutes per epoch on a Nvidia 1650Ti GPU

**References:**
- https://arxiv.org/abs/1611.07004
- https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
---

**Note:** This project is an implementation of the Pix2Pix paper by Phillip Isola et al. It follows the architecture and principles outlined in the paper to achieve image colorization through conditional adversarial learning.
