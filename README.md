# Hello World of Tiny Anesthesia

Welcome to our repository! Here, we showcase a proof of concept for running machine learning models on custom hardware, specifically designed for on-device inference of brain data. This project is part of a groundbreaking approach in the field of tiny machine learning (tinyML) and neuroscience, focusing on the classification of anesthesia levels in local field potentials (LFP).

For more information about TinyML and many more related projects check out the following resources: 
- [Harvard CS249r Course](https://scholar.harvard.edu/vijay-janapa-reddi/classes/cs249r-tinyml)
- [Machine Learning Systems Open-source book](https://harvard-edge.github.io/cs249r_book/)

## Overview

We have developed two distinct methodologies:

1. **Single Model Approach**: This utilizes one comprehensive model to differentiate between three anesthesia levels in LFP data.
2. **Dual Model Approach**: This method first classifies data into sleep vs. non-sleep states. If classified as non-sleep, it further classifies the specific state.

Both approaches are supported by Python scripts for training Keras models and C/C++ files for deploying them on TensorFlow Lite for Microcontrollers (TFLM) supported boards.

## Repository Contents

- **Python Scripts**: For training Keras models, converting them to TensorFlow Lite, and then to TensorFlow Lite Micro format.
- **C/C++ Code**: For deploying the models on hardware platforms supported by TFLM.
- **Custom Board Compatibility**: Tested on a board powered by the nRF52840 microcontroller.

Visit [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers#supported_platforms) for more information on supported platforms.

## Installation and Demonstration Data

The data and environment required to run the repository can be found in our associated arXiv paper: [Hopfield Deep NN for Brain State Artifact Preprocessing](https://arxiv.org/abs/2311.03421).

- To set up the environment, follow the instructions in the [GitHub repository](https://github.com/arnaumarin/Hopfield-Deep-NN-for-Brain-State-Artifact-Preprocessing) to install the Conda environment.
- Download the dataset as specified in the repository.

## How to Use

The project consists of two primary steps:

1. **Model Training and Conversion**:
    - Train the models using TensorFlow.
    - Convert them to TensorFlow Lite and subsequently to TensorFlow Lite Micro format.
2. **Deployment on Arduino**:
    - Deploy the converted models on Arduino using the provided C/C++ files.

Instructions for each step are provided within the respective directories, along with necessary scripts and deployment files.
