# Advanced Machine Learning - Transformer and Differential Transformer

*Owners :* 
- Charlotte de Romémont - charlotte.romemont@gmail.com
- Jules CHAPON - jules.b.chapon@gmail.com
- Jérémie Darracq - jeremie.darracq@ensae.fr

*Version :* 1.0.0

## Description

This repository contains the code for our project in **Advanced machine Learning** during our final Master's year at ENSAE Paris.

We developped both a Transformer and a Differential Transformer, and compared them on a translation task.

## How to use this repo

### Install dependencies

To install all dependencies, you can do as follow :

#### Using Poetry (easier)

- Create an environment with Python 3.10 and install Poetry>1.7 :

```bash

conda create -n my_env python=3.10
conda activate my_env
pip install poetry>1.7

```

Then, you can install all dependencies with the following command :

```bash

poetry install

```

#### Using uv (faster)

- Create an environment with Python 3.10 and install uv 0.4.29 :

```bash

conda create -n my_env python=3.10
conda activate my_env
pip install uv==0.4.29

```

Then, you can install all dependencies with the following command :

```bash

uv pip install -r pyproject.toml

```


### Run the pipeline in local


To run the full pipeline on your own computer, you can run the following command in your terminal :

```bash

python -m src.model.train -e 0 100 -i 0 0 --samples --full

```

Do not forget the ```--samples``` part as it loads only 1,000 samples in the training set (rather than 1,000,000 which is really heavy for a laptop). This command will train both models (they have the same parameters as the ones we trained) for 2 epochs, and the results will be available in the **output** folder.

## Project Documentation

The folder **src** contains all important functions (config, visualization, models, pipeline...).

The folder **remote_training** contains all functions that we used to launch the training on the cloud (Kaggle).

The folder **output** contains the results of the pipelines (training and testing).

The folder **analysis/notebooks** contains notebooks that you can use to check results. If you ran the pipeline on your computer with models 0 and 100, you can find two notebooks that correspond to them. The other ones correspond to our trained models that you can download directly from our Google Drive.

Hope you enjoy !
