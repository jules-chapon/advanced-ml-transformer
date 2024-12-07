# Advanced Machine Learning - Transformer and Differential Transformer

*Owners :* 
- Charlotte de Romémont - charlotte.romemont@gmail.com
- Jules CHAPON - jules.b.chapon@gmail.com
- Jérémie Darracq - jeremie.darracq@ensae.fr

*Version :* 1.0.0

## Description

Description.

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

### Update dependencies

You can add/delete/update packages directly in the *pyproject.toml* file.
Once done, you must update the *poetry.lock* file.
To do so, run the following command in your terminal (make sure poetry is installed) :

```bash

poetry lock

```

Now, you can install the new packages.
To do so, you can refer to the previous part of the documentation.

### Run the pipeline

This is based on the work done by Balthazar Neveu in this repo : https://github.com/balthazarneveu/mva_pepites.

#### Local

To run the full pipeline of experiments 0, 1 and 2 on your own computer, with data loaded from your computer, you can run the following command in your terminal :

```bash

python -m src.model.train -e 0 1 2 --local_data --full

```

If you want to load data from other sources (Hugging Face, Kaggle...), you can run the previous command without the ``` --local_data ``` argument :

```bash

python -m src.model.train -e 0 1 2 --full

```

If needed, you can run only the learning or the testing parts of the pipeline by changing the ``` --full ``` argument and running one of the following commands :

```bash

python -m src.model.train -e 0 1 2 --learning

```

```bash

python -m src.model.train -e 0 1 2 --testing

```

#### Kaggle

To run the pipeline on a Kaggle Notebook, you first need to create a file *__kaggle_login.py* in the folder **remote_training**. This file must contain the following code :

```bash

kaggle_users = {
    "user1": {
        "username": <your_username>,
        "key": <your_kaggle_key>}
}

```

Once done, you need to run the following command in your terminal :

```bash

python -m remote_training.remote_training -u user1 -e 0 1 2 -b branch_name --full -p

```

If you want to execute the code present in your *main* branch, you can forget the ``` -b ``` argument. If you want to force the use of CPU, you can add the ``` --cpu ``` argument. This will give you the following command :

```bash

python -m remote_training.remote_training -u user1 -e 0 1 2 --cpu --full -p

```

If you want to execute specific parts of the code (learning, testing...), this works as mentionned above.

If you want to download the output, you can do it via the Kaggle interface or by modifying the ``` -p ``` argument and running the following command in your terminal :

```bash

python -m remote_training.remote_training -u user1 -d

```

Sometimes, you can obtain errors while trying to execute the code in a Kaggle notebook. You might delete the notebook on Kaggle website and try again.


## Project Documentation

Explain process and code.
