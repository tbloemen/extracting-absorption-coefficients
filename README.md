# Extracting absorption coefficients from a room impulse response using a convolutional neural network with domain adaptation

This codebase has been developed for the Research Project of the Bachelor Computer Science of the TU Delft. The purpose of this project is to estimate absorption coefficients from a room impulse response using domain adaptation.

## Installation

To install this codebase, clone it to your local machine. For storage reasons, the dEchorate dataset is not included in this repository. Instead, the dataset should be downloaded from [their repository](https://github.com/Chutlhu/dEchorate). It can then be added to this file structure, making the folder structure look like this:
```bash
.
├── constants.py
├── data_gen.py
├── dEchorate
│   ├── dEchorate_annotations.h5
│   ├── dEchorate_database.csv
│   ├── dEchorate_rir.h5
│   └── Load dataset.ipynb
├── estimator.py
├── InHouse
│   ├── bigger
│   │   ├── Closed Curtains
│   │   │   ├── RIR_1.mat
│   │   │   ├── RIR_3.mat
│   │   │   ├── RIR_4.mat
│   │   │   ├── RIR_5.mat
│   │   │   ├── RIR_6.mat
│   │   │   └── RIR_7.mat
│   │   ├── main.m
│   │   ├── Open Curtains
│   │   │   └── RIR_3.mat
│   └── smaller
│       ├── curtains_closed_real_room_3.wav
│       ├── curtains_open_real_room_3.wav
│       └── main2.m
├── inhouse.py
├── loss.py
├── main.py
├── models
│   ├── my_model.pth
├── Pipfile
├── Pipfile.lock
├── preprocessing.py
├── printing.py
├── README.md
├── tol_colors.py
└── utils.py
```

Dependencies are managed with Pipenv. Firstly, make sure pipenv is installed: 
```bash
pip install pipenv
```
Then install all packages:
```bash
pipenv install
```
And run a file of your choosing by prepending `pipenv run` before a command. This ensures your command is run within the virtual environment created by pipenv. Alternatively, you can activate the virtual environment in your shell by running `pipenv shell`. Then, a python command can be run: 
```bash
pipenv run python main.py
```
Or:
```bash
pipenv shell
python main.py
```

## Usage
To train and test this model, run the `main` function in the `main.py` file. To only look at the results of your trained model, run the `get_results` function in the same file. To validate the model, run the `inhouse.py` file. Various figures can be printed in the `printing.py` file. To alter hyperparameters for the project, tweak the values in the `constants.py` file before training the model.