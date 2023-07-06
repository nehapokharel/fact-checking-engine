# RDF Graph Analysis for Fact Verification

This repository contains a Python script that utilizes Machine Learning (ML) techniques to verify the veracity of facts encoded as RDF triples. The project involves preprocessing RDF data, extracting features using the Node2Vec graph embedding method, training a eXtreme Gradient Boosting Classifier, and using the trained model to verify the truthfulness of the facts.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Running the Script](#running-the-script)

## Getting Started

### Prerequisites

To run the script, you need Python 3.6 or later. Additional requirements include the following Python packages:

- rdflib
- networkx
- scikit-learn
- numpy
- node2vec

These can be installed via pip:

```bash
pip install rdflib networkx scikit-learn numpy node2vec
```
or 

```bash
pip install -r requirements.txt
```

### File Structure

- `main.py` : The main script to run.
- `data_preprocessing.py` : Contains functions for parsing RDF data and converting it into a NetworkX graph. Contains functions for generating Node2Vec embeddings and feature vectors.
- `model.py` : Contains functions for training and evaluating the ML model.
- `output.py` : Contains functions for fact verification and result generation.

## Running the Script

To run the script, navigate to the directory containing `main.py` and use the following command:

```bash
python main.py
```
or 
```bash
python3 main.py
```

## Dataset
**Training data** have been used.
The statements in the training file have a veracity value assigned (0 or 1).

`fokgtrain.nt` is the data. If you install other dataset:
Please note, you need to provide the path to your input `.ttl` file in the `main.py`.

The output will be written to a file named `result.ttl` in the same directory.

