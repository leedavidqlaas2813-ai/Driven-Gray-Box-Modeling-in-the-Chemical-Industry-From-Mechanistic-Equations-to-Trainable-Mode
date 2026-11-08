# HybridML - Open Source Platform for Hybrid Modeling

- [HybridML - Open Source Platform for Hybrid Modeling](#HybridML---Open-Source-Platform-for-Hybrid-Modeling)
  - [Introduction](#introduction)
  - [Technologies](#technologies)
  - [Examples](#examples)
  - [Project Structure](#project-structure)
  - [Software Architecture](#software-architecture)
    - [Creation](#creation)
      - [1. Parsing](#1-parsing)
      - [2. Building](#2-building)
    - [NodeBuilder, NodeParser and Node](#nodebuilder-nodeparser-and-node)
  - [Constructing Models](#constructing-models)
  - [Implementing new node types](#implementing-new-node-types)
  - [Citation](#citation)
## Introduction
This tool aims to create hybrid machine learning models using tensorflow and keras.
In this context hybrid means, that the models consist of submodels that are connected in a feed forward graph.
These submodels, called nodes are machine learning models themselves.
We currently support neural networks, mechanistic models (formulas) and linear odes as nodes.
It is possible to add more types of models.
More can be found under "How to to implement new node types".


## Technologies
* tensorflow
* casadi
* numpy
* matlotlib
* tqdm

The original HybridML implementation uses legacy TensorFlow and CasADi
dependencies. The tutorials use a separate environment; see
[tutorials/index.md](tutorials/index.md) before installing dependencies.

## Examples
### Chemical Engineering Grey-Box Modeling Book
The Chinese [Jupyter Book tutorial](tutorials/index.md) introduces grey-box modeling with a modern Keras CSTR example and guided tours of the repository's existing HybridML examples.

Available model description and data examples:

- [Simple model JSON](demo/simple_model.json), used in the repository tour.
- [Theophylline model JSON](demo/theoph/projects/theoph_demo/theoph_model_2_compartments.json)
  and its [normalized dataset](demo/theoph/projects/theoph_demo/theoph_normalized.csv),
  also used in the repository tour.


## Project Structure
HybridML can also be used with the project architecture.
A project contains a name, one or multiple models and one or multiple datasources.
A new model can be created by using:
```python
import Project
project = Project.create(containing_dir, "project_name")
```
Then a file of the name `project_name.json` is created in the containing directory.
There, new models and datasources can be added.
To open an already existing model the method  `Project.open(project_file_path)` can be used, where `project_file_path` is the path to the formerly created `project_name.json`.
The project class has the ability to create and load all of its models at once, using the methods `generate_models()` and `load_models()`.
Note, that it is not necessary to state the location of the model description or the saved model file, as both are handled by the project instance.
All models can be saved, using the `save()` method.
The data sources can be loaded, using `load_data_sources()`, where we again don't have to state the location of the data sources, because they are handled by the project instance.
The data sources can then be accessed via the `data_sources` propoerty, which is a list of data_sources.


## Software Architecture

The model creation process consists of the two phases parsing and building.
In the parsing phase, the model description file is read and loaded into memory.
In the building phase, the parsed model is then built.
The two phases are independant from each other.

### Creation
The model creation process is coordinated by the `ModelCreator`.
It has a `NodeRegistry`, which contains parsers and builders for all possible types of nodes, as well as all custom Tensorflow objects.
Furthermore, it has a `ModelParser`, which recieves the model-json.
The parsed output is then given to its `ModelBuilder`, which performs the final step of building the tensorflow model.

#### 1. Parsing
At first the simple parameters of the model, such as `name`, `mse`, `loss`, etc are parsed.
Then the network structure of the model is parsed by an instance of `NetParser`.
The parsing itself consists of two passes.
In the first pass, all nodes of the network are parsed by `NodeParsers`.
The resulting `ParsedNode`s are connected with `NodeConnors`, which are represented by variable names in the model-json.s
In the second pass the parser walks through the network and propagates the input and output sizes of each node thorugh the `ParsedNodes` and `NodeConnectors`.

#### 2. Building
In the first step, a `NetworkBuilder` is used to build the network.
The tensorflow/keras functional api is used to construct the network.
At first, `Input`s are built, which represent the `NodeConnectors`, labelled as inputs in the model description.
Then the builder walks throught the network, building the nodes using `NodeBuilders`, and connecting them to their input tensors, and collecting all input and output tensors in a dictionary.

This parsed network is then used to build a `model` and an `additional_outputs_model`.


### NodeBuilder, NodeParser and Node
For each type of node there is an implementation of each of the three classes `NodeBuilder`, `NodeParser`, `Node`.
The parsers and builders select the correct implementation during the process.
More on that in [How to implement new node types](create-nodes.md).


## Constructing Models
See [model-json.md](model-json.md).

## Implementing new node types
See [create-nodes.md](create-nodes.md).

# Citation
Our paper can be retrieved at https://doi.org/10.1016/j.compchemeng.2022.107736.
If you use HybridML in your work, please cite our paper using the following citation.
```bibtex
@article{MERKELBACH2022107736,
title = {HybridML: Open source platform for hybrid modeling},
journal = {Computers & Chemical Engineering},
volume = {160},
pages = {107736},
year = {2022},
issn = {0098-1354},
doi = {https://doi.org/10.1016/j.compchemeng.2022.107736},
url = {https://www.sciencedirect.com/science/article/pii/S0098135422000771},
author = {Kilian Merkelbach and Artur M. Schweidtmann and Younes Müller and Patrick Schwoebel and Adel Mhamdi and Alexander Mitsos and Andreas Schuppert and Thomas Mrziglod and Sebastian Schneckener},
keywords = {Hybrid modeling, Machine learning, Modeling tools, Tensorflow, Python, Pharmacokinetics},
abstract = {Hybrid modelling, i.e., the combination of data-driven modelling with mechanistic model components, reduces the data demand and enables extrapolation of data-driven models. However, building, training and evaluation of hybrid models is cumbersome with current frameworks. We developed HybridML, an open-source modeling platform, in which hybrid models can be trained, i.e., combinations of artificial neural networks, arithmetic expressions, and differential equations. We employ TensorFlow for artificial neural network training and Casadi to integrate ordinary differential equations and provide gradients of differential model equations enabling continuous time representations. HybridML provides also a JSON interface for the model development. We apply HybridML to an industrial case study, in which the trained model is used to predict drug concentrations over time, based on physiological information about the patients. To demonstrate its versatility, we also present a nonlinear application, where HybridML is used to model the spread of the COVID-19 pandemic in German federal states based on the state’s socio-economic attributes.}
}
```
