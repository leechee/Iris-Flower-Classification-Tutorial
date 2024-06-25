# Iris-Flower-Classification

![Iris Flowers](assets/iris_flowers.png)

The Iris flower data set or Fisher's Iris data set is a multivariate data set that contains 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). In this project, I manually coded a deep learning model "under the hood" using a neural network to identify each species. The network is a multilayer perceptron with one hidden layer (model uses no machine learning frameworks).

Included is a Jupyter Notebook that details the project step by step --> [here.](iris_classification.ipynb)

The dataset contains a set of 150 records under 5 attributes:
- Petal Length
- Petal Width
- Sepal Length
- Sepal width
- Class(Species)

## Getting started
#### Python Environment
Download and install Python 3.8 or higher from the [official Python website](https://www.python.org/downloads/)

Optional, but I would recommend creating a venv. For Windows installation:
```
py -m venv .venv
.venv\Scripts\activate
```
For Unix/macOS:
```
python3 -m venv .venv
source .venv/bin/activate
```

Next, install the AI stack needed:
```
pip install numPy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install seaborn
```



