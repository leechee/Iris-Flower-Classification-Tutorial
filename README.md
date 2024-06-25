# Iris-Flower-Classification

![[Iris Flowers]|100](assets/iris_flowers.png)

The Iris flower data set or Fisher's Iris data set is a multivariate data set that contains 50 samples from each of three species of Iris (Iris Setosa, Iris Virginica, and Iris Versicolor).

The [dataset](IRIS.csv) contains a set of 150 records under 5 attributes:
- Petal Length, Petal Width, Sepal Length, Sepal width, and Species

## Objective

Many coders jump straight into using machine learning frameworks (PyTorch, TensorFlow, Keras, Theano) without understanding what's happening "under the hood". Nowadays, it's hard to even find the infamous Iris Classification solved without using such tools.

In this project, I **hand-coded** a deep learning model using a neural network to identify each species of Iris. The network is a multilayer perceptron with one hidden layer (two neurons), coded with Python, and 100% accuracy.

### I created a Jupyter Notebook tutorial detailing the project step-by-step [**here**](iris_classification.ipynb)
Click [here](iris_classification_NN.py) for just the source code.

## Getting started
### Python Environment
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

Now install the necessary AI stack in the venv terminal. These libraries will aid with computational coding, data visualization, accuracy reports, preprocessing, etc.
```
pip install numPy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install seaborn
```
Note: sklearn was **not** used to create the model or for supervised learning.
For more details on how I used libraries, visit the Jupyter Notebook I made [here.](iris_classification.ipynb)

### Data Input
To input data from the Iris dataset, use the pandas library:
```
data = pd.read_csv('IRIS.csv')
```

### Data visualization
In the [Jupyter Notebook](iris_classification.ipynb), there are more plots/graphs. Here are just some important ones for data visualization:

Using [seaborn](https://seaborn.pydata.org) and [matplotlib.pyplot](https://matplotlib.org), create a pairplot. This graph visualizes the relationship between each pair of variables in the Iris dataset. Since there are four inputs, you will notice the plot becomes a 4x4 with 16 plots in total.
```
sns.pairplot(data, hue= 'species', height= 3)
plt.show()
```
![iris_pairplot](assets/pairplot.png)

A boxplot is useful for understanding the distrubution of data:
```

```



