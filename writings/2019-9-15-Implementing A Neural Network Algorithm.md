---
title: "Implementing A Neural Network Algorithm"
categories:
  - neural networks
tags:
  - neural networks
  - algorithm implementation
  - python
--- 

In this post, we will implement a simple two-layer neural network from scratch to explain how neural
networks work. We consider the 
[Predicting Short Term Trucking Rates problem](https://emrahcimren.github.io/transportation%20rates/Predicting-Short-Term-Trucking-Rates-with-Random-Forests/)
and develop a neural network model to predict transportation rates using the the European long-haul Truckload data.

![image-center](/images/2019-9-15_neurons.jpg){: .align-center}

## What is a Neural Network?

Neural networks use an architecture inspired by the neurons in the brain. 
A brain neuron receives an input and based on that input and fires off an output that is used by another neuron. 
Neurons interconnected to other neurons forms a network. 
The neural network simulates this behavior in learning about collected data and then predicting outcomes.

Neural Networks consist of the following components:

- An input layer, 
- Hidden layers, 
- An output layer,, 
- A set of weights and biases between each layer,
- Activation function for each hidden layer.

The number of nodes in the input layer is determined by 
the dimensionality of our data, i.e.
features we use in prediction. 
The dimensionality (the number of nodes) of the hidden layer are given. 
The number of nodes in the output layer is determined by the number of prediction classes.

## Algorithm

The following are the steps to build the model:

1. Define the network structure, i.e. number of input features.

2. Initialize parameters.
 - Number of iterations
 - Number of layers in the neural network
 - Size of the hidden layers
 - Learning rate
 
3. For each iteration:
 - Forward propagation (calculate current loss)
 - Compute cost function
 - Backward propagation (calculate current gradient)
 - Update parameters (using parameters, and gradients from backpropagation)
 
4. Use trained parameters to predict labels.

## Problem

We consider the [Predicting Short Term Trucking Rates problem](https://emrahcimren.github.io/transportation%20rates/Predicting-Short-Term-Trucking-Rates-with-Random-Forests/)
and develop a neural network model to predict transportation rates using the the European long-haul Truckload data.

| ![_config.yml]({{ site.baseurl }}/images/trans_rate_random_forest_customer_warehouse_network.jpg) | 
|:--:| 
| *Figure 1: Distribution centers and customers* |

Our objective is to determine FTL and LTL rates for each distribution center to each customer. We develop
a model to predict transportation rates. 

The dataset consists of the European long-haul Truckload data, including
mode, distance covered per shipment, average shipment weight per truck, average shipment cost per truck,
and trailer type (see Figure 2). 

| ![_config.yml]({{ site.baseurl }}/images/trans_rate_random_forest_input_data.jpg) | 
|:--:| 
| *Figure 2: Transportation data set* |

** ADD NETWORK DIAGRAM HERE

We now build the neural network model by following algorithm steps.

## 1. Define Network Structure

Let $$L$$ be the set of layers. Let $$n^{\ell}$$ be the number of elements at layer $$\ell \in L$$. 
Let $$X$$ be the set of input layer where $$X=\{x_1, \dots, x_n^0\}$$. Let $$A^{\ell}$$ be set of 
non-input layer $$\ell \in L$$ elements where $$A^{\ell}=\{a^{\ell}_1,\dots, a^{\ell}_{n^{\ell}}\}$$.
Note that $$A^{\ell}$$ is used to define hidden and output layer elements. 

Let $$W^{\ell}$$ and $$b^{\ell}$$ be the set of weights and biases between layer $$\ell-1$$ and 
$$\ell$$. Let $$W$$ and $$b$$ denote weights and biases in general, respectively.

A neuron computes a linear function 
\begin{equation}
z = w_1x+1 + \dots + w_nx_n 
\end{equation} 

\begin{equation}
Z = WX + b
\end{equation} 

An output of a
neuron $$a=g(z)$$ where $$g$$ is the activation function.
 
We now define structure of each layer $$\ell \in \{0, 1, \dots,|L|\}$$ as follows. 
Let $m$ be the size of the training set used for modeling.

| Layer | Activation Function | Output | 
|-------|-------|-------|
| 1 | $$Z^1_{n^1 \times m} = W^1_{n^1\times n^0} X_{n^0\times m} + b^1_{n^1\times 1}$$ |  $$A^1_{n^1\times m} = g^1(Z^1_{n^1 \times m})$$ |
| 2 | $$Z^1_{n^2 \times m} = W^2_{n^2\times n^1} A^1_{n^1\times m} + b^2_{n^2\times 1}$$ |  $$A^2_{n^2\times m} = g^2(Z^2_{n^2 \times m})$$ |
| $$\dots$$ | $$\dots$$ | $$\dots$$ |
| L-1 | $$Z^{L-1}_{n^{L-1} \times m} = W^{L-1}_{n^{L-1}\times n^{L-2}} A^{L-2}_{n^{L-2}\times m} + b^{L-1}_{n^{L-1}\times 1}$$ | $$A^{L-1}_{n^{L-1}\times m} = g^{L-1}(Z^{L-1}_{n^{L-1} \times m})$$ |
| L | $$Z^{L}_{n^{L} \times m} = W^{L}_{n^{L}\times n^{L-1}} A^{L-1}_{n^{L-1}\times m} + b^{L}_{n^{L}\times 1}$$ | $$A^{L}_{n^{L}\times m} = g^{L}(Z^{L}_{n^{L} \times m})$$ |

## 2. Initialize Parameters 

For the transportation rates prediction problem, $$L=\{0, 1, 2\}$$, $$n^0=4$$ from features distance miles,
lbs per truck,dry van indicator, and temperature controlled truck  indicator,
$$n^1 = 5$$, and $$n^2=1$$. We define the structure of the network as follows.

| Layer | Activation Function | Output | 
|-------|-------|-------|
| 1 | $$Z^1_{5 \times m} = W^1_{5\times 4} X_{4\times m} + b^1_{5\times 1}$$ |  $$A^1_{5\times m} = g^1(Z^1_{5 \times m})$$ |
| 2 | $$Z^1_{1 \times m} = W^2_{1\times 5} A^1_{5\times m} + b^2_{1\times 1}$$ |  $$A^2_{1\times m} = g^2(Z^2_{1 \times m})$$ |

Common choices for activation functions are tanh, the sigmoid function, or ReLUs. 
We use ReLUs function for $$g^1$$ and $$g^2$$ as follows.

$$
g^1(Z^1_{5 \times m}) = \max(0, Z^1_{5 \times m})
$$

\begin{equation}
g^2(Z^2_{1 \times m}) = \max(0, Z^2_{1 \times m})
\end{equation} 

## 3. Iterations

The process of calculating $$W$$ and $$b$$ from the input data is known as training the Neural Network. 
Each iteration of the training process consists of the following steps.

### Forward Propagation  
Given input from the previous layer, we first compute $$Z$$ and then apply selected activation function
to calculate the predicted output $$A$$. 

### Loss Function
The Loss Function evaluates the goodness of predictions. 
Our goal in training is to find the best set of $$W$$ and $$b$$ that minimizes the loss function.

Let $$Y = \{y_0, \dots, y_m\} $$ be the observations from the train data. We define the loss function as follows.

$$
J(A^L, Y) = -\frac{1}{m}\sum_{i=1}^my_i log(a^{\ell}_{n^{\ell}i})+(1-y_i)log(1-a^{\ell}_{n^{\ell}i})
$$

### Backward Propagation
Backward propagation is used to calculate the gradient of the loss function with respect to the parameters.
We use the chain rule to calculate derivatives. 

Let $$h(x)$$ be a function defined as 

$$
h(x) = f_1(f_2(f_3(x)))
$$

where $$f_1(x)$$, $$f_2(x)$$, and $$f_3(x)$$ are functions.

$$
\frac{\partial f}{\partial x} = \frac{\partial f_1}{\partial f_2} \frac{\partial f_2}{\partial f_3} \frac{\partial f_3}{\partial x}
$$




- **Backward propagation**: Updating $$W$$ and $$b$$.




## 4. Predict Labels


## 5. Application 


