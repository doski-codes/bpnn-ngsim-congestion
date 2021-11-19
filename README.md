# Neural Network

## 1. Introduction

This is an implementation of Road Traffic Congestion prediction using a back propagation Neural Network.

The neural network is built using Keras with a Tensorflow backend.

The **_US101 Highway data_ from the  Next Generation Simulation dataset (NGSIM)** was used to build the model.

## 2. Environment

To install the packages used in this project and you would require `conda`.

After cloning the repository run the following commands:


`cd bpnn-ngsim-congestion`

then

`conda env create -f environment.yml`

Now run the command `conda activate deeplearning_env` to activate the conda environment with the required packages.

You can run `conda deactivate` to deactivate the environment.

## 3. Dataset

The fact sheet for the US101 Next Generation Simulation dataset (NGSIM) can be found [here](https://www.fhwa.dot.gov/publications/research/operations/07030/07030.pdf).

You can download the US101 metadata [here](https://data.transportation.gov/api/views/8ect-6jqj/files/ddb2c29d-2ef4-4b67-94ea-b55169229bd9?download=true&filename=1-%20US%20101%20Metadata%20Documentation.pdf).

The full dataset can also be located [here](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm).

The congestion label is calculated using the slow moving cars that are close to each other (i.e. vehicles with velocity and space headway below average are represented as in a congested environment - 1, and not congested otherwise - 0).

The congestion label is also calculated by clustering the data to group similar rows.

Other calculations for congestion label to be considered:

- Using the total frames to determine congestion. The logic with this is that vehicles that move faster through the camera would be recorded in fewer frames hence no congestion, but vehicles in congested zones would record higher total frames since they spend more time in the eye of the cameras.

- Calculating congestion using the above methods for each vehicle class to get results that are more specific to the vehicle types.

## 4. Model

- Logistic Regression
This model was built with no fine tuning

- Neural Network (Multi Layer Perceptron)
This model was built with 3 layers; the input layer with 5 nodes, 1 hidden layer with 9 nodes and the output layer with 1 node.

## 5. Summary

This model is still subject to hyperparameter tuning and various data transformations to improve it's performance.

The model in this state can be used at congestion prone areas to predict when congestion would occur and to also test the effectiveness of the model.
