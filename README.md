# Interpretable Neural Network
Trying to find more interpretable neural network topologies 

## First approach
Let's assume we want to predict the survival chances of an individual using the classic titanic dataset. We have numeric variables, like age or fare, and categorical ones, like class or gender. The plan is to train N linear models on the numerical variables (for example, connecting a dense layer with N nodes directly to the numeric input values), and try to force the model to choose one those to make the final prediction. The   
