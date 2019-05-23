# Interpretable Neural Network
Trying to find more interpretable neural network topologies 

## First approach
Let's assume we want to predict the survival chances of a passenger using the classic titanic dataset. We have numeric variables, like age or fare, and categorical ones, like class or gender. The plan is to train N linear models on the numerical variables (for example, connecting a dense layer with N nodes directly to the numeric input values), and try to force the model to choose one those to make the final prediction. This will be done by doing a scalar product on the dense layer activators and the output of an embedding layer, trained  all on the possible categorical combinations. We use a unique regularization function to force the embedding layer activators into a unit vector, if possible. So we assign the categorical variable combinations (including even engineered categorical variables) into N groups,  to decide what linear model should be used. Since we try to lower N as much as possible, this approach won't always provide us one model to choose from, but probably a combination of several ones. This is okay, as far as the larger groups focus on one model only. 

This method could not only provide us several, easily interpretable linear models, but also categories the input by the categorical variables. 

We can also read the result in an excel format

![Alt text](.jpg?raw=true "")
