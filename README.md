# PopulationPredictionLSTM

This program is designed as a population predicting tool. The objective is to predict the net migration of a given state in the US for the next year, given that we know the net migration for the current year. The neural network used to implement is the LSTM.


# Input
1.Each state has a size 51 vector for each year. 
2.Data available for every state from 1993 – 2009
3.Taken year by year

# Neural Netowrk Architecture
1. Inputs contain 51 parameters.
2. Outputs contain 51 parameters.
3. Single hidden layer with LSTM cell.
4. LSTM cell state maintained across iterations and epochs.
5. Like RNNs, hidden layer contains an extra cell (LSTM in this case).
6. Unlike RNNs, it is not simply added to the hidden layer output, instead it performs a series of functional transformations of the output from the hidden layer.

# Learning paradigm
Numerical unconstrained optimization method was the stochastic gradient descent method. It proved to be effective for earlier epochs.

# References

The gradients and backprop equations were obtained from the following blog: http://blog.varunajayasiri.com/numpy_lstm.html
The application was modified frmo the word prediction in the example, and also the numerical optimization method was chagned.

The core concept of LSTM used were obtained from http://colah.github.io/posts/2015-08-Understanding-LSTMs/

This blog again refers to the original paper that defined the architecture of the LSTM cells: http://www.bioinf.jku.at/publications/older/2604.pdf
