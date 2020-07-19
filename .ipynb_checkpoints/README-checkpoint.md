# Logistic Regression from Scratch

The logistic regression is used when the dependent variable(target) is categorical. The model uses the logistic function (Sigmoid) to squeeze the output of a linear equation between 0 and 1, which can then be mapped to two or more discrete classes. Every real value can be mapped to a value between 0 or 1, which signifies the probability for belonging to a class.

Gradient descent algorithm is used for finding a minimum of a differentiable function by taking steps proportional to the negative of the gradient of the function at the current point. We use mini-Batch Gradient Decent, which takes a mini-batch(eg, 64) random instances from training sample for computing gradients.




### Data
Kaggle: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

### Reference
* https://course.fast.ai/videos/?lesson=8
* https://course.fast.ai/videos/?lesson=9
* https://youtu.be/het9HFqo1TQ
* Xavier Glorot, Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks,In: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, PMLR 9:249-256, 2010.





# Logistic Regression using Pytorch

Logistic regression is basically a neutral network consisting of
   * A linear layer followed by a non-linear Layer called Sigmoid function(or Logit function).
   * Loss function - binary cross entropy loss
   * Optimizer - Stochastic Gradient Descent(SGD)
    
### Data
kaggle: https://www.kaggle.com/c/santander-customer-transaction-prediction/data