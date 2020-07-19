## Logistic

\begin{equation}
h_\theta(x) = g(\theta^Tx) = \frac{1}{1 + e^{-\theta^Tx}} \\
\end{equation}

\begin{equation}
Where,\ Sigmoid\ Function : g(x) = \frac{1}{1 + e^{-x}} \\
\end{equation}


The likelihood function is given by 
\begin{equation}
L(\theta) = p(\vec{y}\ |X;\theta)
\end{equation}

Here, Let us assume that

\begin{align}
P(y = 1\ |\ x;\ \theta) &= h_\theta(x) \\
P(y = 0\ |\ x;\ \theta) &= 1 - h_\theta(x) \\
\end{align}
and, $y\in\{0, 1\}$

Combining both equations,

\begin{equation}
p(y\ |\ x;\ \theta) = (h_\theta(x))^y\ (1 - h_\theta(x))^{1-y}
\end{equation}

Now, we can write the likelihood as:

\begin{align}
L(\theta) &= p(\vec{y}\ |X;\theta) \\
&= \prod_{i=1}^{N}\ p(y^{(i)} \ |x^{(i)};\theta) \\
&= \prod_{i=1}^{N}\ (h_\theta(x^{(i)}))^{y^{(i)}} \ (1 - h_\theta(x^{(i)}))^{1-y^{(i)}} \\
\end{align}
Where,<br>
N: The number of samples<br>
$p(y\ |\ x;\ \theta)$: Read as " Probability of y given x and parametrized by θ "

To make computation easy, we take the log of likelihood:

\begin{align}
\ell(\theta) &= log\ L(\theta) \\
&= \sum_{i=1}^{N} y^{(i)}\ log\ h_\theta(x^{(i)}) + (1-y^{(i)})\ log(1 - h_\theta(x^{(i)})) \\
\end{align}

Hence, we can use Gradient Ascent to find the maximum of the likelihood function for a particular value of θ.

Usually, Gradient Descent is used to find the optimum value of θ instead of Gradient Ascent. So, we consider Cross entropy:

\begin{equation}
H(p,q) = - \sum_{i} p_i\ log\ q_i
\end{equation}
Note that Cross entropy can also be used for multi-class problems.

Here,<br>
$p\in\{y, 1-y\}$: True probability<br>
$q\in\{h_\theta(x), 1-h_\theta(x)\}$: Predicted value

\begin{equation}
H(p, q) = - y\ log\ h_\theta(x) - (1-y)\ log(1 - h_\theta(x))
\end{equation}

For N samples, we use average of cross entropy.<br>
So, Cross entropy loss function:
\begin{align}
J(\theta) &= - \frac{1}{N} \log L(\theta) \\
&= - \frac{1}{N} \sum_{i=1}^{N} \left( y\ log\ h_\theta(x) + (1-y)\ log(1 - h_\theta(x)) \right) \\
\end{align}


Notice that maximizing the log-likelihood function is the same as minimizing the cross entropy loss.


Derivative of sigmoid:

\begin{align}
\frac{\partial g}{\partial z} &= \frac{d}{dz} \left(\frac{1}{1 + e^{-z}}\right) \\
&= \frac{e^{-z}}{(1 + e^{-z})^2} \\
&= \left(\frac{1}{1 + e^{-z}}\right) \cdot \left(1 -\frac{1}{1 + e^{-z}}\right) = g \cdot (1 - g) \\ 
\end{align}


Back Propagation:
\begin{align}
\frac{\partial}{\partial\theta_j}J(\theta) &= \frac{\partial J}{\partial g} \frac{\partial g}{\partial (\theta^Tx)} \frac{\partial (\theta^Tx)}{\partial \theta_j} \tag{Chain rule}\\
&= - \sum\left(y\frac{1}{g(\theta^Tx)} - (1-y)\frac{1}{(1 - g(\theta^Tx))}\right)\ \frac{\partial}{\partial\theta_j} g(\theta^Tx) \\
&= \sum \left((1-y)g(\theta^Tx) - y(1 - g(\theta^Tx))\right) x_j \\
&= \sum (h_\theta(x) - y)\ x_j \\
\end{align}

Above, we used the fact that $g'(x) = g(x)(1 - g(x))$.

Removing the summation term by converting it into a matrix form for the gradient with respect to weights and the bias term.

\begin{align}
\frac{\partial}{\partial W}J(\theta) &= \frac{1}{N} X^T[h_\theta(x) - y)]  \tag{w.r.t weights} \\ 
\frac{\partial}{\partial b}J(\theta) &= \frac{1}{N} [h_\theta(x) - y)] \tag{w.r.t bias} \\ 
\end{align}


Stochastic Gradient Decent:<br>
SGD takes one random instance from training sample for computing gradients.

The parameter θ is updated by adding the negative of the gradient of the loss function:
\begin{equation}
\theta = \theta - \alpha \frac{\partial J}{\partial\theta}
\end{equation}
Where,<br>
$\alpha$: Learning rate

Gradient desent step:
\begin{align}
W &= W - \alpha\frac{1}{N} X^T[h_\theta(x) - y)]  \\
b &= b - \alpha \frac{1}{N} [h_\theta(x) - y)] 
\end{align}
jdsksd

asd


\begin{align}
\frac{\partial J}{\partial g} &= \frac{1}{N} \frac{\partial (-y\ log\ g - (1-y)\ log(1 - g))}{\partial g}\\
&= \frac{1}{N} \left( \frac{1 - y}{1 - g} - \frac{y}{g} \right)\\
\end{align}


\begin{align}
J(\theta) &= - log\ L(\theta) \\
&= - \sum_{i=1}^{N} y^{(i)}\ log\ h_\theta(x^{(i)}) + (1-y^{(i)})\ log(1 - h_\theta(x^{(i)})) \\
\end{align}

\begin{align}
\frac{\partial z}{\partial W} &= - \frac{\partial (WX + b)}{\partial W} = X^T \\
\\
\frac{\partial z}{\partial b} &= - \frac{\partial (WX + b)}{\partial b} = 1 \\
\end{align}

asd

asd


\begin{align}
\frac{\partial J}{\partial W} &= \frac{1}{N} \frac{\partial J}{\partial g} \frac{\partial g}{\partial z} \frac{\partial z}{\partial W} \tag{Chain rule}\\
&= \frac{1}{N} \left(\frac{1 - y}{1 - g} - \frac{y}{g}\right) \left(g(1 - g)\right) X^T  \\
&= \frac{1}{N} (g - y)\ X^T  \\
&= \frac{1}{N} (h_\theta(X) - y)\ X^T  \\
\\
\frac{\partial J}{\partial b} &= \frac{1}{N} \frac{\partial J}{\partial g} \frac{\partial g}{\partial z} \frac{\partial z}{\partial b} \tag{Chain rule}\\
&= \frac{1}{N} \left(\frac{1 - y}{1 - g} - \frac{y}{g}\right) \left(g(1 - g)\right)\cdot 1 \\
&= \frac{1}{N} (g - y) \\
&= \frac{1}{N} (h_\theta(X) - y) \\
\end{align}


\begin{align}
W &= W - \alpha \frac{1}{N} (h_\theta(X) - y)\ X^T \tag{w.r.t weights} \\
b &= b - \alpha \frac{1}{N} (h_\theta(X) - y)  \tag{w.r.t bias} \\
\end{align}

\begin{align}
inp.g &= \frac{\partial (Sigmoid)}{\partial inp} * out.g \\
\end{align}