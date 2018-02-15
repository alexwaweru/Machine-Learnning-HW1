#Probability Distridution and Linear Regression#
This problem set requires you to provide written answers to a few technical questions. In
addition, you will need to implement some learning algorithms in Python and apply them to the
enclosed data sets.

##Problem Three##
Let’s flip a coin. Let c ∈ {0, 1} be a random variable indicating the result of the
flip (1 for heads, 0 for tails). The probability that the coin lands heads on any trial is given
by a parameter μ. Note that we can write the distribution of c as: P (c ; μ) = μ c (1 − μ) 1−c .
We flip the coin m times, and denote the result of the i-th flip by variable c (i) . We assume
that the coin flips are independent. We observe heads H times.
* Write the likelihood function, i.e., the probability of the data D = {c (1) , ..., c (m) } given the model described above. Keep in mind that the likelihood is the probability of the observed training set D (rather than all possible training sets containing H heads in m examples). Thus your likelihood function should not include a combinatorial term counting the number of different ways H heads could occur in m trials.
* Derive the parameter μ using Maximum Likelihood estimation (hint: maximize the log likelihood).
* Code the Python function q3 likelihood.py to compute the likelihood given scalar input arguments H, m and an input vector μ (the function should return a vector of likelihood values as big as μ). You can now inspect how the likelihood function differs for the three cases {m = 1, H = 1}, {m = 100, H = 100}, and {m = 100, H = 80} by executing the script q3c.py, which calls your function q3 likelihood.py. To execute the script type ”run q3c” in your Python shell.

Let us now assume that we have good reasons to believe that the coin is not counterfeit. However, we are not certain. We model this prior knowledge in the form of a prior distribution over μ:
$p(μ; a) = \frac{1}{z} μ^{a−1} (1 − μ)^{a−1}$
where a is a parameter governing the prior distribution and Z is a normalization constant (so that $\int_{0}^{1} p(μ; a)dμ = 1$).
* Implement the prior function by coding the file q3 prior.py. Then, execute the script q3d.py, which plots the prior for a = 2 (Z = 1/6), and a = 10 (Z = 1/923780).
* Assuming the prior p(μ; a) in Eq. 1, derive the analytical expression of parameter μ using Maximum A Posteriori (MAP) estimation (hint: maximize the log posterior and drop the term related to the evidence, i.e., the denominator in Bayes’ rule).
* By looking at the MAP estimate derived in the previous point, can you provide an interpretation of parameter a in terms of training examples?
* Implement the posterior by coding the file q3 posterior.py, and plot the posterior for the same coin flipping results considered above (i.e., {m = 1, H = 1}, {m = 100, H = 100}, and {m = 100, H = 80}), by running the script q3g.py. Do not include the evidence term (i.e., the denominator) in the posterior calculation. There is no way to estimate the evidence and in any case it is just a constant. Make sure to plot the posterior, not the log posterior.

##Problem Four##
Write Python code implementing a regression algorithm for multi-dimensional inputs x and 1D outputs y. Your software must learn the regression hypothesis by minimizing the regularized least-square objective where θ = [θ 0 , θ 1 , ..., θ d ] ⊤ and b(x) is a vector that encodes either of the following two distinct choices of features:
- b l (x) = [1, x 1 , ..., x d ] ⊤ (where, d is the number of input entries in vector x and x j denotes the j-th element of vector x), or
- b q (x) = [1, x 1 , ..., x d , x 21 , x 1 x 2 , ..., x 1 x d , x 22 , x 2 x 3 , ..., x 2 x d , ...., x 2 d ] ⊤ (i.e., a quadratic function of the elements of x).
The closed-form solution optimizing this variant of the least-squares regularized objective is obtained by solving the following system of linear equations:
$(B^TB + λU)θ = B^Ty$
where U is a diagonal matrix having value 0 in position (1, 1) and value 1 in the other diagonal entries.

* Implement the Python functions described as follows:
  - q4 features.py computes the linear and quadratic features b l (x) and b q (x).
  - q4 mse.py calculates the mean squared error.
  - q4 train.py learns model parameters θ given a training set of examples by solving the system in Eq. 3 (use Python numpy function numpy.linalg.solve to solve the system).
  - q4 predict.py performs prediction by evaluating a given learned model for the input examples.

  * The performance of the algorithm will depend on the choice of the parameter λ. Implement the function q4 cross validation error.py to perform N -fold cross-validation. Given a training set of examples and a finite set of values for hyperparameter λ, this function should return the cross-validation score, i.e., the average of the mean squared errors over the validation sets. Once you have coded this function, you can run the script q4b.py to plot the 10-fold cross-validation scores for λ ∈ {10 −5 , 10 −3 , 10 −1 , 10, 10 3 ,10 5 , 10 7 } with both the linear model and the quadratic model.
  * Are there any values of λ producing underfitting? If yes, which values?
  * Are there any values of λ producing overfitting? If yes, which values?
  * Which of the two versions of feature vector b(x) produces more overfitting, b l (x) or b q (x)? Can you explain why?
  * Code the function q4 test error.py which trains a model on the training set and returns the test error. Then, execute the script q4f.py, which plots the test set mean squared error for the same set of values of λ as before.
  * Is the cross-validation score a good predictor of performance on the test set? Please comment on why or why not.
