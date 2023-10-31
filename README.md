# RAB

Hello! There are a few approaches I've settled on here, and each has a small description given below. I'll add some LaTeX files later that have more information on the math that happens behind each approach. For definitions, the problem that affects my model most significantly is that there are minority groups with low accuracy rates, a feature called representation disparity. Disparity amplification, on the other hand, arises from positive feedback loops within the model itself that lower the accuracy for any minority group. Below are some approaches that may impact both.

1) Adversarial De-Biasing

  In this approach, there are two (or more) models, one which is responsible for encoding information, and a second which is responsible for removing latent representations from the model, i.e., racial information. More generally, if there is some sensitive element "Z" in your data, we would train our secondary variable to predict against a second, noncorrelated variable "Y" (which I may think about the applications of introducing to the dataset as a random element) and NOT predict on Z. In this, effectively, the secondary model would prevent the main model from learning from any parameters in Z in a significant way, while not removing data that may otherwise be highly correlated.

3) Robust Optimization

   If you're familiar with nothing else, understand that machine learning models work by running a complex series of computations extraordinarily quickly -- the most important computation being an optimization formula. An optimization formula would, usually, minimize the mean loss across all groups in the dataset, looking for the best performance overall but not necessarily the best performance for any one group. A robust optimization formula, then, would minimize the loss for all groups in the dataset -- including minimal ones. Each group in the dataset makes up some "alpha" of the dataset proportion, and has its own probability distribution. This approach would effectively approximate those unique probability distributions given some input Z, and look to minimize the expected loss of each individual group, that is, the worst case scenario given some maximal case.

5) Spurious Correlation

   This one may be reaching a little bit but I believe it is possible to use an unsupervised debiasing technique based on stochastic label noise -- essentially preventing a model from capturing spurious correlation noise without necessarily having any prior knowledge of bias in the dataset. I'll have to think a bit to prove it, but it's on the scratchpad. We'll preturb the existing ground-truth labels with a probability of p, ensuring higher performance by (I think) mimicking an ensemble effect.

I have more ideas, but they're all of increasing mathematical complexity and of significantly more limited application. I'll add more as they seem viable. 
