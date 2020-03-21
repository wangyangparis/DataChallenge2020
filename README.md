# Data Challenge
Fusion of Face Recognition Algorithms | 
MS Telecom Big Data | MDI343 Machine Learning

<br>

<p align="center">
  <img src="https://www.statworx.com/wp-content/uploads/machine.png"  width="450" height="450"/>
</p>

<br>

## (customized) Lightgbm
### Custom Loss Function, penalty for FP
Xgboost use a **second ordre Taylor approximation**,light gbm also request the gradient and the hessien in its cost function, very alike to Xgboost. For binary classification, we use a log loss:
𝐿=−𝑦ln𝑝−𝛽(1−𝑦)ln(1−𝑝)$$ $p$ as the probability estimated by sigmoid function. 𝛽 is the multiplier factor to increase the weight of FP loss.

In order to **penalise False Positive**, I put a penalty Beta on the FP, I can calculate the gradient as: 

grad =∂𝐿/∂𝑥=∂𝐿/∂𝑝*∂𝑝/∂𝑥=𝑝(𝛽+𝑦−𝛽𝑦)−𝑦 ,

and hessien as:

hess =∂2/𝐿∂𝑥2=𝑝(1−𝑝)(𝛽+𝑦−𝛽𝑦) 

 **Customizing the training loss** in LightGBM requires defining a function that takes in two arrays, the targets and their predictions. In turn, the function should return two arrays of the gradient and hessian of each observation. As noted above, we need to use calculus to derive gradient and hessian and then implement it in Python.
 
### Custom Eval Metric TPR

Validation loss: This is the function that we use to evaluate the performance of our trained model on unseen data. This is often not the same as the training loss. For example, in the case of a classifier, this is often the area under the curve of the receiver operating characteristic (ROC) — though this is never directly optimized, because it is **not differentiable**. This is often called the “performance or evaluation metric”. The validation loss is often used to tune hyper-parameters. 

Because it doesn’t have as many functional requirements like the training loss does, **the validation loss can be non-convex, non-differentiable, and discontinuous, we can use our Specific True Positive Rate (conditionned by FP < 10e-4) as validation loss**. The validation loss in LightGBM is called metric.
