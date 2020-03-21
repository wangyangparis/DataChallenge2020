# Data-Challenge-2020

Xgboost use a **second ordre Taylor approximation**,light gbm also request the gradient and the hessien in its cost function, very alike to Xgboost. For binary classification, we use a log loss:
$$𝐿=−𝑦ln𝑝−𝛽(1−𝑦)ln(1−𝑝)$$ $p$ as the probability estimated by sigmoid function. **$𝛽$ is the multiplier factor to increase the weight of FP loss**.

In order to **penalise False Positive**, I put a penalty Beta on the FP, I can calculate the gradient as: 

$$grad =\frac{∂𝐿}{∂𝑥}=\frac{∂𝐿}{∂𝑝}\frac{∂𝑝}{∂𝑥}=𝑝(𝛽+𝑦−𝛽𝑦)−𝑦 $$,

and hessien as:

$$ hess =∂2𝐿∂𝑥2=𝑝(1−𝑝)(𝛽+𝑦−𝛽𝑦) $$

 **Customizing the training loss** in LightGBM requires defining a function that takes in two arrays, the targets and their predictions. In turn, the function should return two arrays of the gradient and hessian of each observation. As noted above, we need to use calculus to derive gradient and hessian and then implement it in Python.
