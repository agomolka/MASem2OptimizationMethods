# Analysis and comparison of *hyperparameter selection* strategies for a classification problem: </br> a case study of 𝐁𝐀𝐘𝐄𝐒𝐈𝐀𝐍 optimization. 
#### MASem2OptimizationMethods
### Project with non-classical Optimization Methods

$${\color{blue}Bayesian \space Optimization \space \color{red}vs \space \color{lightgreen}Grid \space Search}$$

Data science and machine learning are vital in today's world of rapidly increasing data. **Hyperparameter tuning**,crucial for precise model calibration, can  be approached through techniques like Grid Search and Bayesian Optimization. This analysis examines their effectiveness, efficiency, and properties in machine learning. Findings can benefit practitioners, advance understanding, and inspire new strategies in this field.
 
![alt text](https://github.com/agomolka/MASem2OptimizationMethods/blob/master/img/hyper_tunning_graph.jpg?raw=true)

I want to introduce the general problem of hyperparameter tuning in the context of machine learning and emphasized its importance in achieving 🔥high-quality predictive models. 🔥 In this chapter, we will focus on two **popular methods of hyperparameter optimization - Bayesian Optimization and Grid Search**. We chose to compare these two methods for several reasons. Firstly, Grid Search is one of the most classical 👶🏻 and widely used techniques, popular due to its simplicity. However, it has limitations such as high computational complexity and the inability to leverage information from previous iterations, which can be inefficient, especially for large hyperparameter spaces. On the other hand, Bayesian Optimization is an 🤓 advanced method that uses Bayesian statistics to intelligently explore the hyperparameter space, often leading to better solutions with fewer trials.

### 🚨 Bayesian Optimization 🚨

Bayesian Optimization is an advanced approach to 🌍 global optimization based on utilizing Bayesian inference to update the probability distribution of the optimization model. Over the years, there has been a growing interest in this approach due to its ability to efficiently explore high-dimensional hyperparameter spaces, which is particularly important in the field of machine learning.

![alt text](https://github.com/agomolka/MASem2OptimizationMethods/blob/master/img/bayesian.png?raw=true)

In Bayesian optimization, the concept of sequential modeling based on prior distribution and data collection to inform subsequent iterations plays a crucial role. In practice, a commonly used model is based on a Gaussian process, which allows modeling uncertainty regarding the quality of the model for a given set of hyperparameters. Based on this, Bayesian optimization uses acquisition functions such as Expected Improvement to identify the most promising regions of the parameter space for exploration.

Although Bayesian Optimization demonstrates ⚡️ higher efficiency ⚡️ compared to methods like Grid Search or Random Search, it also has its limitations. The computational complexity of the Gaussian process increases non-linearly with the number of observations, which can pose scalability challenges, especially for large datasets. Additionally, modeling nonlinear dependencies in high-dimensional spaces can be difficult and sometimes requires careful tuning of the optimization model's hyperparameters.
![alt text](https://github.com/agomolka/MASem2OptimizationMethods/blob/master/img/hyperopt.png?raw=true)
It is also worth noting that Bayesian optimization can be susceptible to local minima, especially in cases where the hyperparameter space is highly irregular.

In terms of implementation, Python offers several libraries for utilizing Bayesian Optimization. In our analysis, we utilized the hyperopt library, which is one of the most popular Bayesian Optimization libraries in Python. This library provides an intuitive interface and allows for flexible definition of hyperparameter spaces, as well as various acquisition functions.


As a supplement, it is worth noting a range of practical applications of Bayesian Optimization that contribute to its popularity in various fields:
* 🧠 Deep Learning:  Due to its ability to efficiently explore high-dimensional spaces, Bayesian Optimization is often used for hyperparameter tuning in deep neural networks, which is crucial for achieving high performance in tasks such as image recognition or natural language processing.
* 💊 Pharmaceuticals and Biotechnology: In the life sciences field, Bayesian Optimization finds application in the design of new drugs by identifying optimal combinations of chemical and biological parameters.
* 🏭 Industrial Process Optimization: In engineering and manufacturing, Bayesian Optimization helps optimize processes such as machine parameter regulation or product quality control, leading to increased efficiency and cost reduction.

### 🚨 Grid Search 🚨

Grid Search, also known as full grid search, is one of the **fundamental approaches** in hyperparameter tuning for machine learning models. The idea of the method is to 👵🏼 systematically explore multiple combinations of hyperparameters to identify the configuration that yields the best results according to a chosen quality criterion (e.g., accuracy, loss value). Despite its limitations, Grid Search remains popular due to its simplicity, determinism, and ability to thoroughly search the parameter space. It is often applied in problems with low complexity and where significant computational resources are available.
![alt text](https://github.com/agomolka/MASem2OptimizationMethods/blob/master/img/grid.png?raw=true)

In practical implementations, Grid Search can be performed using various programming libraries. In the case of Python, a popular tool is GridSearchCV from the sklearn.model_selection library.

Grid Search, despite its simplicity, finds applications in many areas of machine learning. The most common areas where it is frequently used include:
* 🥱 Classification Model Tuning: For algorithms such as Support Vector Machines (SVM) or decision trees, Grid Search allows for optimization of hyperparameters to improve classification performance.
* 🥱 Regression Model Tuning: In regression analysis, Grid Search is used to select hyperparameters for models such as linear regression, LASSO, or Ridge regression.

In conclusion, Grid Search is a versatile yet computationally expensive tool often used in situations where the hyperparameter space is relatively small or when exhaustive search is necessary.

### ‼️ Comparison of the Adopted Algorithms ‼️

In summary, both Grid Search and Bayesian Optimization are important tools in the process of hyperparameter tuning for machine learning models, but their characteristics and applications differ significantly.

Grid Search is a deterministic approach that exhaustively searches the hyperparameter space. It is easy to understand and implement, but its computational complexity grows exponentially with the number of hyperparameters. This can be a limitation, especially for large hyperparameter spaces.

On the other hand, Bayesian Optimization is a probabilistic approach that leverages information from previous trials to intelligently explore the hyperparameter space. This can lead to better solutions with fewer trials compared to Grid Search. However, implementing Bayesian Optimization can be more complex, and the method itself may sometimes be less intuitive.

The choice between Grid Search and Bayesian Optimization depends on the specifics of the problem, the size of the hyperparameter space, the available computational resources, and preferences regarding the determinism or adaptability of the method. Grid Search may be more suitable for smaller spaces, while Bayesian Optimization may offer higher efficiency in complex and high-dimensional hyperparameter spaces.

### 🔥 Summary 🔥

During the study, several difficulties and observations emerged when comparing GridSearch and Hyperopt as hyperparameter optimization algorithms.

One of the primary challenges encountered with GridSearch was its computationally intensive nature, especially when dealing with large hyperparameter spaces. GridSearch exhaustively searches through all possible combinations of hyperparameters, resulting in a time-consuming process. This can be particularly problematic when working with complex models that require significant computational resources, as it may limit the number of experiments that can be performed within a reasonable timeframe.

![alt text](https://github.com/agomolka/MASem2OptimizationMethods/blob/master/img/hyperparam.png?raw=true)

In contrast, Hyperopt demonstrated a more efficient approach to hyperparameter optimization. By leveraging Bayesian optimization and decision trees, Hyperopt intelligently explores the hyperparameter space, focusing on promising regions and adapting its sampling strategy. This adaptive sampling allows Hyperopt to converge to optimal solutions more quickly, resulting in faster optimization times compared to GridSearch. As a result, Hyperopt proved to be highly effective, delivering satisfactory results within significantly shorter timeframes.

![alt text](https://github.com/agomolka/MASem2OptimizationMethods/blob/master/img/learning_rate.png?raw=true)

Another notable difference between the two algorithms lies in their ability to utilize information from previous iterations. GridSearch does not incorporate any information from previous experiments and treats each combination of hyperparameters independently. In contrast, Hyperopt employs Bayesian optimization, which updates the probability distribution over the hyperparameter space based on previous evaluations. This enables Hyperopt to make more informed decisions about which hyperparameters to explore next, leading to faster convergence and better overall performance.

![alt text](https://github.com/agomolka/MASem2OptimizationMethods/blob/master/img/learning_rate_vs_n_estim.png?raw=true)
<p align="center"><img src="[drawing.jpg](https://github.com/agomolka/MASem2OptimizationMethods/blob/master/img/learning_rate_vs_n_estim.png)" alt="drawing" width="200"/></p>

In terms of interpretability, GridSearch offers a straightforward and easily understandable structure. It allows researchers and practitioners to systematically explore the entire hyperparameter space, providing a clear understanding of the search process. On the other hand, Hyperopt's implementation can be more complex and requires a deeper understanding of Bayesian optimization and decision trees. However, the knowledge gained from understanding Hyperopt's workings extends beyond hyperparameter tuning and provides valuable insights into the broader field of machine learning.

In conclusion, the study highlighted the limitations of GridSearch, particularly in terms of computational complexity and the lack of adaptability. The findings strongly supported the effectiveness of Hyperopt as a more efficient alternative, demonstrating its ability to achieve satisfactory results within shorter timeframes. Hyperopt's adaptive sampling strategy, Bayesian optimization, and utilization of information from previous iterations contributed significantly to its superior performance. While GridSearch remains a valuable tool, especially for smaller hyperparameter spaces and scenarios where interpretability is crucial, exploring alternative techniques like Hyperopt is essential for achieving efficient and effective hyperparameter optimization in machine learning.
