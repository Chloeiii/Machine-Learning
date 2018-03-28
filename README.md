
# Machine Learning :checkered_flag:

### Tutorials && Courses :dog:
* [Machine Learning Tutorial for Beginners](https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners)
* [Machine Learning - Andrew Ng](https://www.coursera.org/learn/machine-learning)

----
### Contents :cookie:
* [What is Machine Learing](#what-is-machine-learning-rice_ball)
* [Supervised vs. Unsupervised Machine Learning](#supervised-and-unsupervised-machine-learning-bread)
* [Random Forest](#random-forest-stuck\_out\_tongue)

----
### What is Machine Learning :rice_ball:
    Machine learning is a field of computer science that gives computer systems the ability 
    to "learn" (i.e. progressively improve performance on a specific task) with data, 
    without being explicitly programmed.
    
    machine learning explores the study and construction of algorithms that can learn from 
    and make predictions on data – such algorithms overcome following strictly static program 
    instructions by making data-driven predictions or decisions, through building a model 
    from sample inputs. Machine learning is employed in a range of computing tasks where 
    designing and programming explicit algorithms with good performance is difficult or infeasible; 

    example applications include email filtering, detection of network intruders or malicious insiders 
    working towards a data breach, optical character recognition (OCR), learning to rank, and computer 
    vision.
    
    Effective machine learning is difficult because finding patterns is hard and often not enough 
    training data are available; as a result, machine-learning programs often fail to deliver.
    
### Supervised and Unsupervised Machine Learning :bread:
* [Learn more](https://www.datascience.com/blog/supervised-and-unsupervised-machine-learning-algorithms)

* [Explaination with a real-life examples](http://dataaspirant.com/2014/09/19/supervised-and-unsupervised-learning/)

    Supervised Machine Learning:  
      the data scientist acts as a guide to teach the algorithm what conclusions it should come up with. 
      It’s similar to the way a child might learn arithmetic from a teacher.
      #Learning from the know label data to create a model then predicting target class for the given input data.
                                  
    Unsupervised Machine Learning:  
      unsupervised machine learning is more closely aligned with what some call true artificial intelligence 
      — the idea that a computer can learn to identify complex processes and patterns without a human to 
      provide guidance along the way.
      #Learning from the unlabeled data to differentiating the given input data.

### Random Forest :stuck\_out\_tongue:
* [Random Forest in Python](http://blog.yhat.com/posts/random-forests-in-python.html)

<img src="http://blog.yhat.com/static/img/decision_tree_example.png" width="430"><img src="http://blog.yhat.com/static/img/a_random_forest.png" width="430">
<img src="http://blog.yhat.com/static/img/log_lm_vs_rf.png" width="900">

* Overview

        Random forest is a highly versatile machine learning method with numerous applications 
        ranging from marketing to healthcare and insurance. It can be used to model the impact of 
        marketing on customer acquisition, retention, and churn or to predict disease risk and 
        susceptibility in patients.

        Random forest is capable of regression and classification. It can handle a large number of 
        features, and it's helpful for estimating which of your variables are important in the 
        underlying data being modeled.
* What is a Random Forest?

        Random forest is solid choice for nearly any prediction problem (even non-linear ones). 
        It's a relatively new machine learning strategy (it came out of Bell Labs in the 90s) 
        and it can be used for just about anything. It belongs to a larger class of 
        machine learning algorithms called ensemble methods.
        
        Random Forest
        The algorithm to induce a random forest will create a bunch of random decision trees automatically. 
        Since the trees are generated at random, most won't be all that meaningful to learning your 
        classification/regression problem (maybe 99.9% of trees).
        
        When you make a prediction, the new observation gets pushed down each decision tree and assigned 
        a predicted value/label. Once each of the trees in the forest have reported its predicted value/label, 
        the predictions are tallied up and the mode vote of all trees is returned as the final prediction.

        Simply, the 99.9% of trees that are irrelevant make predictions that are all over the map and cancel 
        each another out. The predictions of the minority of trees that are good top that noise and yield a 
        good prediction.
* Uses

        1. Variable Selection 
            One of the best use cases for random forest is feature selection. One of the byproducts of trying 
            lots of decision tree variations is that you can examine which variables are working best/worst 
            in each tree.
            
            e.g. figure out which variables are most important for classifying a wine as being red or white.

        2. Classification
            Random forest is also great for classification. It can be used to make predictions for categories 
            with multiple possible values and it can be calibrated to output probabilities as well. One thing 
            you do need to watch out for is overfitting. Random forest can be prone to overfitting, especially 
            when working with relatively small datasets. You should be suspicious if your model is making 
            "too good" of predictions on our test set.
            
        3. Regression
            random forest--unlike other algorithms--does really well learning on categorical variables or 
            a mixture of categorical and real variables. Categorical variables with high cardinality 
            (# of possible values) can be tricky, so having something like this in your back pocket can 
            come in quite useful.    
* RandomForestClassifier
<img src="http://scikit-learn.org/stable/_images/sphx_glr_plot_forest_iris_001.png" width="700">

        A random forest is a meta estimator that fits a number of decision tree classifiers on various 
        sub-samples of the dataset and use averaging to improve the predictive accuracy and control 
        over-fitting. The sub-sample size is always the same as the original input sample size but 
        the samples are drawn with replacement if bootstrap=True (default).
        
        An Example - the iris dataset:
            https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/
            
