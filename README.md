# utah-cs5350-fa21
Machine Learning 

This is a machine learning library developed by Vijay Bajracharya for CS5350/6350 in University of Utah

Specific Instructions to modify python scripts for testing purposes (Assignemnt 2)

All of the scripts have comment blocks to indicate code that was used in performing experiments for the written report.
Uncomment these sections using the following guidelines to verify experiment results.

There are two separate run.sh files one in EnsembleLearning directory and the other in LinearRegression directory

#################
Ensemble Learning
#################

file: adaboost.py

-> Modify size of for loop in main method to change the number of iterations in adaboost

-> Uncomment middle portion of ada_fit() to print individual stump errors

-> Uncomment open() commands to write errors to a local file



file: bagging.py

-> Modify size of for loop in main method to change the number of trees in bagging

-> Uncomment open() commands to write errors to a local file

-> Uncomment last portion of main method to compute bias/variance tradeoff for single and bagged trees



file: randomforest.py

-> Modify size of for loop in main method to change the number of random trees in random forest

-> Change "rf = RandomForest(no_classifiers=size, G=6)" on line 174 and set G to desired feature subset size

-> Uncomment open() commands to write errors to a local file

-> Uncomment last portion of main method to compute bias/variance tradeoff for single trees and random forests

#################
Linear Regression
#################

file: grad_desc.py

-> LEARNING_RATE is declared as a global variable and can be changed at the top of the file

-> Uncomment plt.show() to display plots if system is able to display plots

-> Uncomment "w = np.dot(np.linalg.inv(np.dot(test.T, test)), np.dot(test.T, y_test))" on line 197 to verify analytical weight vector calculation


