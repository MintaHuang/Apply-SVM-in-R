Apply a support vector machine to classify hand-written digits by using R library e1071

Steps
1. Randomly select about 20% of the data and set it aside as a test set.
2. Train a linear SVM with soft margin. Cross-validate the margin parameter.
3. Train an SVM with soft margin and RBF kernel. You will have to cross-validate both the soft-margin parameter and the kernel bandwidth.
4. After you have selected parameter values for both algorithms, train each one with the parameter value you have chosen. Then compute the misclassification rate (the proportion of misclassified data points) on the test set.
