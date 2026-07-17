# Some ideas:
    # 1. Collect more data
    # 2. Collect more divers training set
    # 3. Train algorithm longer with gradient descent
    # 4. Try Adam instead of gradient descent
    # 5. Try bigger newtwork
    # 6. Try smaller network
    # 7. Try dropout
    # 8. Add L2 regularization
    # 9. Network archiotecture : activation functions,hidden units ....
     


# Orthogonalization;
    #  a design principle ensuring that tuning one hyperparameter or component of a model affects only a specific aspect of performance without causing unintended side effects on others.


# chain of assumptions in ML:
    # Fit training set well on cost function.
    # fit dev set well on cost function
    # fit test set well on cost fuction
    # performs well in real world.


# Single Number evaluation metric:
    # Precision (Positive Predictive Value): Focuses on minimizing False Positives (e.g., in email spam detection, ensuring legitimate emails are not marked as spam).
    # Recall (Sensitivity): Focuses on minimizing False Negatives (e.g., in medical diagnosis, ensuring no sick patients are overlooked).
    # f1-score :  It is particularly useful for imbalanced datasets, offering a better balance between false positives and false negatives than accuracy alone. 
    # Formula: 
    # f1=2*(precision*recall)/(precision+recall)

# Precision= TP/(TP+FP)
# Recall = TP/(TP+FN)


# Satisficing and optimizing metrics:
# classfier   Accuracy        Running Time
# A             90%             80ms
# B             92%             95ms
# C             95%             1500ms

# because we want to maximize the accuracy it is optimizing metrics
# and minimizing the running time so it is satisficing metrics.


# saticficing metric means it should satisfy some threshold like Running time should be < 100ms. 
# so in this case we will not consider Model C . 
# Because Model B is more accurate than Model A.




# Train/dev/test distributions:
    # For dev / test set:
        # dev / test samples  should comes from the same sample distribution. choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on.





# Size of dev and test set :
    #  for smaller dataset size we consider 70:30 or 60:20:20
    # but for larger or much larger dataset size for example we have 1 million examples we consider 98:1:1 (1% = 10000 examples which is enough for testing and validation)

    # size of test set: set your set to be big enough to give high confidence in the overall performance of your system.
    # sometime not having test set if okay.(train and dev split is enough)

# When to change dev/test sets and metrics:
    #  for example   Model A : 3% classification error but is does not follow company privacy policies
        # Model B : 5% classfication error but it is follows company privacy policies.



    #  so We will consider Model B because it follows correct policies.
    # Here either  we change the evaluation metric or change the dev/test dataset. 

    # Error = sum(y_pred+y_act)/m
    # for  change the evaluation metric 
    # we add a weight term here.
    # Error = sum(w*(y_pred+y_act))/m
    # where W = {1 if x[i] does follows correct policies}
        # else W={10 if x[i] dones not follows policies.}.       the weight term is very high that means that it increase the error if some example unfollow the policies.

# cost function J = sum(W*Loss_func(y_hat,y))/m
 
# overtime keep training the algorithm bigger model and more data the performance approaches but never surpasses some theoretical limit (Human level performance) which is called Bayes optimal Error (best possible error).


# Avoidable Bias:
    # if models performance is very close to the human performance (human erro : 8%.  and.  model error : 7.5%.  avoidable bias: 0.5% ) then it is more likely to the human performance. this is called avoidable bias because we can not reduce the error more. the difference between the training error and the Bayes optimal error (often approximated by human-level performance).



# Improving your model performance:
    # twofundamental assumptions of supervised learning:
        # we can fit the training set pretty well.
        # training set performance generalize pretty well to the dev/test set.

    # Reducing bias ans variance:
        # Human Level
        #    |  Avoidable Bias
        # Training Error
        #    | variance
        # Dev error

    # for reducing  avoidable bias: train bigger model . train longer/better optimization algorithm. NN architecutre/hyperparameters search.(RNN/CNN)
    # for reducing variance : use more data,regularization , 


# Correcting incorrect dev/test set examples:
    # apply same pricess to your dev and test to make sure they continue to come from the same distribution.
    # Consider examining examples your algorithm got right as well as ones it got wrong.
    # train and dev/test data may now come from slightly different distribution.


# Addressing data mismatch:
    # carry out manual error analysis to try yo understand differences between training and dev/test sets.
    #make training data more similar, or collect more data similar to dev/test sets.

# Transferlearning:
    # if we have a small dataset then just retrain the one last layer at the output of transfer learning model but if a lot of data then may be can retrain all the parameters in the network and if we retrain all the parameters in the NN then this initial phase of training on image recognition is sometimes called pre-training brcause we are using image recognition data to preinitializa or really pretrain the weights of the neural network and then if we update all the weights afterwards then training on the radiology data sometimes that is called finetunning .



#  When multitask learning make sense:
    # training on a set of tasks that could benefits fromhaving shared lower level features
    # Amount of data you have for eaach task is quite similar.
    # can train a big enough NN to do well on all the tasks.
    # 