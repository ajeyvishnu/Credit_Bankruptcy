library(aod)
library(caret)
library(caTools)
library(dplyr)
library(ggplot2)
library(ISLR)
library(mlbench)
library(plyr)
library(readr)
library(stats)
library(MASS)

###Explore data set###

#Initial Observations
#12 car0 and 337 car - Changed car0 to car
#no other NA values found

dim(credit)
summary(credit)

#Initially did not exclude any columns as I am not sure which will effect the deafulters and which will not

###Split data###
set.seed(1000)
credit$default <- as.factor(credit$default)

Training <- createDataPartition(credit$default, p=0.8, list=FALSE)
training <- credit[ Training, ]
testing <- credit[ -Training, ]

preproc <- c("center", "scale")
control <- trainControl(method = "cv", number = 10, savePredictions = TRUE)   

########################## LOGISTIC MODEL ##########################

mod_log_t1 <- train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                    + amount + savings_balance + employment_duration + percent_of_income
                    + years_at_residence + age + other_credit + housing + existing_loans_count
                    + job + dependents + phone,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t1)
pred_log_t1 = predict(mod_log_t1, newdata=testing)

confusionMatrix(data=pred_log_t1, testing$default, mode = 'prec_recall')

mod_log_t2 <- train(default ~ checking_balance + months_loan_duration + credit_history + amount + percent_of_income,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t2)
pred_log_t2 = predict(mod_log_t2, newdata=testing)

confusionMatrix(data=pred_log_t2, testing$default, mode = 'prec_recall')

mod_log_t3 <- train(default ~ checking_balance + months_loan_duration + credit_history + amount + factor(percent_of_income),
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t3)
pred_log_t3 = predict(mod_log_t3, newdata=testing)

confusionMatrix(data=pred_log_t3, testing$default, mode = 'prec_recall')

mod_log_t4 <- train(default ~ checking_balance + months_loan_duration + credit_history + percent_of_income,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t4)
pred_log_t4 = predict(mod_log_t4, newdata=testing)

confusionMatrix(data=pred_log_t4, testing$default, mode = 'prec_recall')

mod_log_t5 <- train(default ~ checking_balance + months_loan_duration + credit_history,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t5)
pred_log_t5 = predict(mod_log_t5, newdata=testing)

confusionMatrix(data=pred_log_t5, testing$default, mode = 'prec_recall')

mod_log_t6 <- train(default ~ months_loan_duration + credit_history,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t6)
pred_log_t6 = predict(mod_log_t6, newdata=testing)

confusionMatrix(data=pred_log_t6, testing$default, mode = 'prec_recall')

mod_log_t7 <- train(default ~ factor(months_loan_duration) + credit_history,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t7)
pred_log_t7 = predict(mod_log_t7, newdata=testing)

confusionMatrix(data=pred_log_t7, testing$default, mode = 'prec_recall')

mod_log_t8 <- train(default ~ checking_balance + factor(months_loan_duration) + credit_history,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t8)
pred_log_t8 = predict(mod_log_t8, newdata=testing)

confusionMatrix(data=pred_log_t8, testing$default, mode = 'prec_recall')

mod_log_t9 <- train(default ~ months_loan_duration,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t9)
pred_log_t9 = predict(mod_log_t9, newdata=testing)

confusionMatrix(data=pred_log_t9, testing$default, mode = 'prec_recall')

mod_log_t10 <- train(default ~ credit_history,
                    data=training, method="glm", family="binomial",
                    metric = "Accuracy",
                    trControl = control, preProcess = preproc)

summary(mod_log_t10)
pred_log_t10 = predict(mod_log_t10, newdata=testing)

confusionMatrix(data=pred_log_t10, testing$default, mode = 'prec_recall')

########################## LDA MODEL ##########################

mod_lda1 <- train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                        + amount + savings_balance + employment_duration + percent_of_income
                        + years_at_residence + age + other_credit + housing + existing_loans_count
                        + job + dependents + phone,  
                        data = training, method = "lda", family="binomial",                 
                        metric = "Accuracy",   
                        trControl = control, preProcess = preproc)                 

mod_lda1                                                
pred_lda1 = predict(mod_lda1, newdata=testing)

confusionMatrix(data=pred_lda1, testing$default, mode = 'prec_recall')    

mod_lda2 <- train(default ~ checking_balance + months_loan_duration + credit_history + amount + percent_of_income,  
                  data = training, method = "lda", family="binomial",                 
                  metric = "Accuracy",   
                  trControl = control, preProcess = preproc)                 

mod_lda2                                               
pred_lda2 = predict(mod_lda2, newdata=testing)

confusionMatrix(data=pred_lda2, testing$default, mode = 'prec_recall')  

mod_lda3 <- train(default ~ checking_balance + months_loan_duration + credit_history + amount + factor(percent_of_income),  
                  data = training, method = "lda", family="binomial",                 
                  metric = "Accuracy",   
                  trControl = control, preProcess = preproc)                 

mod_lda3                                               
pred_lda3 = predict(mod_lda3, newdata=testing)

confusionMatrix(data=pred_lda3, testing$default, mode = 'prec_recall')  

mod_lda4 <- train(default ~ checking_balance + months_loan_duration + credit_history + percent_of_income,  
                  data = training, method = "lda", family="binomial",                 
                  metric = "Accuracy",   
                  trControl = control, preProcess = preproc)                 

mod_lda4                                              
pred_lda4 = predict(mod_lda4, newdata=testing)

confusionMatrix(data=pred_lda4, testing$default, mode = 'prec_recall')  

mod_lda5 <- train(default ~ checking_balance + months_loan_duration + credit_history,  
                  data = training, method = "lda", family="binomial",                 
                  metric = "Accuracy",   
                  trControl = control, preProcess = preproc)                 

mod_lda5                                              
pred_lda5 = predict(mod_lda5, newdata=testing)

confusionMatrix(data=pred_lda5, testing$default, mode = 'prec_recall')

mod_lda6 <- train(default ~ months_loan_duration + credit_history,  
                  data = training, method = "lda", family="binomial",                 
                  metric = "Accuracy",   
                  trControl = control, preProcess = preproc)                 

mod_lda6                                              
pred_lda6 = predict(mod_lda6, newdata=testing)

confusionMatrix(data=pred_lda6, testing$default, mode = 'prec_recall')

mod_lda7 <- train(default ~ factor(months_loan_duration) + credit_history,  
                  data = training, method = "lda", family="binomial",                 
                  metric = "Accuracy",   
                  trControl = control, preProcess = preproc)                 

mod_lda7                                              
pred_lda7 = predict(mod_lda7, newdata=testing)

confusionMatrix(data=pred_lda7, testing$default, mode = 'prec_recall')

mod_lda8 <- train(default ~ checking_balance + factor(months_loan_duration) + credit_history,  
                  data = training, method = "lda", family="binomial",                 
                  metric = "Accuracy",   
                  trControl = control, preProcess = preproc)                 

mod_lda8                                              
pred_lda8 = predict(mod_lda8, newdata=testing)

confusionMatrix(data=pred_lda8, testing$default, mode = 'prec_recall')

mod_lda9 <- train(default ~ months_loan_duration,  
                  data = training, method = "lda", family="binomial",                 
                  metric = "Accuracy",   
                  trControl = control, preProcess = preproc)                 

mod_lda9                                              
pred_lda9 = predict(mod_lda9, newdata=testing)

confusionMatrix(data=pred_lda9, testing$default, mode = 'prec_recall')

mod_lda10 <- train(default ~ credit_history,  
                  data = training, method = "lda", family="binomial",                 
                  metric = "Accuracy",   
                  trControl = control, preProcess = preproc)                 

mod_lda10                                           
pred_lda10 = predict(mod_lda10, newdata=testing)

confusionMatrix(data=pred_lda10, testing$default, mode = 'prec_recall')


########################## QDA MODEL ##########################

mod_qda1 <- train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                    + amount + savings_balance + employment_duration + percent_of_income
                    + years_at_residence + age + other_credit + housing + existing_loans_count
                    + job + dependents + phone,     
                        data = training, method = "qda", family="binomial",  
                        metric = "Accuracy",                          
                        trControl = control, preProcess = preproc)      

mod_qda1
pred_qda1 = predict(mod_qda1, newdata=testing)

confusionMatrix(data=pred_qda1, testing$default, mode = 'prec_recall')                      

mod_qda2 <- train(default ~ checking_balance + months_loan_duration + credit_history + amount + percent_of_income,     
                  data = training, method = "qda", family="binomial",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc)      

mod_qda2
pred_qda2 = predict(mod_qda2, newdata=testing)

confusionMatrix(data=pred_qda2, testing$default, mode = 'prec_recall')

mod_qda3 <- train(default ~ checking_balance + months_loan_duration + credit_history + amount + factor(percent_of_income),     
                  data = training, method = "qda", family="binomial",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc)      

mod_qda3
pred_qda3 = predict(mod_qda3, newdata=testing)

confusionMatrix(data=pred_qda3, testing$default, mode = 'prec_recall')  

mod_qda4 <- train(default ~ checking_balance + months_loan_duration + credit_history + percent_of_income,     
                  data = training, method = "qda", family="binomial",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc)      

mod_qda4
pred_qda4 = predict(mod_qda4, newdata=testing)

confusionMatrix(data=pred_qda4, testing$default, mode = 'prec_recall')  

mod_qda5 <- train(default ~ checking_balance + months_loan_duration + credit_history,     
                  data = training, method = "qda", family="binomial",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc)      

mod_qda5
pred_qda5 = predict(mod_qda5, newdata=testing)

confusionMatrix(data=pred_qda5, testing$default, mode = 'prec_recall')  

mod_qda6 <- train(default ~ months_loan_duration + credit_history,     
                  data = training, method = "qda", family="binomial",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc)      

mod_qda6
pred_qda6 = predict(mod_qda6, newdata=testing)

confusionMatrix(data=pred_qda6, testing$default, mode = 'prec_recall') 

mod_qda7 <- train(default ~ factor(months_loan_duration) + credit_history,     
                  data = training, method = "qda", family="binomial",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc)      

mod_qda7
pred_qda7 = predict(mod_qda7, newdata=testing)

confusionMatrix(data=pred_qda7, testing$default, mode = 'prec_recall') 

mod_qda8 <- train(default ~ checking_balance + factor(months_loan_duration) + credit_history,     
                  data = training, method = "qda", family="binomial",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc)      

mod_qda8
pred_qda8 = predict(mod_qda8, newdata=testing)

confusionMatrix(data=pred_qda8, testing$default, mode = 'prec_recall') 

mod_qda9 <- train(default ~ months_loan_duration,     
                  data = training, method = "qda", family="binomial",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc)      

mod_qda9
pred_qda9 = predict(mod_qda9, newdata=testing)

confusionMatrix(data=pred_qda9, testing$default, mode = 'prec_recall')

mod_qda10 <- train(default ~ credit_history,     
                  data = training, method = "qda", family="binomial",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc)      

mod_qda10
pred_qda10 = predict(mod_qda10, newdata=testing)

confusionMatrix(data=pred_qda10, testing$default, mode = 'prec_recall')

########################## KNN ##########################

mod_knn1 <- train(default ~ checking_balance + months_loan_duration + credit_history + purpose
                    + amount + savings_balance + employment_duration + percent_of_income
                    + years_at_residence + age + other_credit + housing + existing_loans_count
                    + job + dependents + phone,     
                    data = training, method = "knn",  
                    metric = "Accuracy",                          
                    trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn1
pred_knn1 = predict(mod_knn1, newdata=testing)

confusionMatrix(data=pred_knn1, testing$default, mode = 'prec_recall')   

mod_knn2 <- train(default ~ checking_balance + months_loan_duration + credit_history + amount + percent_of_income,     
                  data = training, method = "knn",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn2
pred_knn2 = predict(mod_knn2, newdata=testing)

confusionMatrix(data=pred_knn2, testing$default, mode = 'prec_recall')    

mod_knn3 <- train(default ~ checking_balance + months_loan_duration + credit_history + amount + factor(percent_of_income),     
                  data = training, method = "knn",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn3
pred_knn3 = predict(mod_knn3, newdata=testing)

confusionMatrix(data=pred_knn3, testing$default, mode = 'prec_recall')  

mod_knn4 <- train(default ~ checking_balance + months_loan_duration + credit_history + percent_of_income,     
                  data = training, method = "knn",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn4
pred_knn4 = predict(mod_knn4, newdata=testing)

confusionMatrix(data=pred_knn4, testing$default, mode = 'prec_recall')

mod_knn5 <- train(default ~ checking_balance + months_loan_duration + credit_history,     
                  data = training, method = "knn",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn5
pred_knn5 = predict(mod_knn5, newdata=testing)

confusionMatrix(data=pred_knn5, testing$default, mode = 'prec_recall')

mod_knn6 <- train(default ~ months_loan_duration + credit_history,     
                  data = training, method = "knn",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn6
pred_knn6 = predict(mod_knn6, newdata=testing)

confusionMatrix(data=pred_knn6, testing$default, mode = 'prec_recall')

mod_knn7 <- train(default ~ factor(months_loan_duration) + credit_history,     
                  data = training, method = "knn",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn7
pred_knn7 = predict(mod_knn7, newdata=testing)

confusionMatrix(data=pred_knn7, testing$default, mode = 'prec_recall')

mod_knn8 <- train(default ~ checking_balance + factor(months_loan_duration) + credit_history,     
                  data = training, method = "knn",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn8
pred_knn8 = predict(mod_knn8, newdata=testing)

confusionMatrix(data=pred_knn8, testing$default, mode = 'prec_recall')

mod_knn9 <- train(default ~ months_loan_duration,     
                  data = training, method = "knn",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn9
pred_knn9 = predict(mod_knn9, newdata=testing)

confusionMatrix(data=pred_knn9, testing$default, mode = 'prec_recall')

mod_knn10 <- train(default ~ credit_history,     
                  data = training, method = "knn",  
                  metric = "Accuracy",                          
                  trControl = control, preProcess = preproc, tuneLength = 100)      

mod_knn10
pred_knn10 = predict(mod_knn10, newdata=testing)

confusionMatrix(data=pred_knn10, testing$default, mode = 'prec_recall')

########################## TREE ##########################

mod_tree1 = train(default ~ checking_balance + months_loan_duration + credit_history + purpose
              + amount + savings_balance + employment_duration + percent_of_income
              + years_at_residence + age + other_credit + housing + existing_loans_count
              + job + dependents + phone,            
              data=training,                 
              method="rpart",                 
              parms = list(split="gini"),     
              metric = "Accuracy",       
              trControl = control,    
              tuneLength = 100)  

mod_tree1
pred_tree1 = predict(mod_tree1, newdata=testing)

confusionMatrix(data=pred_tree1, testing$default, mode = 'prec_recall') 

mod_tree2 = train(default ~ checking_balance + months_loan_duration + credit_history + amount + percent_of_income,            
                  data=training,                 
                  method="rpart",                 
                  parms = list(split="gini"),     
                  metric = "Accuracy",       
                  trControl = control,    
                  tuneLength = 100)  

mod_tree2
pred_tree2 = predict(mod_tree2, newdata=testing)

confusionMatrix(data=pred_tree2, testing$default, mode = 'prec_recall') 

mod_tree3 = train(default ~ checking_balance + months_loan_duration + credit_history + amount + factor(percent_of_income),            
                  data=training,                 
                  method="rpart",                 
                  parms = list(split="gini"),     
                  metric = "Accuracy",       
                  trControl = control,    
                  tuneLength = 100)  

mod_tree3
pred_tree3 = predict(mod_tree3, newdata=testing)

confusionMatrix(data=pred_tree3, testing$default, mode = 'prec_recall') 

mod_tree4 = train(default ~ checking_balance + months_loan_duration + credit_history + percent_of_income,            
                  data=training,                 
                  method="rpart",                 
                  parms = list(split="gini"),     
                  metric = "Accuracy",       
                  trControl = control,    
                  tuneLength = 100)  

mod_tree4
pred_tree4 = predict(mod_tree4, newdata=testing)

confusionMatrix(data=pred_tree4, testing$default, mode = 'prec_recall') 

mod_tree5 = train(default ~ checking_balance + months_loan_duration + credit_history,            
                  data=training,                 
                  method="rpart",                 
                  parms = list(split="gini"),     
                  metric = "Accuracy",       
                  trControl = control,    
                  tuneLength = 100)  

mod_tree5
pred_tree5 = predict(mod_tree5, newdata=testing)

confusionMatrix(data=pred_tree5, testing$default, mode = 'prec_recall') 

mod_tree6 = train(default ~ months_loan_duration + credit_history,            
                  data=training,                 
                  method="rpart",                 
                  parms = list(split="gini"),     
                  metric = "Accuracy",       
                  trControl = control,    
                  tuneLength = 100)  

mod_tree6
pred_tree6 = predict(mod_tree6, newdata=testing)

confusionMatrix(data=pred_tree6, testing$default, mode = 'prec_recall') 

mod_tree7 = train(default ~ factor(months_loan_duration) + credit_history,            
                  data=training,                 
                  method="rpart",                 
                  parms = list(split="gini"),     
                  metric = "Accuracy",       
                  trControl = control,    
                  tuneLength = 100)  

mod_tree7
pred_tree7 = predict(mod_tree7, newdata=testing)

confusionMatrix(data=pred_tree7, testing$default, mode = 'prec_recall') 

mod_tree8 = train(default ~ checking_balance + factor(months_loan_duration) + credit_history,            
                  data=training,                 
                  method="rpart",                 
                  parms = list(split="gini"),     
                  metric = "Accuracy",       
                  trControl = control,    
                  tuneLength = 100)  

mod_tree8
pred_tree8 = predict(mod_tree8, newdata=testing)

confusionMatrix(data=pred_tree8, testing$default, mode = 'prec_recall') 

mod_tree9 = train(default ~ months_loan_duration,            
                  data=training,                 
                  method="rpart",                 
                  parms = list(split="gini"),     
                  metric = "Accuracy",       
                  trControl = control,    
                  tuneLength = 100)  

mod_tree9
pred_tree9 = predict(mod_tree9, newdata=testing)

confusionMatrix(data=pred_tree9, testing$default, mode = 'prec_recall') 

mod_tree10 = train(default ~ credit_history,            
                  data=training,                 
                  method="rpart",                 
                  parms = list(split="gini"),     
                  metric = "Accuracy",       
                  trControl = control,    
                  tuneLength = 100)  

mod_tree10
pred_tree10 = predict(mod_tree10, newdata=testing)

confusionMatrix(data=pred_tree10, testing$default, mode = 'prec_recall') 






