##### **5. Methodology**
####5.1 naive bayes model weakest model
#load(file = "Preprocessdata1.rda")
# To get columns with integers
library(caTools)
library(caret)
library(e1071)
library(Metrics)
library(pROC)
load(file = 'data.rda')
sample <- sample.split(df.train,SplitRatio = 0.7)
train_70 <- subset(df.train,sample == T)
train_30 <- subset(df.train,sample == F)
response_30 <- train_30$Response
train_30 <- train_naive_30[,-125]
train_naive_70$Response <- as.factor(train_naive_70$Response)
#NAive bayes algorithm (No need of one hot data)
naive_model=naiveBayes(Response~.,data = train_naive_70)
summary(naive_model)
#prediction Naive Bayes
naive_prediction=predict(naive_model,train_naive_30)
View(naive_prediction)

kappa_naive<-ScoreQuadraticWeightedKappa(response_30,naive_prediction)
table(response_30,naive_prediction)
accuracy_naive<-accuracy(response_30,naive_prediction)

Metrics::precision(response_30,naive_prediction)
Metrics::recall(response_30,naive_prediction)
auc_naive<-Metrics::auc(response_30,naive_prediction)
pROC::multiclass.roc(response_30,as.numeric(naive_prediction),levels = levels(factor(response_30)))


# ### **5.2 Multinominal Logistic Model**
# 
# 
# #We start by dividing the data into two portions and using one portion to predict the model 
# #and the other portion for model validation and insample prediction.
# #View(train_apca)
# 
# train_apca6$Response<-train$Response
# train_apca6$random <- runif(nrow(train_apca6))
# #write.csv(train_apca6,'file1.csv')
# train_apca_70 <- train_apca6[train_apca6$random <= 0.7,]
# train_apca_30 <- train_apca6[train_apca6$random > 0.7,]
# train_apca_90 <- train_apca6[train_apca6$random <= 0.9,]
# train_apca_10<-train_apca6[train_apca6$random > 0.9,]
# 
# #We look at distribution of response on both the portions of train data.
# 
# temp1<-data.frame(round(table(train_apca_70$Response)/nrow(train_apca_70),2))
# round(table(train_apca_70$Response)/nrow(train_apca_70),2)
# View(temp1)
# 
# 
# temp1$Freq_test<-as.data.frame(round(table(train_apca_30$Response)/nrow(train_apca_30),2))[,2]
# 
# round(table(train_apca_30$Response)/nrow(train_apca_30),2)
# 
# 
# #### Comparing distribution of test and train data
# 
# plot_ly(data = temp1, x=~Var1,y=~Freq, type = "bar", name="Train")%>%
#   add_trace(y=~Freq_test, name="Test")%>%
#   layout(barmode = 'group')
# 
# colnames(train_apca_70)
# 
# #############Alternative data 
# 
# 
# #train_apca6$Response<-train_clean$Response
# train_clean_1<-train_clean
# # train_clean_1 is only for the multinom model
# train_clean_1$random <- runif(nrow(train_clean_1))
# #write.csv(train_apca6,'file1.csv')
# train_clean_1_70 <- train_clean_1[train_clean_1$random <= 0.7,]
# train_clean_1_30 <- train_clean_1[train_clean_1$random > 0.7,]
# #train_apca_90 <- train_apca6[train_apca6$random <= 0.9,]
# #train_apca_10<-train_apca6[train_apca6$random > 0.9,]
# 
# #The distribution looks same for both the portions therefore we assume that both the portions 
# #follow similar distribution
# #class(train_clean_1_70$Response)
# #class(train_apca_70$Response)
# library(nnet)
# View(train_clean_1_70)
# ################one hot encoding data
# 
# 
# factorFeatures <- which(sapply(data.all, is.factor))
# df <- data.all[,factorFeatures]
# #View
# #Get one hot encoding of factors
# oneHotData <- data.frame(model.matrix(~ . + 0, data=df, 
#                                       contrasts.arg = lapply(df, 
#                                                              contrasts, contrasts=FALSE)))
# df <- data.all[,-factorFeatures]
# #########3PCA applied on continous
# pc <- princomp(df,cor='TRUE')
# summary(pc)
# plot(pc)
# loadings(pc)
# #var(df)
# #pc$loadings
# optimal_PCA<-13
# #mydata_hat<-predict(pc, as.data.frame(mydata))
# df<-pc$scores[,1:optimal_PCA]
# #View(train_apca)
# #colnames(train_apca6)[grepl("PC",colnames(train_apca6))]<-paste0(i,1:optimal_PCA)
# ####
# #View(train_apca)
# 
# df <- data.frame(df, oneHotData)
# df[is.na(df)] <- -1
# df.train <- df[1:nrow(train),]
# df.test <- df[(nrow(train)+1):nrow(df),]
# str(train)
# ###########
# 
# nodel.mlm <- multinom(Response ~ ., data = df.train)
# 
# summary(train_clean_1_70)
# #str(train_apca_70)
# #summary(nodel.mlm)
# train_apca_30$Prediction<-predict(nodel.mlm, newdata=train_apca_30, type="class")
# kappa_multi<-ScoreQuadraticWeightedKappa(as.numeric(train_apca_30$Prediction),
#                                          as.numeric(train_apca_30$Response))
# auc_multi<-multiclass.roc(train_apca_30$Response,
#                           as.numeric(train_apca_30$Prediction),
#                           levels = levels(factor(train_apca_30$Response)))
# accuracy_multi<-(classificationMetrics(train_apca_30$Response,
#                                        as.numeric(train_apca_30$Prediction)))[1]
# accuracy_multi
# str(auc_multi)
# #### Performance Metrics
# 
# #The Kappa score for this multinominal model is `
# a<-ScoreQuadraticWeightedKappa(as.numeric(train_apca_30$Prediction),
#                                as.numeric(train_apca_30$Response))
# View(a)
# library(pROC)
##################5.3 Random Forest#########################################################

load(file="naive&DT&random.rda")
# Hyper parameter tuning for random forrest
library(ranger)
for (n_trees in seq(500, 5500, by = 1000)){
  for (n_mtry in seq(12, 120, by = 24)){
    set.seed(888)
    model_rf <- ranger(Response ~.
                       , data = train_70
                       , num.trees = n_trees
                       , mtry = n_mtry
                       , importance = "impurity"
                       , write.forest = T
                       , min.node.size = 20
                       , num.threads = 8
                       , verbose = T
    )
    pred_valid <- predict(model_rf,train_30 )
    pred_valid <- predictions(pred_valid)
    pred_valid <- round(pred_valid)
    pred_valid[pred_valid < 1] <- 1
    pred_valid[pred_valid > 8] <- 8
    
    
    kappascore <- ScoreQuadraticWeightedKappa(response_30,pred_valid)
    
    print(paste("kappascore",kappascore ,"ntrees",n_trees,"mtry",n_mtry))
  }
}


### Based on kappa score obtained from hyperparameter tuning final Random Forest Number of trees
##used are 3500 and mtry as 36 


model_rf <- ranger(Response ~.
                   , data = train_70
                   , num.trees = 3500
                   , mtry = 36
                   , importance = "impurity"
                   , write.forest = T
                   , min.node.size = 20
                   , num.threads = 8
                   , verbose = T
)

### Prediction

pred_test <- predict(model_rf,train_30)
pred_test <- predictions(pred_test)
pred_test <- round(pred_test)
pred_test[pred_test < 1] <- 1
pred_test[pred_test > 8] <- 8

### Evaluation
accuracy_rf <- Metrics::accuracy(response_30,pred_test)
kappa_rf <- ScoreQuadraticWeightedKappa(response_30,pred_test)
auc_rf <- Metrics::auc(response_30,pred_test)

### 5.4 XGBoost#####################################################################################################################

library(readr)
library(dplyr)
library(Metrics)
library(caret)
library(xgboost)

#Get columns with type integer
findInt <- function(x){
  id <- is.integer(x)
  return(id)
}

#Get columns with type character
findChar <- function(x){
  id <- is.character(x)
  return(id)
}

#Get factor
getFactors <- function(x, maxn){
  id <- (length(x)<maxn)
}

# #Replace infrequent factor values with most frequent one
# replaceLow <- function(x){
#   if(length(unique(x))>2){
#     tt <- table(x)
#     tt <- sort(tt)
#     if(tt[1]<20){
#       toChange <- as.integer(names(tt)[1])
#       target <- as.integer(names(tt)[length(tt)])
#       x[x==toChange] <- target
#     }
#     
#   }
#   return(x)
# }

#Return Quadratic Weighted Kappa Score
returnScore <- function(truelabels, predictions){
  score <- ScoreQuadraticWeightedKappa(truelabels, predictions)
  return(-score)
}

train = read_csv("train.csv")
test = read_csv("test.csv")
#sample_submission = read_csv("../input/sample_submission.csv")

data.all <- rbind(train[,-ncol(train)], test)
str(data.all)
naCols <- sapply(data.all, is.na)
naCols <- colSums(naCols)/nrow(naCols)
#View(naCols)
getCols <- which(naCols < 0.8)

data.all <- data.all[,getCols]
sum(is.na(data.all))
data.all[is.na(data.all)] <- -1

#Get columns to preprocess
idInt <- which(sapply(data.all, findInt))
unique.int.values <- sapply(data.all[,idInt], unique)


intChars <- which(sapply(unique.int.values, getFactors, 50))
factorFeatures <- names(unique.int.values)[intChars]

data.all.new <- data.all[,factorFeatures]

# 
# #Replace infrequent factor values with most frequent one
# replaceLow <- function(x){
#   if(length(unique(x))>2){
#     tt <- table(x)
#     tt <- sort(tt)
#     if(tt[1]<20){
#       toChange <- as.integer(names(tt)[1])
#       target <- as.integer(names(tt)[length(tt)])
#       x[x==toChange] <- target
#     }
#     
#   }
#   return(x)
# }
length(unique(data.all.new))
#Remove infrequent factor observations
data.all.new <- data.frame(sapply(data.all.new, replaceLow))

#change the columns to factor type
data.all.new <- data.frame(sapply(data.all.new, as.factor))
#summary(data.all.new)
data.all[,factorFeatures] <- data.all.new
#Change product info 2 to factor as well
data.all$Product_Info_2<-as.factor(data.all$Product_Info_2)

factorFeatures <- which(sapply(data.all, is.factor))
df <- data.all[,factorFeatures]
#View
#Get one hot encoding of factors
oneHotData <- data.frame(model.matrix(~ . + 0, data=df, 
                                      contrasts.arg = lapply(df, 
                                                             contrasts, contrasts=FALSE)))
df <- data.all[,-factorFeatures]
df <- data.frame(df, oneHotData)
df[is.na(df)] <- -1
df.train <- df[1:nrow(train),]
df.test <- df[(nrow(train)+1):nrow(df),]

y <- df.train_30$Response
View(y)
testId <- df.test$Id
trainId <- df.train$Id

df.train$Response <- train$Response

# df.test <- df.test[,-1]
# df.train <- df.train[,-1]

df.train$random <- runif(nrow(df.train))
df.train_70 <- df.train[df.train$random <= 0.7,] 
df.train_30 <- df.train[df.train$random > 0.7,] 
y = df.train_70$Response

df.train_70 <- df.train_70[,-340]

xgbact_30 <- df.train_30$Response
df.train_30 <- df.train_30[,-340]

test_clean$random <- runif(nrow(test_clean))
dim(test_clean)


#Overfits to train data
clf1 <- xgboost(data        = data.matrix(df.train_70),
                label       = y,
                eta         = 0.3,
                gamma       = 5,
                depth       = 5,
                nrounds     = 178,
                missing     = -1,
                booster     = "gbtree",
                objective   = "count:poisson",
                eval_metric = "rmse",
                colsample_bytree=1,
                min_child_weight=1,
                subsample=0.5)
#dtrain <- xgb.DMatrix(data = as.matrix(df.train),label = as.matrix(y))
#84 is best for gbtree 
#2372/2872 gblinear 
# params <- list(booster = "gblinear",
#                objective = "reg:linear",
#                eta = 0.3,
#                gamma =0,
#                max_depth= 6,
#                min_child_weight=1,
#                subsample = 1, 
#                colsample_bytree =1)
# 
# xbgcv <- xgb.cv(params = params,
#                 data = dtrain,
#                 nrounds = 3000,
#                 eval_metric = "rmse",
#                 nfold = 5,
#                 showsd = T,
#                 stratified = T,
#                 print_every_n = 50,
#                 early_stopping_rounds = 10,
#                 maximize = F)
#gblinear/reg:linear,0.3,g=0,max_depth=6,minchildwt=1,sub=1,colbytree=1
trainPreds <- predict(clf1, data.matrix(df.train), missing = -1)
trainPreds <- round(trainPreds)
trainPreds[trainPreds < 1] <- 1
trainPreds[trainPreds > 8] <- 8

cat("Get QWK score of train data...")
ScoreQuadraticWeightedKappa(y, testPreds)

cat("Confusion matrix of classes...")
confusionMatrix(testPreds, y)

testPreds <- predict(clf1, data.matrix(df.train_30), missing = -1)
testPreds <- round(testPreds)
testPreds[testPreds < 1] <- 1
testPreds[testPreds > 8] <- 8
library(Metrics)
table(xgbact_30,testPreds)
View(xgbact_30)
accuracy_xg<-accuracy(xgbact_30,testPreds)
#View(trainPreds)
Metrics::precision(xgbact_30,testPreds)
Metrics::recall(xgbact_30,testPreds)
auc_xg<-Metrics::auc(xgbact_30,testPreds)
pROC::multiclass.roc(xgbact_30,as.numeric(testPreds),levels = levels(factor(xgbact_30)))
feature.names <- names(df.train_70)
importance_matrix <- xgb.importance(feature.names, model = clf1)
xgb.plot.importance(importance_matrix[1:20,])

trPreds <- data.frame(trainPreds, y)

testPreds1 <- predict(clf1, data.matrix(df.test), missing = -1)
testPreds1 <- round(testPreds1)
testPreds1[testPreds1 < 1] <- 1
testPreds1[testPreds1 > 8] <- 8

submission <- data.frame("Id" = testId)
submission$Response <- round(testPreds1)

write_csv(submission, "remove_infreq_xgboost3.csv")
save(clf1,file="xgb_insurance1.rda")
#load(file="xgb_insurance.rda")

## **6. Output and Results section####################################################################################**


### **6.1 Performance Metrics**
performance_mat<-data.frame(Algorithm=c("Naive Bayes", "Random Forest", "XGBoost"))
performance_mat$Accuracy<-c(0.528,0.456,0.651)
performance_mat$AUC<-c(0.389,0.598,0.759)
performance_mat$Kappa<-c(0.471,0.589,0.643)
datatable(performance_mat, options = list(initComplete = JS(
  "function(settings, json) {",
  "$(this.api().table().header()).css({'background-color': '#000', 'color': '#fff'});",
  "}")))


plot_ly(data = performance_mat, x=~Algorithm, y=~Accuracy, type = 'scatter', mode = 'lines', name="Accuracy")%>%
  add_trace(y=~AUC, name="AUC")%>%
  add_trace(y=~Kappa, name="Kappa Score")%>%
  layout(title= "Comparision of Preformance Metrics", xaxis=list(title= "Model Type", showgrid=T), yaxis=list(title="Value"))


#Comparing all the three models we find that training the data using xgBoost Algorithim works better
