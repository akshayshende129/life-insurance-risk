### Introductory Code
library(forecast)
library(ade4)
library(e1071)
library(ggplot2)
library("rpart")
library("rpart.plot")
library(corrplot)
library(plotly)
library(dplyr)
library(reshape2)
#install.packages("nFactors")
library(nFactors)
library(nnet)
#install.packages("stargazer")
library(stargazer)
library(Metrics)
#install.packages("rattle")
#library(rattle)
library(rpart)
library(randomForest)
library(ggplot2)
library(xgboost)
library(DT)
#install.packages("pROC")
library(pROC)
library(MASS)
#install.packages("performanceEstimation")
library(performanceEstimation)
#require(rattle)
train <- read.csv('train.csv')
test <- read.csv("test.csv")
#str(train)



########## remove variables with excess NAs in both test and train#######

rmNAvars<-function(dat,threshold){
dat<-dat[, -which(colMeans(is.na(dat)) > threshold)]
}

train_clean<-rmNAvars(train,0.3)
#?intersect
test_clean<-test[,intersect(colnames(test), colnames(train_clean))]
#View(test_clean)
################### replacing Missing value with median ###########
sort(colSums(is.na(train_clean)),decreasing = TRUE)
#str(train_clean)
manage_na <- function(datafra)
{
  for(i in 1:ncol(datafra))
  {
    if(is.numeric(datafra[,i]))
    {
      datafra[is.na(datafra[,i]),i] <- median(datafra[!is.na(datafra[,i]),i])
    }
  }
  datafra
}

#####################Function and linear model



train_clean <- manage_na(train_clean)
test_clean <- manage_na(test)
#levels(train_clean$Medical_Keyword_46)
###### converting nonnumeric column to numeric ####
train_clean[,!(sapply(train_clean,class) == "numeric" | sapply(train_clean, class) == "integer")]<-
  as.numeric(train_clean[,
                         !(sapply(train_clean, class) == "numeric" | sapply(train_clean, class) == "integer")])
str(train_clean)

test_clean[,!(sapply(train_clean,class) == "numeric" | sapply(test_clean, class) == "integer")]<-
  as.numeric(test_clean[,
                        !(sapply(test_clean, class) == "numeric" | sapply(test_clean, class) == "integer")])



## **1.	Introduction**

#In a one-click shopping world, the life insurance application process is antiquated. 
#Customers provide extensive information to identify risk classification and eligibility, 
#including scheduling medical exams, a process that takes an average of 30 days for the purchase of 
#an Insurance product. Hence, only 40% of U.S. households own individual life insurance. 
#We aim to make the process of issuing life insurance quicker and less labor intensive for new and existing
#customers to get a quote while maintaining privacy boundaries. The model aims to automate the process
#of risk assessment for issue of various insurance products. This model will help the firm to generate
#more revenues by optimally targeting more profitable and less risky customers. The data set used includes
#the insurance company's earlier data-set that contains various parameters of an insurance application along
#with a risk score computed by the company's earlier internal model. The results will to better understand
#the predictive power of the data points in the existing assessment, enabling streamlining the process.

## **2.	Data used**

#The model data set is pre-separated among training and test sample in the ratio of 3:1 and
#have a random sampling being done. The train data set is a transactional (low level) data consists of 
#59,381 customers and 126 variables as predictors. The test dataset contains the same variables for another
#set of 19,766 customers. The variables are categorized based on the type of information they provide. 
#Due to proprietary reasons the variable names have been masked, however they have been numbered within 
#the category type for identification. The categories of variables present are: 


#* Product Information
#* Insurance Age
#* Height
#* Weight 
#* BMI
#* Employment Information
#* Insured Information
#* Insurance History
#* Family History
#* Medical History
#* Medical Keyword

#Each of these variable categories contain multiple variables they represent different items under 
#that variable class.

## **3. Explanatory Data Analysis**

### **3.1	Variable Types**

#A primary data analysis was performed through visual inspection of the training data-set to identify
#the different types of variables among Continuous, Categorical (Nominal and Ordinal) and
#Dummy (Binary/Indicator) variables. This analysis helps us identify the choice of variable selection
#and reduction algorithms in the next stage of modelling. 
# How did you identify those are categorical and continuous 

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
#View(unique.int.values)


intChars <- which(sapply(unique.int.values, getFactors, 50))
#View(intChars)
factorFeatures <- names(unique.int.values)[intChars]
#View(factorFeatures)
data.all.new <- data.all[,factorFeatures]
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
#length(unique(data.all.new))
#Remove infrequent factor observations
#data.all.new <- data.frame(sapply(data.all.new, replaceLow))

#change the columns to factor type
data.all.new <- data.frame(sapply(data.all.new, as.factor))
#summary(data.all.new)
data.all[,factorFeatures] <- data.all.new
#Change product info 2 to factor as well
data.all$Product_Info_2<-as.factor(data.all$Product_Info_2)
factorFeatures <- which(sapply(data.all, is.factor))
#View(factorFeatures)
df <- data.all[,factorFeatures]
#View(df)
#summary(df)
temp1<- data.frame(Variable_Type = c("Product Information",
                                     "Insurance Age",
                                     "Height",
                                     "Weight", 
                                     "BMI",
                                     "Employment Information",
                                     "Insured Information",
                                     "Insurance History",
                                     "Family History",
                                     "Medical History",
                                     "Medical Keyword"))
#str(data.all.new)
temp1$Continous<-c(1,1,1,1,1,3,0,1,4,0,0)
temp1$Categorical<-c(6,0,0,0,0,3,7,8,1,41,0)
temp1$Dummy<-c(0,0,0,0,0,0,0,0,0,0,48)
temp1$Total<-rowSums(temp1[,-1])
temp1[12,2:5]<-colSums(temp1[,-1])
temp1$Variable_Type[12]<-"Total"
temp1$Continous
temp1$Variable_Type
temp1$Variable_Type
temp2<-temp1[,2:5]
temp2[12]<-"Total"

datatable(temp2, options = list(initComplete = JS(
  "function(settings, json) {",
  "$(this.api().table().header()).css({'background-color': '#000', 'color': '#fff'});",
  "}")))
#View(temp1)
#str(temp1)
##Continuous variables are analyzed using summary statistics,box plots and density plots. 
##The categorical variables are analyzed using event rate chart to track the variation to the response. 


### **3.2 Histogram of Response plot**


#The response is a nominal variable with levels from 1 to 8 and associates to the risk level of a customer.


p<-ggplot(train, aes(x=Response))+ geom_histogram(fill="Red", alpha=0.3)
#colorwise needed for different colors
p
ggplotly(p, color=~Response, width = 800, height = 400)%>%
  layout(title="Distribution of Response Variable", 
         plot_bgcolor= "white", 
         xaxis=list(gridcolor="lightgrey", opacity=0.5), 
         yaxis=list(gridcolor="lightgrey",opacity = 0.5),
         autosize = T, width = 800, height = 400)



#While it is not mentioned whether the scale is in increasing order of riskiness or otherwise, 
#from the distribution of the response variable we can infer that 8 could possibly refer to less 
#risky customers

missing_prct<-data.frame(variable=colnames(train),
                         missing=sapply(train,
                                        function(x){
                                          sum(is.na(x))
                                        }/nrow(train)
                         )
)
missing_prct_test<-data.frame(variable=colnames(test),
                              missing=sapply(test,
                                             function(x){
                                               sum(is.na(x))
                                             }/nrow(test)
                              )
)
missing_prct1<-missing_prct[missing_prct$missing>0,]
datatable(missing_prct1, options = list(initComplete = JS(
  "function(settings, json) {",
  "$(this.api().table().header()).css({'background-color': '#000', 'color': '#fff'});",
  "}")))
#View(missing_prct)
#missing_prct_test

### **3.3	Summary Statistics**

#To allow for easier convergence of machine learning algorithms variables are normalized to the 
#range of [0, 1]. The most common normalizing function used is given below:
#X_{norm} = frac{x_{i}-x_{min}}{x_{max}-x_{min}}

#Ans for why normalization and not standardization
#1. normlization is for normalizing the data into [0,1]
#2. it is not negative so some variables like age,bmi shouldnot be negative
#3. standardization is 0 mean and std dev 1 
#4. it considers -1 to 1 value but it is used generally where we dont want to lose outliers information
#5. but standardization is is used which varies a lot 
#The same function had been applied to the continuous variables in the input data-set. 
#The summary statistics help understand the distribution of the underlying dataset,
#the box plots and density plots enable visualizing the data-set
## Generating Summary Table
#str(train_clean)
train_conti<-train_clean[,c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI",
                            "Employment_Info_1", "Employment_Info_4", "Employment_Info_6")] 

summ_conti<-data.frame(Variables =  colnames(train_conti))
summ_conti$Min<-apply(train_conti,2,function(x){min(x, na.rm = T)})
summ_conti$Max<-apply(train_conti,2,function(x){max(x, na.rm = T)})
summ_conti$Mean<-apply(train_conti,2,function(x){mean(x, na.rm = T)})
summ_conti$Median<-apply(train_conti,2,function(x){median(x, na.rm = T)})
datatable(summ_conti, options = list(initComplete = JS(
  "function(settings, json) {",
  "$(this.api().table().header()).css({'background-color': '#000', 'color': '#fff'});",
  "}")))




### **3.4 Continuous Variable Analysis** 

#### Box Plots
#The box plots enable visualization of the data-set especially in relation to outliers. 
#However considering the large number of data

#temp1<-train_conti[1:5]

p<-plot_ly( data=melt(temp1), type = "box",
            split = ~variable,y = ~value)%>%
  layout( title = "Box-Plots of variables")
head(temp1)
p
#plt[[1]] <- as_widget(p)
p<-plot_ly( data=melt(train_conti[,6:length(train_conti)]), type = "box",
            split = ~variable,y = ~value)%>%
  layout( title = "Box-Plots of variables")
p
#plt[[2]] <- as_widget(p)
#plt
#### Density Plot
#View(train_conti$Employment_Info_4)
#temp1$Ht


#The density plots help visualize the characteristics of the distribution 
#including statistical metrics such as mean, standard deviation and kurtosis. 
#It also enables us to visually identify if any relationship exists with the response variable.
#For example: The density plot of variable Employment_Info_6 is similar to the histogram of the 
#response variable, this probably indicated that this variable could be a good predictor of the 
#response variable
#train_conti[,1:2]
temp_melt<-melt(train_conti[,1:2])
#str(train_conti)
p1<-ggplot(temp_melt,aes(value, fill = variable ))+geom_density(alpha = 0.5)+ggtitle("Density Plots")
p1
ggplotly(p1, height= 800, width = 1000)%>%
  layout(plot_bgcolor="transparent",paper_bgcolor= "transparent",autosize = F, width = 1000, height = 800)
temp_melt<-melt(train_conti[,c(3,4,5)])

p2<-ggplot(temp_melt,aes(value, fill = variable ))+geom_density(alpha = 0.5)+ggtitle("Density Plots")
p2
ggplotly(p2, height= 800, width = 1000)%>%
  layout(plot_bgcolor="transparent",paper_bgcolor= "transparent",autosize = F, width = 1000, height = 800)
temp_melt<-melt(train_conti[,c(6,8)])
p3<-ggplot(temp_melt,aes(value, fill = variable ))+geom_density(alpha = 0.5)+ggtitle("Density Plots")
p3
ggplotly(p3, height= 800, width = 1000)%>%
  layout(plot_bgcolor="transparent",paper_bgcolor= "transparent",autosize = F, width = 1000, height = 800)
temp_melt<-melt(train_conti[,7])
temp_melt$variable<-"Employment_Info_4"
p4<-ggplot(temp_melt,aes(value, fill = variable ))+geom_density(alpha = 0.5)+ggtitle("Density Plots")
p4
ggplotly(p4, height= 800, width = 1000)%>%
  layout(plot_bgcolor="transparent",paper_bgcolor= "transparent",autosize = F, width = 1000, height = 800)

### **3.5 Missing Value Analysis** 


#### Missing Value  Plots {.tabset #missing1}

#Missing value percentage charts evaluate whether the variable has sufficient 
#number of data records for predictions. The plot presents the percentage of observations 
#missing for each variable

var_kind<-c("Product_Info_", "Ins_Age", "Ht", "Wt","BMI","Employment_Info_","InsuredInfo_",
            "Insurance_History_", "Family_Hist_","Medical_History_", "Medical_Keyword_")

par(mfrow=c(2,2))
for(i in var_kind){
  plot(x=as.factor(as.character(missing_prct[grep(i, row.names(missing_prct)),1])),
       y=missing_prct$missing[grep(i, row.names(missing_prct))], ylim = c(0,1),
       main =gsub("_"," ",i))
  
}


#### Missing vs Response Chart


#This chart enables us to identify whether variables with a 
#high percentage of missing values actually help in predicting the response variable. 
#The distribution missing values for each response category has been retained within the 
#variable and hence the missing values are random in nature.


train_na_response <- sapply(sort(unique(train$Response)), function(x) { 
                            apply(train[train$Response == x, ], 2, function(y) { sum(is.na(y)) }) })
train_na_response<-data.frame(train_na_response)
train_na_response<-train_na_response[which(rowSums(train_na_response)>0),]
train_na_response$ID<-rownames(train_na_response)
train_na_response_melt<-melt(train_na_response)
plot_ly(train_na_response_melt, x = ~ID, y =~value , color = ~variable)%>%
  layout(title ="Missing vs Response Chart")


### **3.6 Event Rate Chart**
#In an attempt to capture the conditional probability of the response given a specific bin 
#of the categorical variable 
#P(y=1|ProdInfo_2= A_1)=\frac{P(y=1  \cap  ProdInfo_2= A_1  )}{P(ProdInfo_2= A_1)}


#### Product Information
train_categ<-train_clean[,-which(colnames(train_clean) %in% colnames(train_conti))]
i="Product_Info"
train_temp<-train_categ[,grep(i,colnames(train_categ))]
index<-1
plt<-htmltools::tagList()
for (i in colnames(train_temp)){
  data_freq<-as.data.frame(table(train_temp[,i],train_clean$Response)/(as.data.frame(table(train_temp[,i]))[,2]))
  p<-plot_ly(data_freq, x = ~Var1, y = ~Freq, color = ~Var2, type="bar")%>%
    layout(title = paste0("Event Rate Chart- ",gsub("_"," ",i)),
           xaxis = list(title = gsub("_"," ",i),showgrid = T))
  plt[[index]] <- as_widget(p)
  index <- index + 1
}
#plt
p


#### Employment Information



i="Employment_Info"
train_temp<-train_categ[,grep(i,colnames(train_categ))]
index<-1
plt<-htmltools::tagList()
for (i in colnames(train_temp)){
  data_freq<-as.data.frame(table(train_temp[,i],train_clean$Response)/(as.data.frame(table(train_temp[,i]))[,2]))
  p<-plot_ly(data_freq, x = ~Var1, y = ~Freq, color = ~Var2, type="bar")%>%
    layout(title = paste0("Event Rate Chart- ",gsub("_"," ",i)),
           xaxis = list(title = gsub("_"," ",i),showgrid = T))
  plt[[index]] <- as_widget(p)
  index <- index + 1
}
p


#### Insured Information

i="InsuredInfo"
train_temp<-train_categ[,grep(i,colnames(train_categ))]
index<-1
plt<-htmltools::tagList()
for (i in colnames(train_temp)){
  data_freq<-as.data.frame(table(train_temp[,i],train_clean$Response)/(as.data.frame(table(train_temp[,i]))[,2]))
  p<-plot_ly(data_freq, x = ~Var1, y = ~Freq, color = ~Var2, type="bar")%>%
    layout(title = paste0("Event Rate Chart- ",gsub("_"," ",i)),
           xaxis = list(title = gsub("_"," ",i),showgrid = T))
  plt[[index]] <- as_widget(p)
  index <- index + 1
}
p



#### Insurance History

i="Insurance_History"
train_temp<-train_categ[,grep(i,colnames(train_categ))]
index<-1
plt<-htmltools::tagList()
for (i in colnames(train_temp)){
  data_freq<-as.data.frame(table(train_temp[,i],train_clean$Response)/(as.data.frame(table(train_temp[,i]))[,2]))
  p<-plot_ly(data_freq, x = ~Var1, y = ~Freq, color = ~Var2, type="bar")%>%
    layout(title = paste0("Event Rate Chart- ",gsub("_"," ",i)),
           xaxis = list(title = gsub("_"," ",i),showgrid = T))
  plt[[index]] <- as_widget(p)
  index <- index + 1
}
#p



#### Medical History

par(mfrow=c(2,2))  
i="Medical_History"
train_temp<-train_categ[,grep(i,colnames(train_categ))]
index<-1
plt<-htmltools::tagList()
for (i in colnames(train_temp)){
  data_freq<-as.data.frame(table(train_temp[,i],train_clean$Response)/(as.data.frame(table(train_temp[,i]))[,2]))
  p<-plot_ly(data_freq, x = ~Var1, y = ~Freq, color = ~Var2, type="bar")%>%
    layout(title = paste0("Event Rate Chart- ",gsub("_"," ",i)),
           xaxis = list(title = gsub("_"," ",i),showgrid = T))
  plt[[index]] <- as_widget(p)
  index <- index + 1
}
p



#### Medical Keyword


i="Medical_Keyword"
train_temp<-train_categ[,grep(i,colnames(train_categ))]
index<-1
plt<-htmltools::tagList()
for (i in colnames(train_temp)){
  data_freq<-as.data.frame(table(train_temp[,i],train_clean$Response)/(as.data.frame(table(train_temp[,i]))[,2]))
  p<-plot_ly(data_freq, x = ~Var1, y = ~Freq, color = ~Var2, type="bar")%>%
    layout(title = paste0("Event Rate Chart- ",gsub("_"," ",i)),
           xaxis = list(title = gsub("_"," ",i),showgrid = T))
  plt[[index]] <- as_widget(p)
  index <- index + 1
}
#p


### **3.7 Correlation Plots** {.tabset .tabset-fade #corr}
#After data analysis and data treatment, 
#the next stage of model development would be variable reduction. 
#### Product Information

i="Product_Info"
if(class(train_clean[,grep(i, colnames(train_clean))])=="data.frame"){
  m<-cor(train_clean[,c(grep(i, colnames(train_clean)),119)])
  corrplot(m, method = "number", type="lower")
}
#?corrplot
#corrplot(train_conti,method = "circle")
#### Employment Information

i="Employment_Info"
if(class(train_clean[,grep(i, colnames(train_clean))])=="data.frame"){
  m<-cor(train_clean[,c(grep(i, colnames(train_clean)),119)])
  corrplot(m, method = "number", type="lower")
}



#### Insured Information

i="InsuredInfo_"
if(class(train_clean[,grep(i, colnames(train_clean))])=="data.frame"){
  m<-cor(train_clean[,c(grep(i, colnames(train_clean)),119)])
  corrplot(m, method = "number", type="lower")
}




#### Insurance History

i="Insurance_History"
if(class(train_clean[,grep(i, colnames(train_clean))])=="data.frame"){
  m<-cor(train_clean[,c(grep(i, colnames(train_clean)),119)])
  corrplot(m, method = "number", type="lower")
}


#### Medical History

i="Medical_History"
if(class(train_clean[,grep(i, colnames(train_clean))])=="data.frame"){
  m<-cor(train_clean[,c(grep(i, colnames(train_clean)),119)])
  corrplot(m, method = "number", type="lower")
}


### Medical Keyword

i="Medical_Keyword"
if(class(train_clean[,grep(i, colnames(train_clean))])=="data.frame"){
  m<-cor(train_clean[,c(grep(i, colnames(train_clean)),118)])
  corrplot(m, method = "number", type="lower")
}
## **4. Variable treatment**

### **4.1 Missing data treatment**

#As a preliminary step in data treatment, variables that have a high percentage
#of missing values are removed. While the threshold for removal is user determined, 
#for this exercise the threshold was 30%.

### **4.2 Missing Value treatment**

#For the variables that are not dropped at the previous step of modeling,
#variables that have missing values in lesser percentages are imputed. 
#The methodology used for imputation is using median of the remaining data series. 
#This is a commonly used industry practice and is efficient as the missing data for all variables 
#is randomly distributed over the response variable.


#Here as well we start by dividing the data inot two portions and using one portion to predict
#the model and the other portion for model validation and insample prediction.
#We will be using the entire data set for prediction.


### **4.3 PCA** {.tabset .tabset-fade #PCA}




#####################Tried by Ganesh #########################
#### Product Information
train_apca<-data.frame(ID=train_clean$Id)
#View(train_apca)
i="Product_Info"
mydata<-train_clean[,grep(i, colnames(train_clean))]
pc <- princomp(mydata,cor='TRUE')
summary(pc)
plot(pc)
View(pc$scores)
loadings(pc)
var(mydata)
pc$loadings###95 % data 
optimal_PCA<-6
#View(pc$scores[,1:6])
#mydata_hat<-predict(pc, as.data.frame(mydata))
#View(mydata_hat)
train_apca1<-cbind(train_apca,pc$scores[,1:optimal_PCA])
colnames(train_apca1)[grepl("PC",colnames(train_apca1))]<-paste0(i,1:optimal_PCA)
View(train_apca1)
#plotnScree(nS)


#### Employment Information
#train_apca<-data.frame(ID=train_clean$Id)
#View(train_apca)
i="Employment_Info"
mydata<-train_clean[,grep(i, colnames(train_clean))]
pc <- princomp(mydata,cor='TRUE')
#summary(pc)

#pc$scores
plot(pc)
#loadings(pc)
#var(mydata)
#pc$loadings
optimal_PCA<-5
#mydata_hat<-predict(pc, as.data.frame(mydata))
train_apca2<-cbind(train_apca1,pc$scores[,1:optimal_PCA])
#View(train_apca)
colnames(train_apca2)[grepl("PC",colnames(train_apca2))]<-paste0(i,1:optimal_PCA)


#### Insured Information
#train_apca<-data.frame(ID=train_clean$Id)
#View(train_apca)
i="InsuredInfo_"
mydata<-train_clean[,grep(i, colnames(train_clean))]
pc <- princomp(mydata,cor='TRUE')
#summary(pc)
plot(pc)
#loadings(pc)
#var(mydata)
#pc$loadings
optimal_PCA<-6
#mydata_hat<-predict(pc, as.data.frame(mydata))
train_apca3<-cbind(train_apca2,pc$scores[,1:optimal_PCA])
#View(train_apca)
colnames(train_apca3)[grepl("PC",colnames(train_apca3))]<-paste0(i,1:optimal_PCA)

#### Insurance History
#train_apca<-data.frame(ID=train_clean$Id)
#View(train_apca)
i="Insurance_History"
mydata<-train_clean[,grep(i, colnames(train_clean))]
pc <- princomp(mydata,cor='TRUE')
#summary(pc)
plot(pc)
#loadings(pc)
#var(mydata)
#pc$loadings
optimal_PCA<-6
#mydata_hat<-predict(pc, as.data.frame(mydata))
train_apca4<-cbind(train_apca3,pc$scores[,1:optimal_PCA])
#View(train_apca)
colnames(train_apca4)[grepl("PC",colnames(train_apca4))]<-paste0(i,1:optimal_PCA)

#### Medical History
i="Medical_History"
#train_apca<-data.frame(ID=train_clean$Id)
#View(train_apca)

mydata<-train_clean[,grep(i, colnames(train_clean))]
pc <- princomp(mydata,cor='TRUE')
#summary(pc)
#plot(pc)
#loadings(pc)
#var(mydata)
#pc$loadings
optimal_PCA<-35
#mydata_hat<-predict(pc, as.data.frame(mydata))
train_apca5<-cbind(train_apca4,pc$scores[,1:optimal_PCA])
#View(train_apca)
colnames(train_apca5)[grepl("PC",colnames(train_apca5))]<-paste0(i,1:optimal_PCA)

#### Medical Keyword
i="Medical_Keyword"
#train_apca<-data.frame(ID=train_clean$Id)
#View(train_apca)
mydata<-train_clean[,grep(i, colnames(train_clean))]
pc <- princomp(mydata,cor='TRUE')
#summary(pc)
#plot(pc)
#loadings(pc)
#var(mydata)
#pc$loadings
optimal_PCA<-46
#mydata_hat<-predict(pc, as.data.frame(mydata))
train_apca6<-cbind(train_apca5,pc$scores[,1:optimal_PCA])
#View(train_apca)
colnames(train_apca6)[grepl("PC",colnames(train_apca6))]<-paste0(i,1:optimal_PCA)
####
View(train_apca6)
Response<-train_clean$Response
train_apca6<-cbind(train_apca6,Response)
train_apca6$random <- runif(nrow(train_apca6))
train_apca6_70 <- train_apca6[train_apca6$random <= 0.7,] 
train_apca6_30 <- train_apca6[train_apca6$random > 0.7,]
train_apca6<-train_apca6[,-c(106,107)]

#model1<-lm(Response~.,data=train_apca6_70)
#summary(model1)
temp1<- data.frame(Variable_Type = c("Product Information",
                                     "Insurance Age",
                                     "Height",
                                     "Weight", 
                                     "BMI",
                                     "Employment Information",
                                     "Insured Information",
                                     "Insurance History",
                                     "Family History",
                                     "Medical History",
                                     "Medical Keyword"))
names(train_apca6)
temp1$Prior_to_PCA<-c(7,1,1,1,1,6,7,7,1,37,48)
temp1$After_PCA<-c(6,1,1,1,1,5,6,6,1,35,46)
colSums(temp1[1,2,drop=FALSE])
temp1[12,2:3]<-colSums(temp1[,-1])
temp1$Variable_Type[12]<-"Total"
datatable(temp1, options = list(pageLength = 13,
                                initComplete = JS(
                                  "function(settings, json) {",
                                  "$(this.api().table().header()).css({'background-color': '#000', 'color': '#fff'});",
                                  "}")))
library(DT)
#**The dimension of data of post PCA is reduced from `r length(train_clean[,1])` * `r length(train_clean)`   to    `r length(train_apca[,1])` * `r length(train_apca)`**
save(data.all,data.all.new,df,df.train,df.test,test,train,train_clean,train_conti,file = "Preprocessdata1.rda")
getwd()
load(file = "Preprocessdata1.rda")
