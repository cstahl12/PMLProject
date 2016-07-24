library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
set.seed(14830)

# use for model training and validation
training_all <- read.csv("pml-training.csv", header = TRUE)
# use for prediction
test <- read.csv("pml-testing.csv", header = TRUE)
inTrain  <- createDataPartition(training_all$classe, p=0.7, list=FALSE)
train <- training_all[inTrain, ]
validate  <- training_all[-inTrain, ]

# Eliminate first 7 columns because they are identification only and provide no predictive value
colnames(train[, 1:7])
train <- train[, -(1:7)]
validate <- validate[, -(1:7)]

# Remove near zero variance in training
train <- train[, -(nearZeroVar(train))]

# Keep only variables with less than 10% NA values
train <- train[, ((sapply(train, function(x) mean(is.na(x)))) < .1)]

# Ensure the columns are consistent
identical(colnames(train), colnames(validate))
colnames(train)

# Decision Tree
model_tree <- rpart(classe ~ ., data=train, method="class")
pred_tree <- predict(model_tree, newdata=train, type="class")
confusionMatrix(pred_tree, train$classe)
pred_tree <- predict(model_tree, newdata=validate, type="class")
confusionMatrix(pred_tree, validate$classe)$overall['Accuracy']


# Pruned Tree
# Select the complexity parameter associated with smallest cross validation error
plotcp(model_tree)
cp <- 0.01
model_ptree <- prune(model_tree, cp=cp)
pred_ptree <- predict(model_ptree, newdata=train, type="class")
confusionMatrix(pred_ptree, train$classe)
pred_ptree <- predict(model_ptree, newdata=validate, type="class")
confusionMatrix(pred_ptree, validate$classe)

# Random Forest
model_rf <- randomForest(classe ~ ., data=train)
pred_rf <- predict(model_rf, newdata=train, type="class")
confusionMatrix(pred_rf, train$classe)
pred_rf <- predict(model_rf, newdata=validate, type="class")
confusionMatrix(pred_rf, validate$classe)

# GBM
model_gbm <- gbm(classe ~ ., data=train, cv.folds=5, distribution="multinomial")
pred_gbm <- predict(model_gbm, train, type="response")
pred_gbm_class <- apply(pred_gbm, 1, which.max)
confusionMatrix(pred_gbm_class, as.numeric(train$classe))

# Predict on the test set for submission
pred_quiz <- predict(model_rf, newdata=test, type="class")
pred_quiz
  