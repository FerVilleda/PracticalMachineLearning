#proyecto Practical Machine Learning
#predict the manner in which they did the exercise.
#liberias
library(readr)
library(caret)
library(rattle)
library(ggplot2)

#Importando Data sets
pml_testing <- read_csv("pml-testing.csv")
pml_training <- read_csv("pml-training.csv")

#estructura y analisis inicial de los datos
str(pml_training)
dim(pml_training)
classes <- as.factor(pml_training$classe)
levels(classes)
prev1 <- ggplot(pml_training, aes(x=classe, fill=classe)) + geom_bar() + scale_fill_brewer(palette = "Pastel1")
prev1

#Crear datos de validacion para el analisis del error
Validationpml <- pml_testing
inTrain  <- createDataPartition(pml_training$classe, p=0.7, list=FALSE)
Trainpml <- pml_training[inTrain, ]
Testpml <- pml_training[-inTrain, ]

#quitar variables identificadoras 155 
Trainpml <- Trainpml[, -(1:5)]
Testpml  <- Testpml[, -(1:5)]
Validationpml <- Validationpml[,-(1:5)]

#Quitar variables de varianza cercana a 0 122
nearZV <- nearZeroVar(Trainpml)
Trainpml <- Trainpml[, -nearZV]
Testpml  <- Testpml[, -nearZV]
Validationpml <- Validationpml[,-nearZV]

#54 variables
mayorityNA    <- sapply(Trainpml, function(x) mean(is.na(x))) > 0.95
Trainpml <- Trainpml[, mayorityNA==FALSE]
Testpml  <- Testpml[, mayorityNA==FALSE]
Validationpml <- Validationpml[,mayorityNA==FALSE]

prev2 <- ggplot(Trainpml, aes(x=classe, fill=classe)) + geom_bar() + scale_fill_brewer(palette = "Pastel1")
prev2


#Empezar a entrenar modelos
#Arbol de decision
set.seed(324)
modDTree <- train(classe ~., method="rpart", data=Trainpml)
print(modDTree$finalModel)
fancyRpartPlot(modDTree$finalModel)
predDtree <- predict(modDTree,newdata=Testpml)
conMDTree <- confusionMatrix(as.factor(Testpml$classe),predDtree)
conMDTree

#Random Forest
controlRFor <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRFor <- train(classe ~., data=Trainpml, method="rf", trControl=controlRFor)
predRFor <- predict(modRFor,newdata = Testpml)
conMRFor <- confusionMatrix(as.factor(Testpml$classe),predRFor)
#mostrar el arbol elegido
conMRFor
modRFor$results[2,]


#boosting
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modGBM <- train(classe ~., data=Trainpml, method="gbm", verbose=FALSE, trControl = controlGBM)
predGBM <- predict(modGBM, newdata = Testpml)
conMGBM <- confusionMatrix(as.factor(Testpml$classe),predGBM)


conMDTree$overall[1]
conMRFor$overall[1]
conMGBM$overall[1]

#el modelo con mayor accuracy es Random Forest
#aplicar modelo en el conjunto de validacion 
predFinal <- predict(modRFor, newdata=Validationpml)
predDF <- data.frame(Validationpml,classe=predFinal)
prev3 <- ggplot(predDF, aes(x=classe, fill=classe)) + geom_bar() + scale_fill_brewer(palette = "Pastel1")
prev3
