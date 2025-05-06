library(Rcpp)
library(RcppArmadillo)
sourceCpp("BFGS_functions.cpp")
sourceCpp("IRLS_functions.cpp")
# set a randomization seed

set.seed(735)

# load heard disease data (add your own filepath to access the data)
heart.data <- read.csv("data/heart_2020_cleaned.csv") 



## data pre-processing
# set factors as factors
heart.data$HeartDisease <- as.factor(heart.data$HeartDisease)
heart.data$Smoking <- as.factor(heart.data$Smoking)
heart.data$AlcoholDrinking <- as.factor(heart.data$AlcoholDrinking)
heart.data$Stroke <- as.factor(heart.data$Stroke)
heart.data$DiffWalking <- as.factor(heart.data$DiffWalking)
heart.data$Sex <- as.factor(heart.data$Sex)
heart.data$AgeCategory <- as.factor(heart.data$AgeCategory)
heart.data$Race <- as.factor(heart.data$Race)
heart.data$Diabetic <- as.factor(heart.data$Diabetic)
heart.data$PhysicalActivity <- as.factor(heart.data$PhysicalActivity)
heart.data$GenHealth <- as.factor(heart.data$GenHealth)
heart.data$Asthma <- as.factor(heart.data$Asthma)
heart.data$KidneyDisease <- as.factor(heart.data$KidneyDisease)
heart.data$SkinCancer <- as.factor(heart.data$SkinCancer)

# set numerics as numeric
heart.data$BMI <- as.numeric(heart.data$BMI)
heart.data$PhysicalHealth <- as.numeric(heart.data$PhysicalHealth)
heart.data$MentalHealth <- as.numeric(heart.data$MentalHealth)
heart.data$SleepTime <- as.numeric(heart.data$SleepTime)



### Model 1 ###
# fit logistic regression model on heart.train (all covariates)
# make a design matrix and the response vector with zeros and ones
Y <- model.matrix(~., data=heart.data)[,2]
X <- model.matrix(~., data=heart.data)[,-2]


# Set different starting value
beta0 <- matrix(0, ncol = 1, nrow = ncol(X))
beta02 <- matrix(0.5, ncol = 1, nrow = ncol(X))

# IRLS
start <- Sys.time()
model.irls.1 <- optim_irls(X=X, Y=Y, beta=beta0)
end <- Sys.time()
(time.elapsed <- end-start)

## IRLS diverges when the starting of 0.5
optim_irls(X=X, Y=Y, beta=beta02)

# BFGS
start <- Sys.time()
model.bfgs.1 <- optim_bfgs(X=X, Y=Y, beta=beta0)
end <- Sys.time()
(time.elapsed <- end-start)

## BFGS converges when the starting of 0.5
optim_bfgs(X=X, Y=Y, beta=beta02)

# glm function
start <- Sys.time()
model.logit.1 <- glm(HeartDisease ~ ., family = "binomial", data = heart.data)
end <- Sys.time()
(time.elapsed <- end-start)
# print logistic regression model parameters
summary(model.logit.1)

# Compare the results from the three functions
est_logistic <- data.frame("GLM"=model.logit.1$coefficients, "IRLS"=model.irls.1$Estimate$Beta, "BFGS"=model.bfgs.1$Estimate$Beta)
row.names(est_logistic) <- names(model.logit.1$coefficients)
est_logistic_rounded <- round(est_logistic, digits = 4)

# Print the result.
est_logistic_rounded