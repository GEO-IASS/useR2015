library(RSSL)
library(ggplot2)
library(ggthemes)
library(parallel)

if (!file.exists("LearningCurves.RData")) {
datasets <- list("2 Gaussian Expected"=generate2ClassGaussian(n=1000,d=2,expected=TRUE),
                 "2 Gaussian Non-Expected"=generate2ClassGaussian(n=1000,d=2,expected=FALSE))
formulae <- list("2 Gaussian Expected"=formula(Class~.),
                 "2 Gaussian Non-Expected"=formula(Class~.))

classifiers <- list("LS" = function(X,y,X_u,y_u) { LeastSquaresClassifier(X,y)},
                    "ICLS" = function(X,y,X_u,y_u) { ICLeastSquaresClassifier(X,y,X_u)},
                    "EMLS" = function(X,y,X_u,y_u) { EMLeastSquaresClassifier(X,y,X_u)},
                    "SLLS" = function(X,y,X_u,y_u) { SelfLearning(X,y,X_u,method = LeastSquaresClassifier)})


lc <- LearningCurveSSL(formulae,datasets,classifiers,measures = list(Error= measure_error,"Loss test"=measure_losstest),n_l=10,sizes = 2^(0:10),type ="unlabeled",verbose=TRUE,repeats=500,mc.cores=1)
p5 <- plot(lc) 



lc_frac <- LearningCurveSSL(formulae,datasets,classifiers,measures = list(Error= measure_error,"Loss test"=measure_losstest),n_l=10,type ="fraction",verbose=TRUE,repeats=500,test_fraction=0.8,mc.cores=1)

save.image("LearningCurves.RData")
} else {
  load("LearningCurves.RData")
}

ggsave(p5  + theme_fivethirtyeight() + theme(plot.background=element_rect(color="white")) ,filename="Figure5.pdf",width=10,height=6)
ggsave(plot(lc_frac) + theme_fivethirtyeight() + theme(plot.background=element_rect(color="white")),filename="Figure6.pdf",width=10,height=6)
