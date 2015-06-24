library(RSSL)
library(magrittr)
library(ggplot2)
library(ggthemes)
library(extrafont)
library(knitr)
loadfonts()

set.seed(11)
data_2gauss <- data.frame(generate2ClassGaussian(n=500,d=2,var=0.2,expected=FALSE)) %>% 
  add_missinglabels_mar(formula=Class~.,prob=0.98)
problem_2gauss <-  data_2gauss %>% df_to_matrices(Class~.)

problem1 <- problem_2gauss

p1 <- ggplot(data_2gauss,aes(x=X1,y=X2,shape=Class,color=Class)) +
  geom_point(aes(size=(!is.na(Class))),alpha=0.8) +
  coord_equal() +
  theme_tufte(base_family = "sans",base_size = 18) +
  scale_shape_stata(na.value=16) +
  scale_color_colorblind(na.value="grey") +
  scale_size_manual(values=c(3,7),guide=FALSE) +
  theme(axis.title.y=element_text(angle = 0, hjust = 0)) +
  scale_linetype_stata() +
  labs(y="",x="")
print(p1)


g_lda <- LinearDiscriminantClassifier(problem1$X,problem1$y)
g_emlda <- EMLinearDiscriminantClassifier(problem1$X,problem1$y,problem1$X_u)
g_sllda <- SelfLearning(problem1$X,problem1$y,problem1$X_u,LinearDiscriminantClassifier)

p1 + geom_classifier("LDA"=g_lda,"EMLDA"=g_emlda,"SLLDA"=g_sllda) + guides(colour = guide_legend(override.aes = list(size=6,linetype=0)))

ggsave(p1 + geom_classifier("LDA"=g_lda,"EMLDA"=g_emlda,"SLLDA"=g_sllda) + guides(colour = guide_legend(override.aes = list(size=6,linetype=0))), filename="Figure1.pdf")

# Figure 2

set.seed(11)
data_slicedcookie <- data.frame(generateSlicedCookie(200,expected=TRUE)) %>% 
  add_missinglabels_mar(prob=0.98,formula=Class~.)
problem_slicedcookie <- data_slicedcookie %>% df_to_matrices

problem2 <- problem_slicedcookie

p2 <- ggplot(data_slicedcookie,aes(x=X1,y=X2,shape=Class,color=Class)) +
  geom_point(aes(size=(!is.na(Class))),alpha=0.8) +
  coord_equal() +
  theme_tufte(base_family = "sans",base_size = 18) +
  scale_shape_stata(na.value=16) +
  scale_color_colorblind(na.value="grey") +
  scale_size_manual(values=c(3,7),guide=FALSE) +
  theme(axis.title.y=element_text(angle = 0, hjust = 0)) +
  scale_linetype_stata() +
  labs(y="",x="")
print(p2)

g_svm <- LinearSVM(problem2$X,problem2$y,scale=TRUE)
g_tsvm <- TSVMcccp_lin(problem2$X,problem2$y,problem2$X_u,C=1,Cstar=1)

p2 + geom_classifier("SVM"=g_svm,"TSVM"=g_tsvm) + guides(colour = guide_legend(override.aes = list(size=6,linetype=0))) + ggtitle("Low Density Separation Assumption")

ggsave(p2 + geom_classifier("SVM"=g_svm,"TSVM"=g_tsvm) + guides(colour = guide_legend(override.aes = list(size=6,linetype=0))),filename = "Figure2.pdf")

# Figure 3
set.seed(16)
data_2circles <- data.frame(generateTwoCircles(500,noise_var = 0.05)) %>% 
  add_missinglabels_mar(prob=0.99,formula=Class~.)
problem3 <- data_2circles %>% df_to_matrices

p3 <- ggplot(data_2circles,aes(x=X1,y=X2,color=Class)) +
  geom_point(aes(size=(!is.na(Class)),shape=Class),alpha=0.8) +
  coord_equal() +
  theme_tufte(base_family = "sans",base_size = 18) +
  scale_shape_stata(na.value=16,guide=FALSE) +
  scale_color_colorblind(na.value="grey",guide=FALSE) +
  scale_size_manual(values=c(3,7),guide=FALSE) +
  theme(axis.title.y=element_text(angle = 0, hjust = 0)) +
  scale_linetype_stata(name="") +
  labs(y="",x="") +
  ylim(c(-3,3)) +
  xlim(-3,3)
print(p3)


g_ls <- LeastSquaresClassifier(problem3$X,problem3$y,scale=TRUE,y_scale=TRUE)
g_lda <- LinearDiscriminantClassifier(problem3$X,problem3$y)

g_krls <- KernelLeastSquaresClassifier(problem3$X,problem3$y,lambda=0.00001,kernel=kernlab::rbfdot(0.1))
Xt <- as.matrix(expand.grid(X1=seq(-3,3,length.out = 100),X2=seq(-3,3,length.out = 100)))
data.frame(Xt,pred=as.numeric(decisionvalues(g_krls,Xt))) %>% 
  ggplot() + geom_contour(aes(x=X1,y=X2,z=pred))

p3 + geom_contour(aes(x=X1,y=X2,z=pred),
                  breaks=c(0.5),
                  data=data.frame(Class="l",
                                  Xt,
                                  pred=as.numeric(decisionvalues(g_krls,Xt)))) +   geom_classifier(g_ls)

g_lap <- problem3 %>% {LaplacianKernelLeastSquaresClassifier(.$X,.$y,.$X_u,lambda=0.01,kernel=kernlab::rbfdot(10),gamma=10000000,adjacency_kernel = kernlab::rbfdot(100))}

p3 <- p3 + geom_contour(aes(x=X1,y=X2,z=pred,linetype=Classifier),
                  color="black",
                  breaks=c(0.5),
                  data=data.frame(Xt,
                                  Classifier="Laplacian Least Squares",
                                  pred=as.numeric(decisionvalues(g_lap,Xt)))) +
geom_classifier("Supervised Least Squares"=g_ls,show_guide=FALSE) + theme(legend.position="bottom") #+ ggtitle("Manifold Assumption")


ggsave(p3,filename = "Figure3.pdf")

# Figure 4
g_ls <- LeastSquaresClassifier(problem1$X,problem1$y)
g_lda <- LinearDiscriminantClassifier(problem1$X,problem1$y)
g_icls <- ICLeastSquaresClassifier(problem1$X,problem1$y,problem1$X_u)
g_mclda <- MCLinearDiscriminantClassifier(problem1$X,problem1$y,problem1$X_u)

p4 <- p1 + geom_classifier("Supervised LDA"=g_lda,"Moment Constrained LDA"=g_mclda,"Supervised LS"=g_ls,"Implicitly Constrained LS"=g_icls) + guides(colour = guide_legend(override.aes = list(size=6,linetype=0))) 
#+ ggtitle("Robust Semi-Supervised Learning")

ggsave(p4,filename = "Figure4.pdf")

lapply(1:4, function(i) { fn <- paste0("Figure",i,".pdf"); embed_fonts(fn, outfile=fn); plot_crop(fn) })



