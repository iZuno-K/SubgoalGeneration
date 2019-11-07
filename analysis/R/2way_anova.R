# Title     : TODO
# Objective : TODO
# Created by: izuno
# Created on: 2019/11/07

install.packages("jsonlite")
install.packages("curl")
library(jsonlite)

# read csv
library(NSM3)
x <- read.table("/tmp/test.csv", header=T, sep=",")
summary(aov(y~Step*Method,data=x))
# visualize
attach(x)
interaction.plot(Step,Method,y)

 x[x$Step=="step1000000" & x$Method=="Default", "y"]

s1d = x[x$Step=="step1000000" & x$Method=="Default", "y"]
s2d = x[x$Step=="step2000000" & x$Method=="Default", "y"]
e1d = x[x$Step=="step1000000" & x$Method=="EExploitation", "y"]
e2d = x[x$Step=="step2000000" & x$Method=="EExploitation", "y"]
l1d = x[x$Step=="step1000000" & x$Method=="large_variance", "y"]
l2d = x[x$Step=="step2000000" & x$Method=="large_variance", "y"]
l = list(s1d=s1d, s2d=s2d, e1d=e1d, e2d=e2d, l1d=l1d, l2d=l2d)
pSDCFlig(l, method='Monte Carlo')

