# Title     : TODO
# Objective : TODO
# Created by: izuno
# Created on: 2019/11/07

install.packages("jsonlite")
install.packages("curl")
library(jsonlite)

# read csv
x <- read.table("/tmp/test.csv", header=T, sep=",")
summary(aov(y~Step*Method,data=x))
# visualize
attach(x)
interaction.plot(Step,Method,y)

 x[x$Step=="step1000000" & x$Method=="Default", "y"]