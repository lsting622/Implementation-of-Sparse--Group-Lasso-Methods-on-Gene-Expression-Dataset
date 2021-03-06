library("newSGL")
library("pls")

timings <- array(0, c(3,4,3,10))

np <- c(10, 200, 100, 400)
lp <- c(150, 10, 100, 50)
n <- c(60, 70, 150, 200)
frac <- c(0.6, 0.6, 0.6, 0.6)
groups <- c(1,2,3)

set.seed(100)

for(ii in 1:4){
  for(jj in 1:3){
    for(iter in 1:10){

p <- rep(lp[ii], np[ii])


index <- rep(1,p[1])
for(i in 2:np[ii]){
  index <- c(index, rep(i,p[i]))
}

X <- matrix(rnorm(n[ii]*p[1]), nrow = n[ii])

for(i in 2:np[ii]){
  X <- cbind(X, X <- matrix(rnorm(n[ii]*p[i]), nrow = n[ii]))
}

X <- stdize(X)

eta <- 0
correct <- NULL
for(i in 1:groups[jj]){
  eta <- eta + X[,1:5 + (i-1)*lp[ii]]%*%(1:5)
}

norm.eta <- sqrt(sum((eta-mean(eta))^2))

eta <- stdize(eta)

y <- eta + 0.5*rnorm(n[ii])

y1 <- rbinom(length(eta),1,exp(y*5)/(1+exp(y*5)))
time <- as.numeric(exp(y))
status <- c(1,rbinom(length(y)-1,1,0.5))


data.lin <- list(x = X, y = y)

time.lin <- system.time(aa <- SGL(data.lin, index, type = "linear", nlam = 20, min.frac = frac[ii]))

timings[1,ii,jj,iter] <- time.lin[1]


write(c(ii,jj,iter),"")

}
}
}

timings.mean <- apply(timings,c(1,2,3),mean)
save.image("short1")

for(i in 1:4){
  output <- "linear &"
  for(j in 1:3){
    output <- paste(output, signif(timings.mean[1,i,j],4), " & ")
  }

  write(output,"")
}
### GROUPWISE ###
