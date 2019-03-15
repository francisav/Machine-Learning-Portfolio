#HMM
#Andres Vargas

library(mnormt)

########################### FUNCTIONS ###########################

#initialize as defined in paper
#elements of Factors will be filled the scaling factors for each time t
#Alphas is a matrix whose columns are the times and rows are the elements of alpha
#i.e. Alphas is a NxT matrix, where T is the number of time steps in the training sequence
#Factors will come in with one element from initialization step
#similarly, Alphas will come in with only first column filled
getalphas <- function(T_, N, A, B, Alphas, Factors, t=1){
  #tcurr is the current time
  if(t==T_) return( list(Alphas=Alphas,Factors=Factors) )
  alphadot <- cbind(rep(0,N))
  for (i in 1:N){
    # browser()
    alphadot <- alphadot + Alphas[i,t]*t(A[i, ,drop=F])*t(B[t+1, ,drop=F])
  }
  c_ <- 1/sum(alphadot)
  alphanew <- c_ * alphadot
  Factors[t+1] <- c_
  Alphas[ ,t+1] <- alphanew
  return(getalphas(T_, N, A, B, Alphas, Factors, t+1))
}

#in contrast to getalpha, this function will take a filled vector Factors
#because the scaling factors will already have been computed from alpha step
#Betas should come in as an EMPTY NxT matrix except for the last column that contains the initialization
#columns will be filled in reverse
#to begin with, function should be called with t=T
getbetas <- function(N, A, B, Betas, Factors, t){
  if(t==1) return(Betas)
  betadot <- cbind(rep(0,N))
  for (j in 1:N){
    betadot <- betadot + A[ ,j,drop=F]*B[t,j]*Betas[j,t]
  }
  betanew <- Factors[t-1]*betadot
  Betas[ ,t-1] <- betanew
  return(getbetas(N, A, B, Betas, Factors, t-1))
}


#calculate alpha_t and beta_t for every t in {1,...,T}
getgreeks <- function(pi_,A,B){
  N <- nrow(A)
  T_ <- nrow(B)
  Alphas <- matrix(nrow=N,ncol=T_)
  Betas <- Alphas
  alphadot1 <- t(pi_*B[1, ,drop=F]) #initial probabilities for each state
  c1 <- 1/sum(alphadot1)
  alphabar1 <- c1*alphadot1
  Alphas[ ,1] <- alphabar1
  Factors <- c1
  alphastuff <- getalphas(T_,N,A,B,Alphas,Factors)
  Alphas <- alphastuff$Alphas
  Factors <- alphastuff$Factors
  betabarT <- cbind(rep(1,N))*Factors[T_]
  Betas[ ,T_] <- betabarT
  Betas <- getbetas(N, A, B, Betas, Factors, T_)
  return( list(Alphas=Alphas, Betas=Betas, Factors=Factors) )
}


updatepi <- function(alpha1, alphaT, beta1, c1){
  # browser()
  return( 1/c1 * t(alpha1*beta1/sum(alphaT)) ) #pi is a row vector
}

updateA <- function(Alphas, Betas, A, B, Factors){
  T_ <- nrow(B)
  N <- nrow(A)
  numerator <- matrix(0,nrow=N,ncol=N)
  denominator <- rep(0,N)
  for (t in 1:(T_-1)){
    numerator <- numerator + 
      sweep(sweep(sweep(A,MARGIN=1,Alphas[ ,t],`*`),MARGIN=2,B[t+1, ],`*`),MARGIN=2,Betas[ , t+1],`*`)
    denominator <- denominator + Alphas[ ,t]*Betas[ ,t]/Factors[t]
  }
  return( sweep(numerator,MARGIN=1,denominator,`/`) )
}

updateB <- function(Alphas, Betas, A, B, Factors, Y){
  #assume that Y is input as a matrix whose rows sensor measurements 
  #and whose columns are times
  #i.e. Y is a DxT matrix
  
  #first loop through once to update mu.
  T_ <- ncol(Y)
  D <- nrow(Y)
  N <- nrow(A)
  numerator <- matrix(0,nrow=D,ncol=N)
  denominator <- rep(0,N)
  for(t in 1:T_){
    ycopies <- matrix(data=Y[ ,t], nrow=D, ncol=N, byrow=F) #each column of ycopies is a copy of Y[ ,t]
    numerator <- numerator + sweep(ycopies,MARGIN=2,Alphas[ ,t]*Betas[ ,t],`*`)/Factors[t]
    denominator <- denominator + Alphas[ ,t]*Betas[ ,t]/Factors[t]
  }
  mu <- sweep(numerator, MARGIN=2, denominator, `/`)
  
  #loop through again to update sigma.  cannot do in one loop because sigma update requires mu
  numerator <- matrix(0,nrow=D,ncol=N)
  #denominator is same as for mu, so recalculating would be redundant
  for(t in 1:T_){
    copies <- (matrix(data=Y[ ,t], nrow=D, ncol=N, byrow=F) - mu)^2 
    #copies of centered and then squared Y[ ,t] for variance calculation
    numerator <- numerator + sweep(copies,MARGIN=2,Alphas[ ,t]*Betas[ ,t],`*`)/Factors[t]
  }
  sigma <- sweep(numerator, MARGIN=2, denominator, `/`)
  # if(sum(sigma==0)!=0) browser()
  return( populateB(mu,sigma,Y) )
}

reestimate <- function(pi_, A, B, Y){
  #assume that Y is input as a matrix whose rows sensor measurements and columns are times
  #i.e. Y is a TxD matrix
  T_ <- ncol(Y)
  greeks <- getgreeks(pi_,A,B)
  # browser()
  pi_ <- updatepi(greeks$Alphas[ ,1], greeks$Alphas[ ,T_], greeks$Betas[ ,1], greeks$Factors[1])
  A <- updateA(greeks$Alphas, greeks$Betas, A, B, greeks$Factors) 
  B <- updateB(greeks$Alphas, greeks$Betas, A, B, greeks$Factors, Y)
  return( list(pi=pi_, A=A, B=B))
}

loglik <- function(pi_,A,B){
  N <- nrow(A)
  T_ <- nrow(B)
  Alphas <- try(matrix(nrow=N,ncol=T_))
  if(class(Alphas)=='try-error') browser()
  alphadot1 <- t(pi_*B[1, ,drop=F]) #initial probabilities for each state
  c1 <- 1/sum(alphadot1)
  alphabar1 <- c1*alphadot1
  Alphas[ ,1] <- alphabar1
  Factors <- c1
  stuff <- getalphas(T_,N,A,B,Alphas,Factors)
  LL <- log(sum(stuff$Alphas[ ,T_])) - sum(log(stuff$Factors)) #avoid underflow with log likelihood
  return(LL)
}


populateB <- function(mu,sigma,Y){
  #columns of B will be states
  #rows will be the probabilities of each observation
  N <- ncol(mu)
  T_ <- ncol(Y)
  B <- matrix(nrow=T_, ncol=N)
  for (t in 1:T_){
    for (j in 1:N){
      test <- try(dmnorm( x=Y[ ,t], mean=mu[ ,j], varcov=diag(sigma[ ,j]) ))
      # if(class(test)=='try-error') browser()
      B[t,j] <- dmnorm( x=Y[ ,t], mean=mu[ ,j], varcov=diag(sigma[ ,j]) )
      # if(B[t,j]>1 || B[t,j]<0) browser()
    }
  }
  return(B)
}

getchangeprobs <- function(lambda,greeks){
  T_ <- nrow(lambda$B)
  N <- ncol(lambda$B)
  changeprobs <- data.frame(time=1:(T_-1),P=rep(0,T_-1))
  denominator <- sum(greeks$Alpha[ ,T_])
  for (t in 1:(T_-1)){
    numerator <- 0
    for (i in 1:N){
      for(j in 1:N){
        if(i != j){
          numerator <- numerator +
            greeks$Alphas[i,t]*lambda$A[i,j]*lambda$B[t+1,j]*greeks$Betas[j,t+1]
        }
      }
    }
    changeprobs[t,2] <- numerator/denominator
  }
  return(changeprobs)
}
######################### MAIN SCRIPT #########################
# The data set I originally used is proprietary, but the code is general enough to accomadate any data set of your choosing.

#initialize necessary objects
N <- 3 # number of hidden states.
eps <- 0.01 # convergence tolerance
data_ <- # fill this in with a data set whose columns are variables and rows are observations ordered in time
index <- # fill this in with a vector that contains either (1) the ordered timestamps or (2) the ordered indices of the observations
Y <- t(as.matrix(data_))
T_ <- nrow(data_)
D <- nrow(Y)
set.seed(10)

pi_ <- rbind(runif(N))
A <- matrix(runif(N*N),ncol=N,nrow=N)
mu <- t(kmeans(data_,centers=N)$centers) #this is of class matrix
sigma <- matrix(data=1, nrow=D,ncol=N) #initialize all variances to 1
B <- populateB(mu,sigma,Y)
lambda <- list(pi=pi_,A=A,B=B)

LL <- -9999999999999
LLnew <- loglik(pi_,A,B) #solely for the purpose of entering loop
t <- 0
#update until convergence
while(LLnew-LL>eps){
  print( c(itr=t,lik=LLnew) )
  LL <- LLnew
  lambda <- reestimate(lambda$pi,lambda$A,lambda$B,Y)
  t <- t+1
  # if(t==7) browser()
  LLnew <- loglik(lambda$pi, lambda$A, lambda$B)
  if(LLnew<LL) stop('updated likelihood is less than previous likelihood')
}

greeks <- getgreeks(lambda$pi,lambda$A,lambda$B)
probmat <- getchangeprobs(lambda,greeks)
probmat <- cbind(probmat,index[1:(T_-1)])
orderbyprob <- probmat[order(probmat$P, decreasing=T), ]
View(orderbyprob) # observations in order of probability of being a change point.

