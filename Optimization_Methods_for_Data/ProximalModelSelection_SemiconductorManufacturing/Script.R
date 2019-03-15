## install packages ##
library(magic)
################### FUNCTIONS ####################################

# Performs full optimization procedure as specified in section 5 of paper
fit_proc <- function(dat, lambda, normal_model=NULL){ #the normal model is a list that contains the normal model.
  # if normal model is supplied, this implies that it is NOT an initial fit context
  # error variance, normal mean, normal variance
  # column order of dat: 'TimeFromRunStart',sensor_placeholder,'WaferID','LotID','TimeStamp'
  colnames(dat)[2] <- 'response' #placeholder column name cause this column will change depending on the iteration of the outer loop  (see main script)
  unscaled.response <- dat$response
  unlots <- unique(as.character(dat$LotID))
  dat[ ,WaferID:=factor(WaferID)] # gets rid of degenerate factor levels by reference
  response.mean <- mean(unscaled.response)
  response.sd <- sd(unscaled.response)
  dat[ ,response:=(response-response.mean)/response.sd]
  unwaf <- as.character(unique(dat$WaferID))
  if(length(unwaf)==1) onewaf <- T else onewaf <- F
  if(nrow(dat)<10 || onewaf){
    return(unwaf)
  }
  # if(onewaf && is.null(normal_model)) {
  #   return(unwaf)
  # }
  
  
  # print(paste0('fitting lots: ',paste(unlots,collapse=', ')))
  it <- 0
  exitloop <- F #switches to true when the vector of estimates (one for each wafer )for any of the parameters has all entries equal ( see parsdiff() )
  #initial guesses for fisher information, prior mean, and prior fisher information
  fisher <- 1 #this seems like a reasonable initial guess I guess?
  switch(.shape,
         'damped linearly driven' = {muS <- c(1,1,1,1,1,1); fisherS <- c(0,0,0,0,0,0)},
         'damped constantly driven' = {muS <- c(1,1,1,1,1); fisherS <- c(0,0,0,0,0)},
         'linear exponential' = {muS <- c(1,1,1,1); fisherS <- c(0,0,0,0)},
         'exponential' = {muS = c(1,1,1); fisherS = c(0,0,0)},
         'linear' = {muS <- c(1,1); fisherS <- c(0,0)},
         'constant' = {muS <- 1; fisherS <- 0}
  )
  obj_old <- 0.010101293817391 #this just needs to not be equal to obj_curr so that the while loop gets entered.
  #Thus, while loop will be entered with probability 1
  # dat$LampPower04 <- dat$LampPower04 - mean(dat$LampPower04, na.rm=T)
  InitialEverything <- getInitial(dat,unscaled.response,fisher,muS,fisherS,unwaf,.shape,onewaf,lambda)
  shiftdata <- InitialEverything$shiftdata
  xshifts <- InitialEverything$shifts
  sub_objs <- InitialEverything$sub_objs
  obj_curr <- Inf
  if(!onewaf){
    meanxshift <- mean(xshifts)
    varxshift <- var(xshifts)
  }
  fmat <- do.call('rbind', InitialEverything$pars)
  pars <- list()
  RST <- .POSIXct(character(0)) #RST stands for run start time.  This line creates an empty POSIXct vector
  for(w in unwaf){
    w.logical <- shiftdata$WaferID==w
    RST[w] <- min(shiftdata$TimeStamp[w.logical]) #minimum timestamp is the run start time
  }
  while(it<5 && round(obj_curr,3)!=round(obj_old,3) && exitloop==F){
    obj_old <- obj_curr
    
    pars.grad.hess <- switch(as.character(it),
                             '0' = nonlinSolver(pars, xshifts, shiftdata, fisher, muS, meanxshift, fisherS, 1/varxshift, unwaf, .shape, lambda, InitialEverything$pars)
                             , nonlinSolver(pars, xshifts, shiftdata, fisher, muS, meanxshift, fisherS, 1/varxshift, unwaf, .shape, lambda)
    )
    pars <- pars.grad.hess[['pars']]
    ssr <- get_ssr_from_sslist(pars, shiftdata, unwaf, .shape)
    sig.sq <- ssr/(nrow(shiftdata)-1)
    fmat <- do.call('rbind',pars)
    # if(parsdiff(pars) && !onewaf){
    #   muS <- apply(fmat,2,mean)
    #   sigma.sq.S <-  switch(.shape,
    #                         'constant' = sum(apply(fmat,1,function(x,muS) (x-muS)^2,muS=muS))/(length(unwaf)-1),
    #                         apply(apply(fmat,1,function(x,muS) (x-muS)^2,muS=muS),1,sum)/(length(unwaf)-1)
    #   )
    #   fisherS <- 1/sigma.sq.S #inverse of the variance
    #   fisher <- 1/sig.sq # fisher info is only equal to inverse variance if parameter estimator is efficient.  
    #   if(fisher==Inf || Inf %in% fisherS){
    #     # if information is infinitie, we have zero error, either in the data fit or in one of the parameters.
    #     # thus, we break and compute the anomaly score if the score is being computed with respect to the normal_model (which will likely never have infinite information) 
    #     if(is.null(normal_model)) warning('Infinite Objective Function During Initial Fit')
    #     # but, if we are in an initial fit setting, then there is no normal model, and thus infinite information will cause algorithm to crash when outputting anomaly scores.
    #     # thus, we call browser() if this occurs.
    #     exitloop <- T
    #   }
    #   obj_curr <- sum(calculate_sub_objs(pars,sub_objs,fisher,muS,fisherS,xshifts,meanxshift,varxshift,unwaf))
    #   it <- it+1
    # }else{
      muS <- apply(fmat,2,mean)
      sigma.sq.S <-  switch(.shape,
                            'constant' = sum(apply(fmat,1,function(x,muS) (x-muS)^2,muS=muS))/(length(unwaf)-1),
                            apply(apply(fmat,1,function(x,muS) (x-muS)^2,muS=muS),1,sum)/(length(unwaf)-1)
      )
      sigma.sq.S[sigma.sq.S==0] <- 1e-11
      fisherS <- 1/sigma.sq.S #inverse of the variance
      fisher <- 1/sig.sq 
      # exitloop <- T
      obj_curr <- sum(calculate_sub_objs(pars,sub_objs,fisher,muS,fisherS,xshifts,meanxshift,varxshift,unwaf))
      it <- it+1
    # }
  }
  # anomaly_scores_orig <- cbind(calculate_sub_objs(pars,sub_objs,fisher,muS,fisherS,xshifts,meanxshift,varxshift,unwaf))
  undone.named <- undoCenterScale.name(muS, fisher, fisherS, as.data.frame(fmat), response.mean, response.sd, .shape)
  pars <- undone.named$pars
  if(!is.null(normal_model)){ 
    init.flag <- F
    anomaly_scores <- cbind(calculate_sub_objs(pars,InitialEverything$unscaled_sub_objs,normal_model$fisher,normal_model$muS,
                                               normal_model$fisherS,xshifts,normal_model$meanxshift,
                                               1/normal_model$fisherxshift,unwaf))
  }else{
    init.flag <- T
    fisher <- undone.named$fisher
    muS <- undone.named$muS
    fisherS <- undone.named$fisherS
    unscaled_sub_objs <- InitialEverything$unscaled_sub_objs
    sub_objs_perm <- unscaled_sub_objs[sample(1:length(sub_objs))] # in the initial fit, permute to prevent overfitting
    anomaly_scores_perm <- cbind(calculate_sub_objs(pars,sub_objs_perm,fisher,muS,fisherS,xshifts,meanxshift,varxshift,unwaf))
    anomaly_scores <- cbind(calculate_sub_objs(pars,unscaled_sub_objs,fisher,muS,fisherS,xshifts,meanxshift,varxshift,unwaf))
  }
  medianS <- apply(undone.named$shapesigs,2,median) 
  shapesigs <- cbind(RunStartTime=cbind.data.frame(RST),undone.named$shapesigs,x=cbind(xshifts), anomaly_scores )
  setnames(shapesigs, c('RST','xshifts','anomaly_scores'),c('RunStartTime','x','AnomalyScore'))
  rownames(shapesigs) <- names(pars)
  lotids <- character(0)
  for(s in rownames(shapesigs)){
    lotid <- as.character(unique(dat[WaferID==s,LotID]))
    if(length(lotid)>1) stop(paste0('waferid ',s,' appears in more than 1 lot'))
    lotids <- append(lotids,lotid)
  }
  
  shapesigs$init.flag <- init.flag
  shapesigs$LotID <- lotids
  # print('done fitting')
  fisherxshift <- 1/varxshift
  # if statement below to be filled in later.  it should compute the gradient and hessian at the optimal solution
  # if(is.null(normal_model)){ # If the condiiton is true, then it is the initial fit, which means we'd like to return the gradient and hessian at the
  #   pars.append <- mapply(function(a,b) append(a,b), pars, xshifts, SIMPLIFY=F)
  #   grad_s <- pars.grad.hess[['grad']]
  #   hess_s.s <- pars.grad.hess[['hess']]
  #   opt.info <- get.optimality.info(dat, unwaf, pars.append, append(muS,meanxshift), append(fisherS,fisherxshift), fisher, grad_s, hess_s.s, .shape)
  # }
  # return(list(shapesigs=shapesigs,muS=muS,meanxshift=meanxshift,fisherS=fisherS,fisherxshift=fisherxshift,fisher=fisher, 
  #             shiftdat=dat, initialpars=InitialEverything$pars, opt.loglik = -sum(shapesigs$AnomalyScore), opt.grad=opt.info$grad, opt.hess=opt.info$hess))
  if(is.null(normal_model)){
    return(list(shapesigs=shapesigs, muS=muS, meanxshift=meanxshift, fisherS=fisherS, fisherxshift=fisherxshift, fisher=fisher, 
                opt.loglik = -sum(anomaly_scores), medianS=medianS))
  }else{
    return(shapesigs)
  }
}

# Data-driven initial guesses.  Called from within getInitial (see next function)
get_init_parms <-
  function(dat,which.method){
    if(nrow(dat)>1 && length(unique(dat$TimeFromRunStart))>1) trendline <- lm(dat[,2]~dat[,1]) else trendline <- NULL
    if(!is.null(trendline)){
      yguess <- coef(trendline)[1] #y intercept of trendline
      detrended <- dat[ ,2] - trendline$fitted.values
      firsthalf <- with(dat,which(TimeFromRunStart<=median(TimeFromRunStart)))
      secondhalf <- with(dat,which(TimeFromRunStart>median(TimeFromRunStart)))
      firstind <- with(dat, which(TimeFromRunStart==min(TimeFromRunStart)))[1]
      gammaguess <- 0.2
      slopeguess <- coef(trendline)[2]
      # pg <- periodogram(detrended, plot=F)
      # powerfullest.index <- order(-pg$spec)[1] #index of most powerful frequency
      # dom.freq <- pg$freq[powerfullest.index]
      # wguess <- dom.freq*2*pi # correct conversion requires multiplying by an additional factor of 10
      # but the sampling rate of the data is low enough to cause aliasing so that an order of magnitude higher frequency than the truth is often output by periodogram
      # dropping the factor of 10 prevents this from happening, for the most part...
      # might be safer to just hard code it to an initial guess of 0.3, as was being done before (see commented code above).
      # at the moment, this seems to be working better.  so keep it for now.
      wguess <- 0.3
      phiguess <- 0
      Rguess <- detrended[firstind]
      if(Rguess == 0) {
        Rguess <- 0.0001 # R can't be zero, because hessian requires dividing by it.
        # wguess <- 0.3 # If the amplitude is zero, then the frequency can be anything.  thus, hard code to 0.3 to stabalize estimates
      }
      
    }else{
      yguess <- median(dat[,2])
      Rguess <- 1
      gammaguess <- 0
      wguess <- 0
      slopeguess <- 0
      phiguess <- 0
    }
    
    returnvec <- c(gammaguess, Rguess, wguess, yguess, phiguess, slopeguess)
    names(returnvec) <- c('gamma','R','omega','y','phi','slope')
    
    return(returnvec)
  }

# Function to get initial parameters
getInitial <- function(initialdat, unscaled.response, fisher.init,muS.init,fisherS.init,waford,onewaf, lambda){ 
  #waford gives the order for the wafers.
  #Return the new data frame that has been shifted, along with the shifts,
  #and the different sub-objective functions (these sub-objective functions are later summed togetHcondliker to obtain the actual objective function),
  # lastly return the value of the initialguesses at each of the values of the sub_objective functions
  shftout <- with(initialdat,by(initialdat,WaferID, shiftdat ))
  reord <- mapord(targetvec=waford,vect=names(shftout))
  shftout <- shftout[reord]
  initialdat.shifted <- do.call('rbind',lapply(shftout,'[[',1)) #select 1st element of each element of shftout, bind result into one large data.table
  unscaledat.shifted <- copy(initialdat.shifted)
  unscaledat.shifted[ ,response:=unscaled.response]
  initialdat.shifted.frame <- as.data.frame(initialdat.shifted)
  unscaledat.shifted.frame <- as.data.frame(unscaledat.shifted)
  xshifts <- sapply(shftout,'[[',2)
  initguesses <- with(initialdat.shifted.frame,by(initialdat.shifted.frame,WaferID,get_init_parms))
  initguesses <- initguesses[mapord(waford,names(initguesses))]
  sub_objs <- with(initialdat.shifted.frame,by(initialdat.shifted.frame,WaferID,get_full_obj1, lambda)) 
  sub_objs <- sub_objs[mapord(waford,names(sub_objs))]
  unscaled_sub_objs <- with(unscaledat.shifted.frame,by(unscaledat.shifted.frame,WaferID,get_full_obj1,lambda))
  unscaled_sub_objs <- unscaled_sub_objs[mapord(waford,names(unscaled_sub_objs))]
  return(list(shiftdata=initialdat.shifted.frame, shifts=xshifts, sub_objs=sub_objs, pars=initguesses, unscaled_sub_objs=unscaled_sub_objs))
  
}

# Returns objective function (not value) as a function of s, fisher, muS, and fisherS=inv(diag(sigma_S))
get_full_obj1 <- function(dat,lambda){
  # returns a the function below, which takes the parameter vector as the only argument
  function(s,fisher,muS,fisherS){ 
    C <- exp(-s[['gamma']]*dat[,1])*cos(s[['omega']]*dat[,1]-s[['phi']]) # need to wrap in try so that program won't crash if any elements of s are missing
    E <- dat[,2]-s[['y']]-s[['R']]*C-s[['slope']]*dat[,1]
    return( -0.5*nrow(dat)*log(fisher) + 0.5*fisher*sum(E^2) - 0.5*sum(log(fisherS))  + 0.5*sum((s-muS)^2*fisherS) + 
              sum(lambda*abs(s)) )
  }
}

# Returns objective function (not value) as a function of just the shape signature s
get_obj1 <- function(dat,fisher.current,muS.current,fisherS.current,lambda){
  #data should be a data frame with two columns; the first corresponding to RunLength and the second one corresponding to sensor values
  function(s){
    C <- exp(-s[['gamma']]*dat[,1])*cos(s[['omega']]*dat[,1]-s[['phi']])
    E <- dat[,2]-s[['y']]-s[['R']]*C-s[['slope']]*dat[,1]
    0.5*fisher.current*sum(E^2) + 0.5*sum((s-muS.current)^2*fisherS.current) + sum( lambda*abs(s) )
  }
}

get_grad_condlik_gamma <- function(dat,fisher.current){
  function(s){
    C <- exp(-s[['gamma']]*dat[,1])*cos(s[['omega']]*dat[,1]-s[['phi']])
    E <- dat[,2]-s[['y']]-s[['R']]*C-s[['slope']]*dat[,1]
    inner <- dat[,1]*E*C
    return(s[['R']]*fisher.current*sum(inner))
  }
}

get_grad_condlik_R <- function(dat,fisher.current){
  function(s){
    C <- exp(-s[['gamma']]*dat[,1])*cos(s[['omega']]*dat[,1]-s[['phi']])
    E <- dat[,2]-s[['y']]-s[['R']]*C-s[['slope']]*dat[,1]
    inner <- E*C
    return(-sum(inner)*fisher.current)
  }
}

get_grad_condlik_omega <- function(dat,fisher.current){
  function(s){
    C <- exp(-s[['gamma']]*dat[,1])*cos(s[['omega']]*dat[,1]-s[['phi']])
    D <- exp(-s[['gamma']]*dat[,1])*sin(s[['omega']]*dat[,1]-s[['phi']])
    E <- dat[,2]-s[['y']]-s[['R']]*C-s[['slope']]*dat[,1]
    inner <- dat[,1]*E*D
    return(s[['R']]*fisher.current*sum(inner))
  }
}

get_grad_condlik_y <- function(dat,fisher.current,.shape){
  function(s){
    C <- exp(-s[['gamma']]*dat[,1])*cos(s[['omega']]*dat[,1]-s[['phi']])
    E <- dat[,2]-s[['y']]-s[['R']]*C-s[['slope']]*dat[,1]
    inner <- E
    return(-sum(inner)*fisher.current)
  }
}

get_grad_condlik_phi <- function(dat,fisher.current,.shape){
  function(s){
    C <- exp(-s[['gamma']]*dat[,1])*cos(s[['omega']]*dat[,1]-s[['phi']])
    D <- exp(-s[['gamma']]*dat[,1])*sin(s[['omega']]*dat[,1]-s[['phi']])
    E <- dat[,2]-s[['y']]-s[['R']]*C-s[['slope']]*dat[,1]
    inner <- E*D
    return(-s[['R']]*sum(inner)*fisher.current)
  }
}

get_grad_condlik_slope <- function(dat,fisher.current,.shape){
  function(s){
    C <- exp(-s[['gamma']]*dat[,1])*cos(s[['omega']]*dat[,1]-s[['phi']])
    E <- dat[,2]-s[['y']]-s[['R']]*C-s[['slope']]*dat[,1]
    inner <- E*dat[,1]
    return(-sum(inner)*fisher.current)
  }
}


get_grad_condlik <- function(dat,fisher.current,.shape){
  function(s) {
    c(gamma=get_grad_condlik_gamma(dat,fisher.current,.shape)(s),
      R=get_grad_condlik_R(dat,fisher.current,.shape)(s),
      omega=get_grad_condlik_omega(dat,fisher.current,.shape)(s),
      y=get_grad_condlik_y(dat,fisher.current,.shape)(s),
      phi=get_grad_condlik_phi(dat,fisher.current,.shape)(s),
      slope=get_grad_condlik_slope(dat,fisher.current,.shape)(s))
  }
}

get_grad <- function(grad_condlik,muS.current,fisherS.current){
  function(s){
    grad_condlik(s) + (s-muS.current)*fisherS.current
  }
}

get_lower_bound <- function(){ #lower bound for parameters 
  c(gamma=-Inf,R=-Inf,omega=0,y=-Inf,phi=-0.5*pi, slope=-Inf)
}

# Performs optimizatin procedure for block 1 (see section 5 of paper) treating the hyper parameters as constants.
nonlinSolver <- function(pars, xshifts, dat, fisher, muS, muxshift, fisherS, fisherxshift, unwaf, lambda, inits=NULL){
  D <- length(muS)
  M <- length(unwaf)
  cnt <- 0
  for(w in unwaf){
    fitdat <- with(dat,dat[WaferID==w,c(1,2)]) #use columns TimeFromRunStart, sensor, and TimeStamp
    if(!is.null(inits)){
      init <- inits[[w]]
    }else{
      init <- pars[[w]]
    }
    obj1 <- get_obj1(fitdat,fisher.current=fisher,muS.current=muS,fisherS.current=fisherS,lambda)
    gcond <- get_grad_condlik(fitdat,fisher)
    g <- get_grad(gcond,muS,fisherS)
    lb <- get_lower_bound()
    ub <- -lb
    ub[['gamma']] <- Inf
    ub[['omega']] <- Inf
    ub[['phi']] <- pi/2
    fit <- proxgrad(init, obj1, g, lambda)
    if(class(fit)=='try-error') {fit <- init; print(paste0('failed on wafer',w))}
    print(paste0('fittted wafer', w))
    pars[[w]] <- fit
    cnt <- cnt+1
  }
  return(list(pars=pars, grad=NULL, hess=NULL))
}

# proximal gradient descent algorithm
proxgrad <- function(s, obj, g, lambda, tol=1e-2){
  objprev <- obj(s)  + 2*tol  # this guarantees entrance into the while loop
  len_s <- length(s)
  it <- 1
  while(abs(objprev-obj(s)) > tol  && it <= 500){
    sprev <- s
    objprev <- obj(sprev)
    alpha <- 10
    stent <- mapply(function(a,b) prox(a,b), s-alpha*g(s), lambda)
    stent <- projgamma(projomega(projphi(stent)))
    # backtracking with armijo sufficient decrease condition, beta hard coded to 1.1
    while( obj(stent) > obj(s) + g(s) %*% (stent-s) + 0.5*sum( (s-stent)^2 )/(alpha*10.1) ){ 
      alpha <- alpha/10
      stent <- mapply(function(a,b) prox(a,b), s-alpha*g(s), lambda)
      stent <- projgamma(projomega(projphi(stent)))
      if(class(stent)=='try-error') browser()
    }
    s <- stent
    it <- it+1
    # print(c(objs=obj(s)))
  }
  print(it)
  return(s)
}

# proximal operator for L1 regularization
prox <- function(x,lambda){
  if(lambda < x) return(x-lambda)
  if(-lambda > x) return(x+lambda)
  return(0)
}

# project omega onto feasible region
projomega <- function(s){
  if(s[['omega']]<0) s[['omega']] <- 0
  return(s)
}

# project gamma onto feasible region
projgamma <- function(s){
  if(s[['gamma']]<0) s[['gamma']] <- 0
  return(s)
}

# project phi onto feasible region
projphi <- function(s){
  if(s[['phi']] > pi/2) {
    s[['phi']] <- pi/2
  }else if(s[['phi']] < -pi/2) {
    s[['phi']] <- -pi/2
  }
  return(s)
}



################################################ MAIN ################################################
# The data used for this project is proprietary.  To apply it to your own data set, make sure it is in the
# form specified in section 2 of the paper. Then
initialdat <- YOURDATA
# adjust regularization paramters for gamma, R, omega, y, phi, and slope as needed
fit <- fit_proc(initialdat, lambda=c(gamma=0.001,R=0.001,omega=0.01,y=0,phi=0.01,slope=0.001))


