ipolate <- function(mat) {
  mat1 <- array(dim=c(dim(mat)[1],length(yrs)))
  ys <- as.numeric(unlist(strsplit(names(mat),"X")))
  est <- seq(2010,2100,5)  #the 5yr estimates in the SSP dataset
  for (i in 1:length(yrs)) {
    y = yrs[i]
    if ("X"%&%y %in% names(pop) == T) {  #if the year falls on the 5-yr interval, use their point estimate. otherwise interpolate between nearest endpoints
      mat1[,i]  <- as.numeric(mat[,which(names(mat)=="X"%&%y)])
    } else {
      z <- y-est
      yl = est[which(z==min(z[z>0]))]  #the 5-year endpoint lower than the year
      y5 = yl+5  #the next endpoint
      el <- as.numeric(mat[,which(names(mat)=="X"%&%yl)])  #values at lower endpoint
      eu <- as.numeric(mat[,which(names(mat)=="X"%&%y5)]) #values at upper endpoint
      if (y > max(ys,na.rm=T)) {  mat1[,i] <- el   #this is to account for growth projections ending in 2095  
      }  else { mat1[,i] <- el + (eu-el)*(y-yl)/5 }
    }
  } 
  mat1 <- data.frame(mat[,1:3],mat1)
  names(mat1)[4:dim(mat1)[2]] <- yrs
  return(mat1)
}



"%&%"<-function(x,y)paste(x,y,sep="")  #define a function for easy string pasting