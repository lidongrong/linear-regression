import numpy
import math
from scipy.stats import t
from scipy.stats import f


#This package provides calculations for linear regression

#testing data
x=[[1,1],[2,0.3],[4,5]]
y=[1,3,4]

#return the mean of an array
def mean(arr):
    return (sum(arr)/len(arr))

#in this program, lm is designed to be a class
class lm:
    def __init__(self,y,x):
        #the X and Y matrix
        self.matrix=numpy.array(x)
        self.y=numpy.array(y)
        self.matrix=numpy.mat(self.matrix)
        self.y=numpy.mat(self.y).T
        self.x=numpy.matrix(self.matrix)

        #Using normal equation to get estimates of coef
        self.coefficients=((((self.x).T)*self.x).I)*(self.x.T)*(self.y)

        #hat matrix
        self.hat=(self.x*(self.x.T*self.x).I)*(self.x.T)
        #fitted values
        self.fitted=self.hat*self.y
        #residuals
        self.residuals=self.y-self.fitted
        #RSS
        self.RSS=self.residuals.T*self.residuals
        
        self.TSS=0
        for i in range(0,len(y)):
            self.TSS=self.TSS+(mean(self.y)-self.y[i])**2

        #R square
        self.Rsquare=1-self.RSS/self.TSS

        #number of observations
        self.obs=len(self.y)

        #the residual standard error
        self.sigma=math.sqrt(self.RSS/(self.obs-self.x.shape[1]))

        #the covariance matrix of residuals
        self.rescov=((self.sigma)**2)*(numpy.ones(self.hat.shape[0])-self.hat)
        #the covariance matrix of coefficients
        self.coefcov=((self.sigma)**2)*(self.x.T*self.x).I
        #the covariance matrix of fitted values
        self.fitcov=(self.sigma**2)*self.hat

        #standard error of coefficients
        self.sd=numpy.zeros(len(self.coefficients))
        for i in range(0,len(self.coefficients)):
            self.sd[i]=self.sigma*math.sqrt(((self.x.T*self.x).I)[i,i])
            

        self.t=numpy.zeros(len(self.coefficients))

        #running t test for each coefficient
        for k in range(0,len(self.coefficients)):
            self.t[k]=self.coefficients[k]/self.sd[k]

        #running f test for the regression
        self.f=(self.TSS-self.RSS)/(self.RSS/(self.obs-self.x.shape[1]))

        #get p value for t test
        self.pt=numpy.zeros(len(self.t))
        for i in range(0,len(self.pt)):
            self.pt[i]=2*(1-(t.cdf(abs(self.t[i]),self.obs-self.x.shape[1])))

        #get p value for f test
        self.pf=1-f.cdf(self.f,1,self.obs-self.x.shape[1])

        #get degrees of freedom
        self.df=self.obs-self.x.shape[1]



    #this function modules the summary() function in R
    def summary(self):
        print("Coefficients:")
        print("Estimates")
        
        coef=numpy.zeros(len(self.coefficients))
        for i in range(0,self.x.shape[1]):
            coef[i]=self.coefficients[i]
        print(coef)

        print()
        
        print("Std. Error")
        print(self.sd)
        print()
        print("t value")
        print(self.t)
        print()
        print("Pr(>|t|)")
        print(self.pt)
        print()
        print("Residual standard error ",self.sigma," on ",self.df," degrees of freedom")
        print("R-squared: ",self.Rsquare)
        print("F-statistic: ",self.f," on ",1," and ",self.df," degrees of freedom")
        print("p-value: ",self.pf)

        
