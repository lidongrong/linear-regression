import numpy
import math
from scipy.stats import t
from scipy.stats import f

#This package provides calculations for linear regression

x=[[1,1],[2,0.3],[4,5]]
y=[1,3,4]

def mean(arr):
    return (sum(arr)/len(arr))

class lm:
    def __init__(self,y,x):
        self.matrix=numpy.array(x)
        self.y=numpy.array(y)
        self.matrix=numpy.mat(self.matrix)
        self.y=numpy.mat(self.y).T
        self.x=numpy.matrix(self.matrix)

        self.coefficients=((((self.x).T)*self.x).I)*(self.x.T)*(self.y)

        self.hat=(self.x*(self.x.T*self.x).I)*(self.x.T)
        self.fitted=self.hat*self.y
        self.residuals=self.y-self.fitted
        self.RSS=self.residuals.T*self.residuals
        
        self.TSS=0
        for i in range(0,len(y)):
            self.TSS=self.TSS+(mean(self.y)-self.y[i])**2

        self.Rsquare=1-self.RSS/self.TSS

        self.obs=len(self.y)
        self.sigma=math.sqrt(self.RSS/(self.obs-self.x.shape[1]))

        self.rescov=((self.sigma)**2)*(numpy.ones(self.hat.shape[0])-self.hat)
        self.coefcov=((self.sigma)**2)*(self.x.T*self.x).I
        self.fitcov=(self.sigma**2)*self.hat
        
        self.sd=numpy.zeros(len(self.coefficients))
        for i in range(0,len(self.coefficients)):
            self.sd[i]=self.sigma*math.sqrt(((self.x.T*self.x).I)[i,i])
            

        self.t=numpy.zeros(len(self.coefficients))
        
        for k in range(0,len(self.coefficients)):
            self.t[k]=self.coefficients[k]/self.sd[k]

        self.f=(self.TSS-self.RSS)/(self.RSS/(self.obs-self.x.shape[1]))

        self.pt=numpy.zeros(len(self.t))
        for i in range(0,len(self.pt)):
            self.pt[i]=2*(1-(t.cdf(abs(self.t[i]),self.obs-self.x.shape[1])))

        self.pf=1-f.cdf(self.f,1,self.obs-self.x.shape[1])
        self.df=self.obs-self.x.shape[1]

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

        
