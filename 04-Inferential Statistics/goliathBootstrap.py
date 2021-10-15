import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set_style("whitegrid")  

class goliathBootstrap():
    '''

    '''


    def __init__(self, sample_data, num_samples=10000, sample_size=100, func=np.mean):
        '''    
        It creates the sample distribution using the bootstrap method (resampling WITH replacement)
        sample_data: sample data (must be representative)
        num_samples: number of samples to generate for the bootstrap method
        sample_size: size of each of num_samples samples to generate
        '''
        self.sample_data = sample_data
        self.num_samples = num_samples
        self.sample_size = sample_size

        self.statistic = func

        self.generateSamples()
        self.createSampleDistribution(self.statistic) 
        

    def generateSamples(self):
        '''
        It returns a DataFrame where each column is a sample with replacement.
        '''
        self.dfSamples = pd.DataFrame()
        for k in range(self.num_samples):
            sample = np.random.choice(self.sample_data, size=self.sample_size, replace=True)
            column_name = 'Sample'+str(k)
            self.dfSamples[column_name] = sample

    
    def createSampleDistribution(self, func):
        '''
        It creates the samples distribution using func.
        '''
        self.sample_distribution = np.array(self.dfSamples.apply(func))
        self.statistic = func       

    
    def setSampleDistribution(self, sample_distribution, func):
        '''
        It modifies sample_distribution and updates statistic.
        (It is used in two independent samples HT)
        '''  
        self.sample_distribution = np.array(sample_distribution)
        self.statistic = func      

    
    def graphSampleDistribution(self):
        '''
        It graph the sample distribution
        '''
        title = 'Sample Distribution - ' + str(self.statistic).split(' ')[1]
        sns.displot(self.sample_distribution, height=3.5, aspect=1.5).set(title=title) 



class ConfidenceInterval(goliathBootstrap):


    def confidenceInterval(self, confidence=95):    
        '''
        It returns a confidence% confident interval using func as a 
        distribution sample.
        '''
        alpha = 100 - confidence
        lower_percentile = alpha / 2.0    
        lower = np.percentile(self.sample_distribution, lower_percentile)
        upper_percentile = lower_percentile + confidence
        upper = np.percentile(self.sample_distribution, upper_percentile)
        return(lower, upper)  


    def graphConfidenceInterval(self, confidence=95):
        '''
        It returns a confidence% confident interval and presents the result graphically.
        '''
        lower, upper = self.confidenceInterval(confidence)
        sns.displot(self.sample_distribution, kde=True, height=3.5, aspect=1.5)
        plt.title('Sample Distribution\n %i%% CI: (%.2f, %.2f)'%(confidence, lower, upper))
        plt.axvline(x = lower, ymin=0, ymax=0.5, color='orangered', linewidth=2)
        plt.axvline(x = upper, ymin=0, ymax=0.5, color='orangered', linewidth=2)
        return(lower, upper)         



class OneSampleHT(ConfidenceInterval):

    def __init__(self, sample_data, num_samples=10000, sample_size=100, func=np.mean):
        super().__init__(sample_data, num_samples, sample_size, func)
        self.sample_mean  = np.round(np.mean(self.sample_distribution),2)
        

    def createSampleDistributionHo(self, pop_value):
        '''
        It creates the samples distribution centered at pop_value (true under Ho).
        '''
        self.sample_distribution_Ho = self.sample_distribution - self.sample_distribution.mean() + pop_value
        self.sample_mean = np.round(np.mean(self.sample_distribution),2)

    
    def getpValue(self, obs_value, alpha=0.05, alternative='two-sided'):
        '''
        It calculates the p-value for one-sample HT
        obs_value: obs_value: observed value 
        alpha: significance level
        alternative: one of the three values: 'two-sided', 'smaller', or 'larger'    
        '''
        ecdf = ECDF(self.sample_distribution_Ho) 
        if alternative=='two-sided':
            if obs_value < np.mean(self.sample_distribution_Ho): 
                p_val = 2*ecdf(obs_value)
            else: 
                p_val = 2*(1-ecdf(obs_value)) 
        elif alternative=='smaller':
            p_val = ecdf(obs_value)
        else:
            p_val = 1-ecdf(obs_value)
        return(p_val)


    def graphpValue(self, obs_value, alpha=0.05, alternative='two-sided'):
        '''
        It calculates the p-value for one-sample HT, and also graph the sample distribution, 
        the critical region, and the obs_value
        obs_value: obs_value: observed value
        alpha: significance level
        alternative: one of the three values: 'two-sided', 'smaller', or 'larger'    
        '''
        ecdf = ECDF(self.sample_distribution_Ho) 
        p_val = self.getpValue(obs_value, alpha, alternative) 
        ax = sns.kdeplot(x=self.sample_distribution_Ho, color='lightskyblue', shade=True, alpha=0.4)
        plt.axvline(x=obs_value, ymin=0, ymax= 0.02, color='black', linewidth=6)
        plt.title('Sampling Distribution')
        if alternative=='two-sided':
            cv1 = np.round(np.percentile(self.sample_distribution_Ho, (alpha/2)*100),2)    
            cv2 = np.round(np.percentile(self.sample_distribution_Ho, 100-alpha*100),2)     
            plt.axvline(x = cv1, ymin=0, ymax=0.5, color='orangered', linewidth=2)
            plt.axvline(x = cv2, ymin=0, ymax=0.5, color='orangered', linewidth=2);            
        elif alternative=='smaller':
            cv1 = np.round(np.percentile(self.sample_distribution_Ho, alpha*100),2)  
            plt.axvline(x = cv1, ymin=0, ymax=0.5, color='orangered', linewidth=2)
        else:
            cv2 = np.round(np.percentile(self.sample_distribution_Ho, 100-alpha*100),2)  
            plt.axvline(x = cv2, ymin=0, ymax=0.5, color='orangered', linewidth=2)
        return(p_val)


    def oneSampleHT(self, pop_value, alpha=0.05, alternative='two-sided'):
        '''
        It computes the bootstrap one-sample test.
        obs_value: observed value
        alpha: significance level
        alternative: one of the three values: 'two-sided', 'smaller', or 'larger'    
        '''
        #sigHo = {'two-sided':' =', 'smaller':'>=', 'larger':'<='}
        sigHa = {'two-sided':'!=', 'smaller':'< ', 'larger':'> '}
        print('--- Bootstrapping Method ---')
        print('    Ho: measure =', np.round(pop_value,2)) 
        print('    Ha: measure', sigHa[alternative], np.round(pop_value,2))    
        self.createSampleDistributionHo(pop_value)
        obs_value = self.sample_mean
        print('    Sample mean = %.2f' %(self.sample_mean))
        p_val = self.getpValue(obs_value, alpha, alternative)
        print('    p-value = '+str(np.round(p_val,4)))

     
    def graphOneSampleHT(self, pop_value, alpha=0.05, alternative='two-sided'):
            '''
            It computes the bootstrap one-sample test and gets graphical results.
            obs_value: observed value
            alpha: significance level
            alternative: one of the three values: 'two-sided', 'smaller', or 'larger'    
            '''
            #sigHo = {'two-sided':' =', 'smaller':'>=', 'larger':'<='}
            sigHa = {'two-sided':'!=', 'smaller':'< ', 'larger':'> '}
            print('--- Bootstrapping Method ---')
            #print('    Ho: measure', sigHo[alternative], np.round(test_value,2)) 
            print('    Ho: measure =', np.round(pop_value,2)) 
            print('    Ha: measure', sigHa[alternative], np.round(pop_value,2))  
            self.createSampleDistributionHo(pop_value)  
            print('    Sample mean = %.2f' %(self.sample_mean))
            p_val = self.graphpValue(self.sample_mean, alpha, alternative)
            print('    p-value = '+str(np.round(p_val,4)))
