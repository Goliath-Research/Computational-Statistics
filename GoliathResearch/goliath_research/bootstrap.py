import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns;
from typing import Callable

sns.set_style("whitegrid")  

class Bootstrap(object):
    '''

    '''

    def __init__(self, sample_data, num_samples=10000, sample_size=100):
        '''    
        It creates the sample distribution using the bootstrap method (resampling WITH replacement)
        sample_data: sample data (must be representative of the population)
        num_samples: number of samples to generate for the bootstrap method
        sample_size: size of each of num_samples samples to generate
        '''
        self._sample_data = sample_data
        self._num_samples = num_samples
        self._sample_size = sample_size

        self.generateSamples()

    @property
    def sample_data(self):
        '''Sample of the original population (list or np.ndarray)'''
        return self._sample_data

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def sample_size(self):
        return self._sample_size

    @property
    def samples(self):
        '''DataFrame with num_samples samples of size sample_size'''
        if self._samples.empty:
            self.generateSamples()
        return self._samples

    def generateSamples(self):
        '''
        It returns a DataFrame where each column (num_samples columns) is a sample with replacement.
        '''
        self._samples = pd.DataFrame()
        for k in range(self.num_samples):
            sample = np.random.choice(self.sample_data, size=self.sample_size, replace=True)
            column_name = 'Sample' + str(k)
            self._samples[column_name] = sample
        self._samples = self._samples.copy()

# Method for generating a sample distribution from a DataFrame with sample columns
# This method receives all samples (a Pandas DataFrame) and the statistical function
# of interest (that is, a function that takes a sample and returns a statistic)

Statistics = Callable[np.ndarray, float]

def calcSampleDistribution(samples : pd.DataFrame, func : Statistics) -> np.ndarray :
    '''
    It creates the samples distribution using func.
    '''
    return np.array(samples.apply(func))

def getConfidenceInterval(d : np.ndarray, confidence: float = 95) -> (float, float):
    alpha = 100 - confidence
    lower_percentile = alpha / 2.0    
    lower = np.percentile(d, lower_percentile)
    upper_percentile = lower_percentile + confidence
    upper = np.percentile(d, upper_percentile)
    return (lower, upper)  

def graphSampleDistribution(d : np.ndarray):
    '''
    It graph the sample distribution
    '''
    sns.displot(d, height=3.5, aspect=1.5).set(title='Sample Distribution') 

def graphConfidenceInterval(d : np.ndarray, confidence: float = 95):
    '''
    It returns a confidence% confident interval and presents the result graphically.
    '''
    lower, upper = getConfidenceInterval(d, confidence)
    sns.displot(d, kde=True, height=3.5, aspect=1.5)
    plt.title('Sample Distribution\n %i%% CI: (%.2f, %.2f)'%(confidence, lower, upper))
    plt.axvline(x = lower, ymin=0, ymax=0.5, color='orangered', linewidth=2)
    plt.axvline(x = upper, ymin=0, ymax=0.5, color='orangered', linewidth=2)
    return (lower, upper)         

"""    def createSampleDistributionHo(self, pop_value):
        '''
        It creates the samples distribution centered at pop_value (true under Ho).
        '''
        self.sample_distribution_Ho = self.sample_distribution - self.sample_distribution.mean() + pop_value
        self.sample_mean_Ho = np.round(np.mean(self.sample_distribution_Ho), 2)
 """    


class OneSampleHT(Bootstrap):

    def __init__(self, sample_data, num_samples=10000, sample_size=100, func=np.mean):
        super().__init__(sample_data, num_samples, sample_size)

        self.sample_distribution = calcSampleDistribution(self._samples, func)
        self.sample_mean = np.round(np.mean(self.sample_distribution), 2)

    def createSampleDistributionHo(self, pop_value):
        '''
        It creates the samples distribution centered at pop_value (true under Ho).
        '''
        self.sample_distribution_Ho = self.sample_distribution - self.sample_distribution.mean() + pop_value
        self.sample_mean_Ho = np.round(np.mean(self.sample_distribution_Ho), 2)
    
    def getpValue(self, obs_value, alpha=0.05, alternative='two-sided'):
        '''
        It calculates the p-value for one-sample HT
        obs_value: obs_value: observed value 
        alpha: significance level
        alternative: one of the three values: 'two-sided', 'smaller', or 'larger'    
        '''
        ecdf = ECDF(self.sample_distribution_Ho) 
        if alternative == 'two-sided':
            if obs_value < self.sample_mean_Ho: 
                pValue = 2 * ecdf(obs_value)
            else: 
                pValue = 2 * (1 - ecdf(obs_value)) 
        elif alternative == 'smaller':
            pValue = ecdf(obs_value)
        else:
            pValue = 1 - ecdf(obs_value)
        return pValue

    def graphpValue(self, obs_value, alpha=0.05, alternative='two-sided'):
        '''
        It calculates the p-value for one-sample HT, and also graph the sample distribution, 
        the critical region, and the obs_value
        obs_value: obs_value: observed value
        alpha: significance level
        alternative: one of the three values: 'two-sided', 'smaller', or 'larger'    
        '''
        ecdf = ECDF(self.sample_distribution_Ho) 
        pValue = self.getpValue(obs_value, alpha, alternative) 
        ax = sns.kdeplot(x=self.sample_distribution_Ho, color='lightskyblue', shade=True, alpha=0.4)
        plt.axvline(x=obs_value, ymin=0, ymax= 0.02, color='black', linewidth=6)
        plt.title('Sampling Distribution')
        if alternative == 'two-sided':
            cv1 = np.round(np.percentile(self.sample_distribution_Ho, (alpha/2)*100), 2)    
            cv2 = np.round(np.percentile(self.sample_distribution_Ho, (1-alpha)*100), 2)     
            plt.axvline(x = cv1, ymin=0, ymax=0.5, color='orangered', linewidth=2)
            plt.axvline(x = cv2, ymin=0, ymax=0.5, color='orangered', linewidth=2);            
        elif alternative == 'smaller':
            cv1 = np.round(np.percentile(self.sample_distribution_Ho, alpha*100), 2)  
            plt.axvline(x = cv1, ymin=0, ymax=0.5, color='orangered', linewidth=2)
        else:
            cv2 = np.round(np.percentile(self.sample_distribution_Ho, (1-alpha)*100), 2)  
            plt.axvline(x = cv2, ymin=0, ymax=0.5, color='orangered', linewidth=2)
        return pValue

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
        print('    Ho: measure =', np.round(pop_value, 2)) 
        print('    Ha: measure'  , sigHa[alternative], np.round(pop_value, 2))    
        self.createSampleDistributionHo(pop_value)
        obs_value = self.sample_mean
        print('    Sample mean = %.2f' %(self.sample_mean))
        p_val = self.getpValue(obs_value, alpha, alternative)
        print('    p-value = ' + str(np.round(p_val,4)))
     
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
        print('    Ha: measure', sigHa[alternative], np.round(pop_value, 2))  
        self.createSampleDistributionHo(pop_value)  
        print('    Sample mean = %.2f' %(self.sample_mean))
        p_val = self.graphpValue(self.sample_mean, alpha, alternative)
        print('    p-value = ' + str(np.round(p_val, 4)))

if __name__ == "__main__":
    # Generate a data sample

    np.random.seed(10)
    data = np.random.randint(158, 175, 50)

    # Create a bootstrap object based on the previous sample
    # The object will generate multiple samples following the
    # bootstrap method (sampling with replacement)

    b = Bootstrap(data)

    # Once we have the bootstrap object, we can use it for
    # creating a sampling distribution of any "statistics"

    d = calcSampleDistribution(b.samples, np.mean)

    # Sample Distribution Histogram

    plt.hist(d)
    #input()

    # 95% confidence interval according to the Sample Distribution

    lower, upper = getConfidenceInterval(d, 95)
    print(lower, upper)

    # Graphical View

    graphSampleDistribution(d, np.mean)
    #input()

    graphConfidenceInterval(d, 95)
    #input()

    # Testing now OneSampleHT

    My1HT = OneSampleHT(data, sample_size=40)

    graphSampleDistribution(My1HT.sample_distribution, np.mean)

    pop_value = 160
    My1HT.createSampleDistributionHo(pop_value)

    obs_value = 165
    pValue = My1HT.getpValue(obs_value, 0.05, 'two-sided')
    print("pValue(%5.2f) = %3.2f" % (obs_value, pValue))

    My1HT.graphOneSampleHT(165)

