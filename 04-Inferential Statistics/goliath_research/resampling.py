# This version uses numpy arrays instead of pandas DataFrames. It is faster!

import numpy as np
import pandas as pd

from statsmodels.distributions.empirical_distribution import ECDF

from abc import ABC, abstractmethod

from itertools import combinations

#import seaborn as sns
import time




class Samples(ABC):
    '''
    Abstract class for generating samples using a resampling method 
    '''

    def __init__(self, sample_data: np.ndarray, num_samples: int = 10000):
        '''    
        It creates the sample distribution using a resampling (bootstrap or permutation)
        sample_data:    sample data (must be representative of the population)        
        num_samples:    number of samples to generate for the bootstrap method        
        '''
        self._sample_data: np.ndarray = sample_data         # sample data   
        self._num_samples: int = num_samples                # number of samples to generate
        self._sample_size: int = len(self._sample_data)     # sample size        
        self._samples: np.ndarray = np.zeros((self._sample_size, self._num_samples))
        
        self.generate_samples()

        self._sample_distribution: np.ndarray = self._samples.mean(axis=0)  # sample distribution
        
        


    @abstractmethod
    def generate_samples(self):
        pass
    
    @property
    def sample_data(self) -> np.ndarray:
        '''Sample of the original population (np.ndarray)'''
        return self._sample_data

    @property
    def sample_size(self) -> int:
        '''Sample size (positive integer)'''
        return self._sample_size

    @property
    def num_samples(self) -> int:
        '''Number of samples (positive integer)'''
        return self._num_samples
    
    @num_samples.setter
    def num_samples(self, value: int) -> None:
        '''Set the number of samples (expects a positive integer)'''
        if not isinstance(value, int) or value <= 0:
            raise ValueError("num_samples must be a positive integer")
        self._num_samples = value

    @property
    def samples(self) -> pd.DataFrame:
        '''Getter method for accessing the DataFrame of _samples'''
        return self._samples  
    
    



class BSamples(Samples):
    '''
    Class for generating samples using the bootstrap method (resampling WITH replacement)    
    '''    

    def generate_samples(self):
        '''
        It returns a DataFrame where each column (num_samples columns) is a sample with replacement.
        It uses _sample_data to generate the samples.
        '''                  
        # Generating the samples WITH replacement
        sample_boot = np.random.choice(self._sample_data, replace=True,
                                       size=(self._sample_size, self._num_samples))
        # Now, sample_boot is a 2D NumPy array with shape (self._sample_size, self._num_samples)
        self._samples = sample_boot
        
                         

    
class PSamples(Samples):
    '''
    Class for generating samples using the permutation method (resampling WITHOUT replacement)    
    '''
    
    def generate_samples(self):
        '''
        It returns a 2D NumPy array where each column is a sample without replacement.        
        '''
        n = len(self._sample_data)
        # Generate an array of indices (0, 1, 2, ..., n-1)
        indices = np.tile(np.arange(n), (self._num_samples, 1))
        # Shuffle each row of the indices array to create permutations
        np.apply_along_axis(np.random.shuffle, 1, indices)
        # Use the shuffled indices to generate the samples array
        self._samples = self._sample_data[indices.T]
        """
        n = len(self._sample_data)
        self._samples = np.empty((n, self._num_samples))  
        # Generate all permutations at once
        for i in range(self._num_samples):
            self._samples[:, i] = np.random.permutation(self._sample_data)
        """



class ResamplingHT(ABC):
    '''
    Abstract class for hypothesis testing using resampling methods (bootstrap and permutation) 
    '''
    
    def __init__(self):
        '''
        It initializes the class. 
        HypothesisTest class computes the p-value given a sample distribution and an observed 
        statistic.        
        '''
        self._sample_distribution: np.ndarray = self.get_sample_distribution()
        self._obs_stat:     float = self.get_observed_stat()
        self._p_value:      float = 1.0
        self._confidence:   float = 0.95
      

    @abstractmethod
    def get_sample_distribution(self) -> np.ndarray:
        pass


    @abstractmethod
    def get_observed_stat(self) -> float:
        pass


    @property
    def confidence(self) -> float:
        '''Getter method for accessing the confidence level'''
        return self._confidence
    

    @confidence.setter
    def confidence(self, value: float) -> None:
        '''Setter method for updating the confidence level'''
        if not isinstance(value, (float, int)) or not (0.0 <= value <= 1.0):
            raise ValueError("Confidence must be a float or int between 0.0 and 1.0 inclusive")
        self._confidence = float(value)
    

    def get_confidence_interval(self, sample: np.ndarray) -> tuple:
        '''
        It returns a confidence% confident interval for sample
        '''
        alpha = 100 - self._confidence * 100               
        lower_alpha = alpha / 2.0    
        upper_alpha = lower_alpha + self._confidence * 100        
        lower_limit = np.percentile(sample, lower_alpha) 
        upper_limit = np.percentile(sample, upper_alpha)                 
        return (lower_limit, upper_limit)


    def get_p_value(self, alternative='two-sided') -> float:
        '''
        It returns the p-value.                
        alternative: 'two-sided', 'smaller', or 'larger'    
        '''   
        ecdf = ECDF(self._sample_distribution)         
        if alternative=='two-sided':
            p_val = 2 * min(ecdf(self._obs_stat), 1 - ecdf(self._obs_stat))            
        elif alternative=='smaller':
            p_val = ecdf(self._obs_stat)
        else:   #alternative=='larger'
            p_val = 1-ecdf(self._obs_stat)
        #print('stat = %.4f    p-value: %.4f' %(self._obs_stat, p_val))
        return p_val
    

    def get_homogeneous_subsets(self, alpha=0.05):
        '''
        It returns a list of homogeneous subsets.
        '''
        # Getting homogeneous subsets          
        self.bonferroni_correction()        
        homog_subsets = []   
        for i in range(self._k):
            current_subset = [i]
            for j in range(self._k):
                if i == j:
                    continue
                if all(self._corrected_p_values[current_subset, j] > alpha):
                    current_subset.append(j) 
            # Sort and convert to 1-based index for better readability
            current_subset = sorted(current_subset)
            current_subset = [x for x in current_subset]
            if current_subset not in homog_subsets:
                homog_subsets.append(current_subset)     
        return homog_subsets
    





class BootstrapIndependentHT(ResamplingHT):
    '''
    Class for independent samples hypothesis testing using the bootstrap method.    
    '''

    def __init__(self, *arrays):
        '''
        (class BootstrapIndependentHT)
        Verifying the inputs and initializing the class                
        '''   
        # Verifying the inputs
        self._k = len(arrays)
        if self._k < 2:
            raise ValueError("At least 2 arrays are required.")   
        # Converting inputs to numpy arrays
        try:
            self._arrays = [np.array(array) for array in arrays]
        except:
            raise ValueError("Failed to convert inputs to NumPy arrays.")   
        # Converting arrays to Sample objects
        self._sample_objects = [BSamples(array) for array in arrays]

        # Getting the overall size
        self._overall_size: int = sum([s.sample_size for s in self._sample_objects])
        # Getting the overall mean                        
        self._overall_mean: float = np.concatenate(self._arrays).mean()
        # Initializing _corrected_p_values for Bonferroni Test
        self._corrected_p_values = np.ones((self._k, self._k))
        # Initializing the parent class
        super().__init__()               


    def get_sample_distribution_2(self) -> np.ndarray:
        '''
        (class BootstrapIndependentHT)
        Difference in means statistic
        '''               
        # Getting the two sample objects
        s0 = self._sample_objects[0]
        s1 = self._sample_objects[1]        
        # Shifting the samples for sharing the mean
        arr0 = s0.samples - s0.sample_data.mean() + self._overall_mean 
        arr1 = s1.samples - s1.sample_data.mean() + self._overall_mean    
        # Returning the difference in means   
        return arr0.mean(axis=0) - arr1.mean(axis=0)

        
    def get_sample_distribution_k(self) -> np.ndarray:    
        '''        
        (class BootstrapIndependentHT)
        The sampling distribution will be the square of... k samples (k>2)
        '''     
        # Shifting the samples for sharing the mean
        arr_shifted = [self._sample_objects[k].samples - self._sample_objects[k].sample_data.mean() 
                            + self._overall_mean for k in range(self._k)]
        # Computing the statistic
        sample_dist = sum([((arr_shifted[k].mean(axis=0) - self._overall_mean)**2) *  
                            (self._sample_objects[k].sample_size / self._overall_size) 
                            for k in range(self._k)])                     
        return np.sqrt(sample_dist)


    def get_sample_distribution(self) -> np.ndarray:
        '''
        (class BootstrapIndependentHT)
        It returns a np.array with the sample distribution based on means.
        '''          
        return self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()        
    

    def get_observed_stat_2(self) -> float:
        '''
        (class BootstrapIndependentHT)
        Difference in means statistic
        '''                
        s0 = self._sample_objects[0]
        s1 = self._sample_objects[1]        
        return s0.sample_data.mean() - s1.sample_data.mean()        
           

    def get_observed_stat_k(self) -> float:
        '''
        (class BootstrapIndependentHT)
        The observed statistic will be will be the square of... k samples (k>2)
        '''                  
        stat = sum([(self._sample_objects[k].sample_data.mean() - self._overall_mean)**2 *
                            (self._sample_objects[k].sample_size / self._overall_size) 
                            for k in range(self._k)])        
        return np.sqrt(stat) 


    def get_observed_stat(self) -> float:
        '''
        (class BootstrapIndependentHT)
        It returns the observed statistic.        
        '''
        return self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()                


    def bonferroni_correction(self):
        '''
        (class BootstrapIndependentHT)
        It compute a bootstrap test for all possible pairs of groups, adjusting the p-values
        according to the Bonferroni correction.        
        '''
        pairwise_p_values = np.ones((self._k, self._k))
        for pair in combinations(range(self._k), 2): 
            # Computing the sample distribution for pair            
            s0 = self._sample_objects[pair[0]]
            s1 = self._sample_objects[pair[1]]        
            # Computing the overall mean of the pair
            overall_mean = np.concatenate((s0.sample_data, s1.sample_data)).mean()
            # Shifting the samples for sharing the mean
            arr0 = s0.samples - s0.sample_data.mean() + overall_mean 
            arr1 = s1.samples - s1.sample_data.mean() + overall_mean    
            # Returning the difference in means   
            self._sample_distribution = arr0.mean(axis=0) - arr1.mean(axis=0)
            # Computing the observed statistic for pair            
            self._obs_stat = s0.sample_data.mean() - s1.sample_data.mean()                 
            # Computing the p-value for pair            
            pairwise_p_values[pair[0]][pair[1]] = self.get_p_value()
            pairwise_p_values[pair[1]][pair[0]] = pairwise_p_values[pair[0]][pair[1]]
        # Calculate the number of unique pairwise comparisons        
        num_comparisons = self._k * (self._k - 1) / 2
        # Perform Bonferroni correction         
        self._corrected_p_values = pairwise_p_values * num_comparisons
        self._corrected_p_values = np.minimum(self._corrected_p_values, 1.0)
        self._corrected_p_values = np.maximum(self._corrected_p_values, 0.0)        
        # Updating self._sample_distribution and self._obs_stat
        self._sample_distribution = self.get_sample_distribution()
        self._obs_stat = self.get_observed_stat()   
        return self._corrected_p_values 
    

    def confidence_intervals(self) -> np.ndarray:
        '''    
        (class BootstrapIndependentHT)    
        It returns a np.array with the confidence intervals for each sample data.
        '''
        ci = [self.get_confidence_interval(self._sample_objects[k].samples.mean(axis=0))             
              for k in range(self._k)]
        return np.array(ci)
                
    



class BootstrapRelatedHT(ResamplingHT):
    '''
    Class for related samples hypothesis testing using the bootstrap method.
    '''

    def __init__(self, *arrays):
        '''
        Verifying the inputs and initializing the class  
        '''   
        # Verifying the inputs            
        self._k = len(arrays)
        if self._k < 2:
            raise ValueError("At least 2 samples are required.")     
        
        # Converting inputs to numpy arrays
        try:
            self._arrays = [np.array(array) for array in arrays]
        except:
            raise ValueError('Failed to convert inputs to NumPy arrays.')    
        
        # All arrays must have the same size
        self._sample_size = len(arrays[0])
        try:
            assert all(len(sample) == self._sample_size for sample in self._arrays)
        except:            
            raise ValueError("All samples must be the same length.")    
        # The block resampling procedure respects the observation pairing of 
        # blocks of related observations.   
              
        # Generating indexes        
        self._boot_indexes = BSamples(np.arange(self._sample_size)) 
        
        # Getting the overall mean
        self._overall_mean: float = np.concatenate(self._arrays).mean()
        
        # Initializing the parent class
        super().__init__()          
                 

    

    def get_sample_distribution_2(self) -> np.ndarray:
        '''                
        (class BootstrapRelatedHT)
        Difference in means statistic
        '''            
        # Getting the two sets of samples
        s0 = self._arrays[0][self._boot_indexes.samples] - self._arrays[0].mean() + self._overall_mean           
        s1 = self._arrays[1][self._boot_indexes.samples] - self._arrays[1].mean() + self._overall_mean               
        # Computing the difference in means
        sample_dist = s0.mean(axis=0) - s1.mean(axis=0)         
        return sample_dist
        

    def get_sample_distribution_k(self) -> np.ndarray:
        '''        
        (class BootstrapRelatedHT)
        It gets the sample distribution based on mean_difference method.  
        '''            
        # Getting the k samples
        s = [self._arrays[i][self._boot_indexes.samples] - self._arrays[i].mean() 
             + self._overall_mean for i in range(self._k)]   
        s_mean = [s[i].mean(axis=0) for i in range(self._k)]                   
        pairs = [np.abs(s_mean[i]-s_mean[j]) for i in range(self._k) for j in range(i+1, self._k)]        
        # Stack the arrays vertically
        stacked_arrays = np.vstack(pairs)
        # Compute the mean along the vertical axis (axis=0)
        sample_dist = np.mean(stacked_arrays, axis=0)     
        return sample_dist



    def get_sample_distribution(self) -> np.ndarray:
        '''
        (class BootstrapRelatedHT)
        It returns a np.array with the sample distribution based on means.
        '''            
        return self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()
            
        
        
    def get_observed_stat_2(self) -> float:
        '''
        (class BootstrapRelatedHT)
        It returns the difference in means of the two samples.
        '''        
        # Getting the two sets of samples
        s0 = self._arrays[0]
        s1 = self._arrays[1]
        # Computing the difference in means
        observed_stat = s0.mean() - s1.mean()
        return observed_stat
    

    def get_observed_stat_k(self) -> float:
        '''
        (class BootstrapRelatedHT)
        It returns the mean of the absolute value of the differences of all pairs of samples.
        '''        
        # Getting the k samples and shifting them for sharing the mean
        s = [self._arrays[i] for i in range(self._k)]
        # Getting the means of the k samples
        s_mean = [s[i].mean() for i in range(self._k)]
        # Getting all possible pairs of samples
        pairs = [np.abs(s_mean[i]-s_mean[j]) for i in range(self._k) for j in range(i+1, self._k)]
        observed_stat = np.mean(pairs)
        return observed_stat
    


    def get_observed_stat(self) -> float:
        '''
        (class BootstrapRelatedHT)
        The observed statistic is mean_difference method computed on the actual observed data. 
        '''  
        return self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()        
                


    def bonferroni_correction(self):
        '''
        (class BootstrapRelatedHT)
        It compute a bootstrap test for all possible pairs of groups, adjusting the p-values
        according to the Bonferroni correction.        
        '''        
        pairwise_p_values = np.ones((self._k, self._k))
        for pair in combinations(range(self._k), 2): 
            # Computing the sample distribution for pair            
            s0 = self._arrays[pair[0]][self._boot_indexes.samples]
            s1 = self._arrays[pair[1]][self._boot_indexes.samples]              
            # Computing the overall mean of the pair
            overall_mean = np.concatenate((s0, s1)).mean()            
            # Shifting the samples for sharing the mean
            s0_shifted = s0 - self._arrays[pair[0]].mean() + overall_mean 
            s1_shifted = s1 - self._arrays[pair[1]].mean() + overall_mean    
            # Returning the difference in means   
            self._sample_distribution = s0_shifted.mean(axis=0) - s1_shifted.mean(axis=0)                          
            # Computing the observed statistic for pair                    
            self._obs_stat = self._arrays[pair[0]].mean() - self._arrays[pair[1]].mean()            
            # Computing the p-value for pair            
            pairwise_p_values[pair[0]][pair[1]] = self.get_p_value()
            pairwise_p_values[pair[1]][pair[0]] = pairwise_p_values[pair[0]][pair[1]]
        # Calculate the number of unique pairwise comparisons        
        num_comparisons = self._k * (self._k - 1) / 2
        # Perform Bonferroni correction         
        self._corrected_p_values = pairwise_p_values * num_comparisons
        self._corrected_p_values = np.minimum(self._corrected_p_values, 1.0)
        self._corrected_p_values = np.maximum(self._corrected_p_values, 0.0)               
        # Updating self._sample_distribution and self._obs_stat
        self._sample_distribution = self.get_sample_distribution()
        self._obs_stat = self.get_observed_stat()   
        return self._corrected_p_values 


    def confidence_intervals(self) -> np.ndarray:
        '''   
        (class BootstrapRelatedHT)            
        It returns a np.array with the confidence intervals for each sample data.
        '''
        ci = [self.get_confidence_interval(self._arrays[k][self._boot_indexes.samples].mean(axis=0))           
              for k in range(self._k)]
        return np.array(ci)
            






class PermutationIndependentHT(ResamplingHT):
    '''
    Class for independent samples hypothesis testing using the permutation method.    
    '''

    def __init__(self, *arrays):
        '''
        Verifying the inputs and initializing the class                
        '''   
        # Verifying the inputs        
        self._k = len(arrays)
        if self._k < 2:
            raise ValueError("At least 2 samples are required.")   
        
        # Converting inputs to numpy arrays
        try:
            self._arrays = [np.array(array) for array in arrays]
        except:
            raise ValueError("Failed to convert inputs to NumPy arrays.")  
        
        # Getting every sample size
        self._sample_sizes: np.array = np.array([len(s) for s in self._arrays])        
        
        # Getting the shuffled np.ndarray
        #self._sample_objects = PSamples(np.concatenate(self._arrays))
        sample_objects = PSamples(np.concatenate(self._arrays))

        # Getting a list of _k shuffled NumPy arrays             
        self._sample_list = np.split(sample_objects.samples, np.cumsum(self._sample_sizes)[:-1])        

        # Getting the overall size
        self._overall_size: int = self._sample_sizes.sum()        

        # Getting the overall mean                        
        self._overall_mean: float = np.concatenate(self._arrays).mean()        

        # Initializing the parent class
        super().__init__()     



    def get_sample_distribution_2(self) -> np.ndarray:
        '''
        (class PermutationIndependentHT)
        Difference in means statistic
        '''   
        # Getting the two sample objects
        #arr0 = self._arrays_list[0]
        #arr1 = self._arrays_list[1]
        arr0 = self._sample_list[0]   
        arr1 = self._sample_list[1]
        # Computing the difference in means
        return arr0.mean(axis=0) - arr1.mean(axis=0)        
            
        
    def get_sample_distribution_k(self) -> np.ndarray:    
        '''
        (class PermutationIndependentHT)
        The sampling distribution will be the square of... k samples (k>2)
        '''        
        #sample_dist = sum([((self._arrays_list[k].mean(axis=0) - self._overall_mean)**2) *  
        #                    (self._sample_sizes[k] / self._overall_size) 
        #                    for k in range(self._k)])        
        sample_dist = sum([((self._sample_list[k].mean(axis=0) - self._overall_mean)**2) *  
                            (self._sample_sizes[k] / self._overall_size) 
                            for k in range(self._k)])                          
        return np.sqrt(sample_dist)


    def get_sample_distribution(self) -> np.ndarray:
        '''
        (class PermutationIndependentHT)
        It returns a np.array with the sample distribution based on means.
        '''      
        return self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()
                

        
    def get_observed_stat_2(self) -> float:
        '''
        (class PermutationIndependentHT)
        Difference in means statistic
        '''     
        return self._arrays[0].mean() - self._arrays[1].mean()                         
        
    
    def get_observed_stat_k(self) -> float:
        '''
        (class PermutationIndependentHT)
        The observed statistic will be will be the square of... k samples (k>2)
        '''                  
        stat = sum([(self._arrays[k].mean() - self._overall_mean)**2 * 
                            (self._sample_sizes[k] / self._overall_size) 
                            for k in range(self._k)])                
        return np.sqrt(stat) 
    

    def get_observed_stat(self) -> float:
        '''
        (class PermutationIndependentHT)
        It returns the observed statistic.        
        '''
        return self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()        
            

  
    def bonferroni_correction(self):
        '''
        (class PermutationIndependentHT)
        It compute a bootstrap test for all possible pairs of groups, adjusting the p-values
        according to the Bonferroni correction.        
        '''
        pairwise_p_values = np.ones((self._k, self._k))
        for pair in combinations(range(self._k), 2): 
            # Computing the sample distribution for pair     
            #arr0 = self._arrays_list[pair[0]]
            #arr1 = self._arrays_list[pair[1]]   
            arr0 = self._sample_list[pair[0]]
            arr1 = self._sample_list[pair[1]]
            # Computing the difference in means
            self._sample_distribution = arr0.mean(axis=0) - arr1.mean(axis=0)    
            # Computing the observed statistic for pair      
            self._obs_stat = self._arrays[pair[0]].mean() - self._arrays[pair[1]].mean()      
            # Computing the p-value for pair            
            pairwise_p_values[pair[0]][pair[1]] = self.get_p_value()
            pairwise_p_values[pair[1]][pair[0]] = pairwise_p_values[pair[0]][pair[1]]
        # Calculate the number of unique pairwise comparisons        
        num_comparisons = self._k * (self._k - 1) / 2
        # Perform Bonferroni correction         
        self._corrected_p_values = pairwise_p_values * num_comparisons
        self._corrected_p_values = np.minimum(self._corrected_p_values, 1.0)
        self._corrected_p_values = np.maximum(self._corrected_p_values, 0.0)        
        # Updating self._sample_distribution and self._obs_stat
        self._sample_distribution = self.get_sample_distribution()
        self._obs_stat = self.get_observed_stat()           
        return self._corrected_p_values 
        
    

    def confidence_intervals(self) -> np.ndarray:
        '''      
        (class PermutationIndependentHT)
        It returns a np.array with the confidence intervals for each sample data.
        '''
        ci = [self.get_confidence_interval(BSamples(self._arrays[k], num_samples = 100).samples.mean(axis=0))           
              for k in range(self._k)]
        return np.array(ci)
    





class PermutationRelatedHT(ResamplingHT):
    '''    
    Class for related samples hypothesis testing using the permutation method.    
    '''

    def __init__(self, *arrays):
        '''
        Verifying the inputs and initializing the class       
        '''   
        # Verifying the inputs
        self._k = len(arrays)
        if self._k < 2:
            raise ValueError("At least 2 samples are required.")  
            
        # Converting inputs to numpy arrays 
        try:
            self._arrays = [np.array(array) for array in arrays]            
        except:            
            raise ValueError("Failed to convert inputs to NumPy arrays.")
        
        # All arrays must have the same size
        self._sample_size = len(arrays[0])
        try:
            assert all(len(sample) == self._sample_size for sample in self._arrays)
        except:            
            raise ValueError("All samples must be the same length.")    
                
        # The block resampling procedure respects the observation pairing of 
        # blocks of related observations.    
        sample_objects = np.array([PSamples(np.array(pair)) for pair in zip(*self._arrays)]) 
        
        # Creating _sample_list
        shape_sample = (self._sample_size, len(sample_objects[0].samples[0]))            
        self._sample_list = np.empty((self._k, shape_sample[0], shape_sample[1]))
                
        # Vectorized operation
        samples_matrix = np.array([[obj.samples[i] for obj in sample_objects] for i in range(self._k)])
        # Assign to _sample_list
        self._sample_list = samples_matrix        
                       
        # Initializing the parent class
        super().__init__()   


    
    def get_sample_distribution_2(self) -> np.ndarray:
        '''
        (class PermutationRelatedHT)
        Difference in means statistic
        '''           
        # Getting the two sample objects
        arr0 = self._sample_list[0]
        arr1 = self._sample_list[1]        
        # Computing the difference in means
        return arr0.mean(axis=0) - arr1.mean(axis=0)    
        

    def get_sample_distribution_k(self) -> np.ndarray:
        '''
        (class PermutationRelatedHT)
        It gets the sample distribution for k samples (k>2) based on mean_difference method.  
        '''    
        # Getting the means of the k samples
        sample_list_mean = [self._sample_list[i].mean(axis=0) for i in range(self._k)]
        # Getting all possible pairs of samples
        pairs = np.array([np.abs(sample_list_mean[i]-sample_list_mean[j]) 
                        for i in range(self._k) for j in range(i+1, self._k)])     
        return pairs.mean(axis=0)  
              


    def get_sample_distribution(self) -> np.ndarray:
        '''
        (class PermutationRelatedHT)
        It returns a np.array with the sample distribution based on means.
        '''      
        return self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()
              
        

    def get_observed_stat_2(self) -> float:
        '''
        (class PermutationRelatedHT)
        Difference in means statistic
        '''     
        return self._arrays[0].mean() - self._arrays[1].mean()                         
        
        
    
    def get_observed_stat_k(self) -> float: 
        '''
        (class PermutationRelatedHT)
        The observed statistic will be will be the square of... k samples (k>2)
        '''             
        # Get the means of the k samples
        sample_mean = [self._arrays[i].mean() for i in range(self._k)]           
        # Getting all possible pairs of samples
        pairs = np.array([np.abs(sample_mean[i] - sample_mean[j]) 
                 for i in range(self._k) for j in range(i+1, self._k)])              
        return pairs.mean()
    
    

    def get_observed_stat(self) -> float:
        '''
        (class PermutationRelatedHT)
        It returns the observed statistic.        
        '''
        return self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()           


    def bonferroni_correction(self):
        '''        
        (class PermutationRelatedHT)
        It compute a bootstrap test for all possible pairs of groups, adjusting the p-values
        according to the Bonferroni correction.        
        '''
        pairwise_p_values = np.ones((self._k, self._k))
        for pair in combinations(range(self._k), 2): 
            # Computing the sample distribution for pair     
            arr0 = self._sample_list[pair[0]]
            arr1 = self._sample_list[pair[1]]   
            # Computing the difference in means
            self._sample_distribution = arr0.mean(axis=0) - arr1.mean(axis=0)    
            # Computing the observed statistic for pair      
            self._obs_stat = self._arrays[pair[0]].mean() - self._arrays[pair[1]].mean()      
            # Computing the p-value for pair            
            pairwise_p_values[pair[0]][pair[1]] = self.get_p_value()
            pairwise_p_values[pair[1]][pair[0]] = pairwise_p_values[pair[0]][pair[1]]
        # Calculate the number of unique pairwise comparisons        
        num_comparisons = self._k * (self._k - 1) / 2
        # Perform Bonferroni correction         
        self._corrected_p_values = pairwise_p_values * num_comparisons
        self._corrected_p_values = np.minimum(self._corrected_p_values, 1.0)
        self._corrected_p_values = np.maximum(self._corrected_p_values, 0.0)        
        # Updating self._sample_distribution and self._obs_stat
        self._sample_distribution = self.get_sample_distribution()
        self._obs_stat = self.get_observed_stat()           
        return self._corrected_p_values 
        
    

    def confidence_intervals(self) -> np.ndarray:
        '''          
        (class PermutationRelatedHT)
        It returns a np.array with the confidence intervals for each sample data.
        '''
        ci = [self.get_confidence_interval(BSamples(self._arrays[k], 
                    num_samples = 100).samples.mean(axis=0))  for k in range(self._k)]
        return np.array(ci)
       