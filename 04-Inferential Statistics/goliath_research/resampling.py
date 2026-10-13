# This version uses numpy arrays instead of pandas DataFrames. It is faster!

import numpy as np
import pandas as pd



from abc import ABC, abstractmethod

from itertools import combinations

#import seaborn as sns





class Samples(ABC):
    '''
    Abstract class for generating samples using a resampling method 
    '''

    def __init__(self, sample_data, num_samples=4_000, seed=2026):
        """Generate samples with observations in rows and resamples in columns."""
        self._sample_data = np.asarray(sample_data)
        self._num_samples = num_samples
        self._sample_size = len(self._sample_data)
        self._rng = np.random.default_rng(seed)
        self.generate_samples()
        self._sample_distribution = self._samples.mean(axis=0)
        
        


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
        self.generate_samples()
        self._sample_distribution = self._samples.mean(axis=0)

    @property
    def samples(self) -> np.ndarray:
        '''Access the NumPy array: observations in rows, resamples in columns'''
        return self._samples  
    
    



class BSamples(Samples):
    '''
    Class for generating samples using the bootstrap method (resampling WITH replacement)    
    '''    

    def generate_samples(self):
        """Draw all bootstrap observations with replacement using NumPy."""
        self._samples = self._rng.choice(self._sample_data, replace=True, size=(self._sample_size, self._num_samples))
        
                         

    
class PSamples(Samples):
    '''
    Class for generating samples using the permutation method (resampling WITHOUT replacement)    
    '''
    
    def generate_samples(self):
        """Independently shuffle each column without replacement."""
        repeated = np.broadcast_to(self._sample_data[:, None], (self._sample_size, self._num_samples))
        self._samples = self._rng.permuted(repeated, axis=0)



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


    def get_p_value(self, alternative=None):
        """Use two-sided differences for two samples, and upper-tail distances for k > 2."""
        if alternative is None:
            alternative = "two-sided" if self._k == 2 else "larger"
        values = np.asarray(self._sample_distribution)
        tolerance = 100 * np.finfo(float).eps * max(1, abs(self._obs_stat))
        left = (np.count_nonzero(values <= self._obs_stat + tolerance) + 1) / (len(values) + 1)
        right = (np.count_nonzero(values >= self._obs_stat - tolerance) + 1) / (len(values) + 1)
        if alternative == "smaller":
            return float(left)
        if alternative == "larger":
            return float(right)
        return float(min(1, 2 * min(left, right)))
    

    def get_homogeneous_subsets(self, alpha=0.05):
        """Return maximal subsets whose every pair is not rejected after Bonferroni."""
        adjusted = self.bonferroni_correction()
        compatible = []
        for size in range(1, self._k + 1):
            for subset in combinations(range(self._k), size):
                if all(adjusted[i, j] > alpha for i, j in combinations(subset, 2)):
                    compatible.append(set(subset))
        return [sorted(a) for a in compatible if not any(a < b for b in compatible)]
    





class BootstrapIndependentHT(ResamplingHT):
    '''
    Class for independent samples hypothesis testing using the bootstrap method.    
    '''

    def __init__(self, *arrays, num_samples=4_000, seed=2026):
        self._k = len(arrays)
        self._arrays = [np.asarray(a) for a in arrays]
        self._num_samples = num_samples
        self._seed = seed
        self._sample_size = len(self._arrays[0])
        self._overall_mean = np.concatenate(self._arrays).mean()
        self._corrected_p_values = np.ones((self._k, self._k))
        self._sample_objects = [BSamples(a, num_samples=num_samples, seed=seed + j) for j, a in enumerate(self._arrays)]
        self._overall_size = sum(len(a) for a in self._arrays)
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

        
    def get_sample_distribution_k(self):
        """Apply the observed weighted-dispersion definition to each null resample."""
        means = np.stack([obj.samples.mean(axis=0) - obj.sample_data.mean() + self._overall_mean for obj in self._sample_objects])
        sizes = np.array([obj.sample_size for obj in self._sample_objects])
        grand = np.average(means, axis=0, weights=sizes)
        return np.sqrt(np.average((means - grand) ** 2, axis=0, weights=sizes))


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
        """Test each pair using its own resampling null, then adjust across all pairs."""
        adjusted = np.ones((self._k, self._k))
        pairs = list(combinations(range(self._k), 2))
        for offset, (i, j) in enumerate(pairs):
            pair_model = type(self)(self._arrays[i], self._arrays[j], num_samples=self._num_samples, seed=self._seed + 100 + offset)
            adjusted[i, j] = adjusted[j, i] = min(1, pair_model.get_p_value("two-sided") * len(pairs))
        self._corrected_p_values = adjusted
        return adjusted
    

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

    def __init__(self, *arrays, num_samples=4_000, seed=2026):
        self._k = len(arrays)
        self._arrays = [np.asarray(a) for a in arrays]
        self._num_samples = num_samples
        self._seed = seed
        self._sample_size = len(self._arrays[0])
        self._overall_mean = np.concatenate(self._arrays).mean()
        self._corrected_p_values = np.ones((self._k, self._k))
        self._boot_indexes = BSamples(np.arange(self._sample_size), num_samples=num_samples, seed=seed)
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
        """Test each pair using its own resampling null, then adjust across all pairs."""
        adjusted = np.ones((self._k, self._k))
        pairs = list(combinations(range(self._k), 2))
        for offset, (i, j) in enumerate(pairs):
            pair_model = type(self)(self._arrays[i], self._arrays[j], num_samples=self._num_samples, seed=self._seed + 100 + offset)
            adjusted[i, j] = adjusted[j, i] = min(1, pair_model.get_p_value("two-sided") * len(pairs))
        self._corrected_p_values = adjusted
        return adjusted


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

    def __init__(self, *arrays, num_samples=4_000, seed=2026):
        self._k = len(arrays)
        self._arrays = [np.asarray(a) for a in arrays]
        self._num_samples = num_samples
        self._seed = seed
        self._sample_size = len(self._arrays[0])
        self._overall_mean = np.concatenate(self._arrays).mean()
        self._corrected_p_values = np.ones((self._k, self._k))
        self._sample_sizes = np.array([len(a) for a in self._arrays])
        self._overall_size = self._sample_sizes.sum()
        pooled = PSamples(np.concatenate(self._arrays), num_samples=num_samples, seed=seed)
        self._sample_list = np.split(pooled.samples, np.cumsum(self._sample_sizes)[:-1])
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
        """Test each pair using its own resampling null, then adjust across all pairs."""
        adjusted = np.ones((self._k, self._k))
        pairs = list(combinations(range(self._k), 2))
        for offset, (i, j) in enumerate(pairs):
            pair_model = type(self)(self._arrays[i], self._arrays[j], num_samples=self._num_samples, seed=self._seed + 100 + offset)
            adjusted[i, j] = adjusted[j, i] = min(1, pair_model.get_p_value("two-sided") * len(pairs))
        self._corrected_p_values = adjusted
        return adjusted
        
    

    def confidence_intervals(self):
        """Estimate group means with unshifted bootstrap draws, not permutations."""
        bootstrap = BootstrapIndependentHT(*self._arrays, num_samples=self._num_samples, seed=self._seed + 200)
        bootstrap.confidence = self.confidence
        return bootstrap.confidence_intervals()
    





class PermutationRelatedHT(ResamplingHT):
    '''    
    Class for related samples hypothesis testing using the permutation method.    
    '''

    def __init__(self, *arrays, num_samples=4_000, seed=2026):
        self._k = len(arrays)
        self._arrays = [np.asarray(a) for a in arrays]
        self._num_samples = num_samples
        self._seed = seed
        self._sample_size = len(self._arrays[0])
        self._overall_mean = np.concatenate(self._arrays).mean()
        self._corrected_p_values = np.ones((self._k, self._k))
        data = np.column_stack(self._arrays)
        rng = np.random.default_rng(seed)
        repeated = np.broadcast_to(data[:, :, None], (len(data), self._k, num_samples))
        shuffled = rng.permuted(repeated, axis=1)
        self._sample_list = [shuffled[:, j, :] for j in range(self._k)]
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
        """Test each pair using its own resampling null, then adjust across all pairs."""
        adjusted = np.ones((self._k, self._k))
        pairs = list(combinations(range(self._k), 2))
        for offset, (i, j) in enumerate(pairs):
            pair_model = type(self)(self._arrays[i], self._arrays[j], num_samples=self._num_samples, seed=self._seed + 100 + offset)
            adjusted[i, j] = adjusted[j, i] = min(1, pair_model.get_p_value("two-sided") * len(pairs))
        self._corrected_p_values = adjusted
        return adjusted
        
    

    def confidence_intervals(self):
        """Estimate group means with unshifted bootstrap draws, not permutations."""
        bootstrap = BootstrapRelatedHT(*self._arrays, num_samples=self._num_samples, seed=self._seed + 200)
        bootstrap.confidence = self.confidence
        return bootstrap.confidence_intervals()
       
