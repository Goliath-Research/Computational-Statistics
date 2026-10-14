# This version uses numpy arrays instead of pandas DataFrames. It is faster!

import numpy as np
import pandas as pd



from abc import ABC, abstractmethod

from itertools import combinations

#import seaborn as sns






def _positive_count(value):
    """Validate and return a positive integer resample count.
    
    Accept Python and NumPy integers; reject booleans. Raise ValueError for
    noninteger or nonpositive values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("num_samples must be a positive integer")
    return int(value)


def _numeric_vector(values):
    """Return a nonempty one-dimensional array of finite real numbers.
    
    Preserve numeric dtype so resampled integer indices remain valid.
    Raise ValueError for empty, multidimensional, nonnumeric, complex,
    or nonfinite input."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("samples must contain real numeric values")
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("each sample must be a nonempty one-dimensional finite numeric array")
    return array


def _validate_groups(arrays, related=False):
    """Validate two or more groups and return their numeric NumPy arrays.
    
    Parameters
    ----------
    arrays : sequence of array-like
        Nonempty one-dimensional samples containing finite real numbers.
    related : bool, default False
        Require equal lengths for paired or repeated measurements.
    
    Returns
    -------
    list of numpy.ndarray
        Validated samples in their original order.
    
    Raises
    ------
    ValueError
        Fewer than two groups, invalid sample data, or unequal related lengths."""
    if len(arrays) < 2:
        raise ValueError("at least two samples are required")
    groups = [_numeric_vector(a) for a in arrays]
    if related and len({len(a) for a in groups}) != 1:
        raise ValueError("related samples must have equal lengths")
    return groups


class Samples(ABC):
    '''
    Abstract class for generating samples using a resampling method 
    '''

    def __init__(self, sample_data, num_samples=4_000, seed=2026):
        """Initialize a sample generator and generate resamples immediately.
        
        Parameters
        ----------
        sample_data : array-like
            Nonempty one-dimensional finite real observations.
        num_samples : int, default 4000
            Positive number of resamples.
        seed : int, default 2026
            Seed passed to NumPy default_rng.
        
        The cached column means are retained for backward compatibility."""
        self._sample_data = _numeric_vector(sample_data)
        self._num_samples = _positive_count(num_samples)
        self._sample_size = len(self._sample_data)
        self._rng = np.random.default_rng(seed)
        self.generate_samples()
        self._sample_distribution = self._samples.mean(axis=0)
        
        


    @abstractmethod
    def generate_samples(self):
        """Generate resamples in self._samples.
        
        Subclasses must create an array shaped (sample_size, num_samples)."""
        pass
    
    @property
    def sample_data(self) -> np.ndarray:
        """Return the original one-dimensional sample array."""
        return self._sample_data

    @property
    def sample_size(self) -> int:
        """Return the number of observations in the original sample."""
        return self._sample_size

    @property
    def num_samples(self) -> int:
        """Return the number of generated resamples."""
        return self._num_samples
    
    @num_samples.setter
    def num_samples(self, value: int) -> None:
        """Set a positive integer resample count and regenerate samples and means.
        
        The existing random generator advances; it is not reseeded.
        Raise ValueError for a boolean, noninteger, or nonpositive count."""
        self._num_samples = _positive_count(value)
        self.generate_samples()
        self._sample_distribution = self._samples.mean(axis=0)

    @property
    def samples(self) -> np.ndarray:
        """Return the resampled array with observations in rows and resamples in columns."""
        return self._samples  
    
    



class BSamples(Samples):
    '''
    Class for generating samples using the bootstrap method (resampling WITH replacement)    
    '''    

    def generate_samples(self):
        """Draw observations with replacement into independent resample columns."""
        self._samples = self._rng.choice(self._sample_data, replace=True, size=(self._sample_size, self._num_samples))
        
                         

    
class PSamples(Samples):
    '''
    Class for generating samples using the permutation method (resampling WITHOUT replacement)    
    '''
    
    def generate_samples(self):
        """Independently permute the original observations within each column.
        
        Each column preserves the original values and their multiplicities."""
        repeated = np.broadcast_to(self._sample_data[:, None], (self._sample_size, self._num_samples))
        self._samples = self._rng.permuted(repeated, axis=0)



class ResamplingHT(ABC):
    '''
    Abstract class for hypothesis testing using resampling methods (bootstrap and permutation) 
    '''
    
    def __init__(self):
        """Cache the subclass null distribution and observed statistic.
        
        Subclasses must initialize their group data and resampling settings first.
        The initial confidence level is 0.95."""
        self._sample_distribution: np.ndarray = self.get_sample_distribution()
        self._obs_stat:     float = self.get_observed_stat()
        self._p_value:      float = 1.0
        self._confidence:   float = 0.95
      

    @abstractmethod
    def get_sample_distribution(self) -> np.ndarray:
        """Calculate and return the null distribution for two or more groups.
        
        Dispatch to the two-group or multiple-group statistic. The constructor
        caches this result; null_distribution accesses the cache without recalculation."""
        pass


    @abstractmethod
    def get_observed_stat(self) -> float:
        """Calculate and return the observed statistic for two or more groups."""
        pass


    @property
    def observed_statistic(self):
        """Return the observed group-comparison statistic cached at construction."""
        return float(self._obs_stat)

    @property
    def null_distribution(self):
        """Return a copy of the cached resampling null distribution."""
        return self._sample_distribution.copy()

    @property
    def confidence(self) -> float:
        """Return the confidence level as a fraction, such as 0.95."""
        return self._confidence
    

    @confidence.setter
    def confidence(self, value: float) -> None:
        """Set the confidence level as a fraction strictly between zero and one.
        
        For example, 0.95 requests a 95% interval. Raise ValueError for an
        invalid confidence level. Changing it does not regenerate resamples."""
        if not isinstance(value, (float, int)) or not (0.0 <= value <= 1.0):
            raise ValueError("Confidence must be a float or int between 0.0 and 1.0 inclusive")
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("confidence must be strictly between 0 and 1")
        self._confidence = float(value)
    

    def get_confidence_interval(self, sample: np.ndarray) -> tuple:
        """Return a percentile interval for the supplied resampled statistics.
        
        Parameters
        ----------
        sample : array-like
            Finite resampled values of the statistic being estimated.
        
        Returns
        -------
        tuple
            Lower and upper percentiles determined by self.confidence.
        
        Notes
        -----
        Interpret as a confidence interval only for an appropriate bootstrap
        estimation distribution, not merely for a permutation null distribution."""
        alpha = 100 - self._confidence * 100               
        lower_alpha = alpha / 2.0    
        upper_alpha = lower_alpha + self._confidence * 100        
        lower_limit = np.percentile(sample, lower_alpha) 
        upper_limit = np.percentile(sample, upper_alpha)                 
        return (lower_limit, upper_limit)


    def get_p_value(self, alternative=None):
        """Return a Monte Carlo p-value from the cached null distribution.
        
        Parameters
        ----------
        alternative : {"two-sided", "smaller", "larger"} or None
            None selects two-sided for two groups and larger for more groups.
            Multiple-group nonnegative statistics require larger.
        
        Returns
        -------
        float
            Tail probability with inclusive comparisons and an add-one correction.
            Two-sided values are twice the smaller tail, capped at one.
        
        Raises
        ------
        ValueError
            An unsupported alternative or tail for a multiple-group statistic."""
        if alternative is None:
            alternative = "two-sided" if self._k == 2 else "larger"
        if alternative not in ("two-sided", "smaller", "larger"):
            raise ValueError("alternative must be 'two-sided', 'smaller', or 'larger'")
        if self._k > 2 and alternative != "larger":
            raise ValueError("the nonnegative multiple-group statistic requires alternative='larger'")
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
        """Return maximal subsets with no rejected Bonferroni pairwise comparisons.
        
        Parameters
        ----------
        alpha : float, default 0.05
            Family significance level strictly between zero and one.
        
        Returns
        -------
        list of list of int
            Zero-based group indices; subsets can overlap.
        
        Notes
        -----
        Non-rejection does not establish equality. Enumeration can be expensive
        when there are many groups.
        
        Raises
        ------
        ValueError
            alpha is outside the permitted range."""
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
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
        """Initialize a independent bootstrap comparison of two or more groups.
        
        Parameters
        ----------
        *arrays : array-like
            Nonempty one-dimensional finite real samples.
            Groups may have different lengths.
        num_samples : int, default 4000
            Positive number of Monte Carlo resamples.
        seed : int, default 2026
            Reproducible seed for resampling.
        
        Raises
        ------
        ValueError
            Invalid samples, group count, or resample count.
        
        Notes
        -----
        Bootstrap samples are centered to impose equal population means."""
        self._k = len(arrays)
        self._arrays = _validate_groups(arrays)
        self._num_samples = _positive_count(num_samples)
        self._seed = seed
        self._sample_size = len(self._arrays[0])
        self._overall_mean = np.concatenate(self._arrays).mean()
        self._corrected_p_values = np.ones((self._k, self._k))
        self._sample_objects = [BSamples(a, num_samples=num_samples, seed=seed + j) for j, a in enumerate(self._arrays)]
        self._overall_size = sum(len(a) for a in self._arrays)
        super().__init__()


    def get_sample_distribution_2(self) -> np.ndarray:
        """Return the first minus second group means for each null resample."""
        # Getting the two sample objects
        s0 = self._sample_objects[0]
        s1 = self._sample_objects[1]        
        # Shifting the samples for sharing the mean
        arr0 = s0.samples - s0.sample_data.mean() + self._overall_mean 
        arr1 = s1.samples - s1.sample_data.mean() + self._overall_mean    
        # Returning the difference in means   
        return arr0.mean(axis=0) - arr1.mean(axis=0)

        
    def get_sample_distribution_k(self):
        """Return the null statistic for every resample for more than two groups.
        
        Use the sample-size-weighted root mean square deviation of group means
        from the weighted grand mean."""
        means = np.stack([obj.samples.mean(axis=0) - obj.sample_data.mean() + self._overall_mean for obj in self._sample_objects])
        sizes = np.array([obj.sample_size for obj in self._sample_objects])
        grand = np.average(means, axis=0, weights=sizes)
        return np.sqrt(np.average((means - grand) ** 2, axis=0, weights=sizes))


    def get_sample_distribution(self) -> np.ndarray:
        """Calculate and return the null distribution for two or more groups.
        
        Dispatch to the two-group or multiple-group statistic. The constructor
        caches this result; null_distribution accesses the cache without recalculation."""
        return self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()        
    

    def get_observed_stat_2(self) -> float:
        """Return the first group mean minus the second group mean."""
        s0 = self._sample_objects[0]
        s1 = self._sample_objects[1]        
        return s0.sample_data.mean() - s1.sample_data.mean()        
           

    def get_observed_stat_k(self) -> float:
        """Return the observed statistic for more than two groups.
        
        Use the sample-size-weighted root mean square deviation of group means
        from the weighted grand mean."""
        stat = sum([(self._sample_objects[k].sample_data.mean() - self._overall_mean)**2 *
                            (self._sample_objects[k].sample_size / self._overall_size) 
                            for k in range(self._k)])        
        return np.sqrt(stat) 


    def get_observed_stat(self) -> float:
        """Calculate and return the observed statistic for two or more groups."""
        return self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()                


    def bonferroni_correction(self):
        """Return the cached Bonferroni-adjusted matrix of pairwise p-values.
        
        Each pair uses its own two-group resampling null and a reproducible seed.
        Multiply each p-value by k*(k-1)/2 and cap at one. The returned matrix
        is symmetric with diagonal entries equal to one; callers receive a copy."""
        if hasattr(self, "_bonferroni_cache"):
            return self._bonferroni_cache.copy()
        adjusted = np.ones((self._k, self._k))
        pairs = list(combinations(range(self._k), 2))
        for offset, (i, j) in enumerate(pairs):
            pair_model = type(self)(self._arrays[i], self._arrays[j], num_samples=self._num_samples, seed=self._seed + 100 + offset)
            adjusted[i, j] = adjusted[j, i] = min(1, pair_model.get_p_value("two-sided") * len(pairs))
        self._corrected_p_values = adjusted
        self._bonferroni_cache = adjusted.copy()
        return adjusted.copy()
    

    def confidence_intervals(self) -> np.ndarray:
        """Return percentile-bootstrap confidence intervals for group population means.
        
        Returns
        -------
        numpy.ndarray
            Shape (k, 2): lower and upper limits in original group order.
        
        Notes
        -----
        Use unshifted bootstrap observations and self.confidence. These intervals
        are for individual group means, not differences between means."""
        ci = [self.get_confidence_interval(self._sample_objects[k].samples.mean(axis=0))             
              for k in range(self._k)]
        return np.array(ci)
                
    



class BootstrapRelatedHT(ResamplingHT):
    '''
    Class for related samples hypothesis testing using the bootstrap method.
    '''

    def __init__(self, *arrays, num_samples=4_000, seed=2026):
        """Initialize a paired bootstrap comparison of two or more groups.
        
        Parameters
        ----------
        *arrays : array-like
            Nonempty one-dimensional finite real samples. Equal lengths are required;
            corresponding positions must identify the same subject.
        num_samples : int, default 4000
            Positive number of Monte Carlo resamples.
        seed : int, default 2026
            Reproducible seed for resampling.
        
        Raises
        ------
        ValueError
            Invalid samples, group count, or resample count.
        
        Notes
        -----
        Bootstrap samples are centered to impose equal population means."""
        self._k = len(arrays)
        self._arrays = _validate_groups(arrays, related=True)
        self._num_samples = _positive_count(num_samples)
        self._seed = seed
        self._sample_size = len(self._arrays[0])
        self._overall_mean = np.concatenate(self._arrays).mean()
        self._corrected_p_values = np.ones((self._k, self._k))
        self._boot_indexes = BSamples(np.arange(self._sample_size), num_samples=num_samples, seed=seed)
        super().__init__()
                 

    

    def get_sample_distribution_2(self) -> np.ndarray:
        """Return the first minus second group means for each null resample."""
        # Getting the two sets of samples
        s0 = self._arrays[0][self._boot_indexes.samples] - self._arrays[0].mean() + self._overall_mean           
        s1 = self._arrays[1][self._boot_indexes.samples] - self._arrays[1].mean() + self._overall_mean               
        # Computing the difference in means
        sample_dist = s0.mean(axis=0) - s1.mean(axis=0)         
        return sample_dist
        

    def get_sample_distribution_k(self) -> np.ndarray:
        """Return the null statistic for every resample for more than two groups.
        
        Use the mean absolute difference across all pairs of condition means."""
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
        """Calculate and return the null distribution for two or more groups.
        
        Dispatch to the two-group or multiple-group statistic. The constructor
        caches this result; null_distribution accesses the cache without recalculation."""
        return self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()
            
        
        
    def get_observed_stat_2(self) -> float:
        """Return the first group mean minus the second group mean."""
        # Getting the two sets of samples
        s0 = self._arrays[0]
        s1 = self._arrays[1]
        # Computing the difference in means
        observed_stat = s0.mean() - s1.mean()
        return observed_stat
    

    def get_observed_stat_k(self) -> float:
        """Return the observed statistic for more than two groups.
        
        Use the mean absolute difference across all pairs of condition means."""
        # Getting the k samples and shifting them for sharing the mean
        s = [self._arrays[i] for i in range(self._k)]
        # Getting the means of the k samples
        s_mean = [s[i].mean() for i in range(self._k)]
        # Getting all possible pairs of samples
        pairs = [np.abs(s_mean[i]-s_mean[j]) for i in range(self._k) for j in range(i+1, self._k)]
        observed_stat = np.mean(pairs)
        return observed_stat
    


    def get_observed_stat(self) -> float:
        """Calculate and return the observed statistic for two or more groups."""
        return self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()        
                


    def bonferroni_correction(self):
        """Return the cached Bonferroni-adjusted matrix of pairwise p-values.
        
        Each pair uses its own two-group resampling null and a reproducible seed.
        Multiply each p-value by k*(k-1)/2 and cap at one. The returned matrix
        is symmetric with diagonal entries equal to one; callers receive a copy."""
        if hasattr(self, "_bonferroni_cache"):
            return self._bonferroni_cache.copy()
        adjusted = np.ones((self._k, self._k))
        pairs = list(combinations(range(self._k), 2))
        for offset, (i, j) in enumerate(pairs):
            pair_model = type(self)(self._arrays[i], self._arrays[j], num_samples=self._num_samples, seed=self._seed + 100 + offset)
            adjusted[i, j] = adjusted[j, i] = min(1, pair_model.get_p_value("two-sided") * len(pairs))
        self._corrected_p_values = adjusted
        self._bonferroni_cache = adjusted.copy()
        return adjusted.copy()


    def confidence_intervals(self) -> np.ndarray:
        """Return percentile-bootstrap confidence intervals for group population means.
        
        Returns
        -------
        numpy.ndarray
            Shape (k, 2): lower and upper limits in original group order.
        
        Notes
        -----
        Use unshifted bootstrap observations and self.confidence. These intervals
        are for individual group means, not differences between means."""
        ci = [self.get_confidence_interval(self._arrays[k][self._boot_indexes.samples].mean(axis=0))           
              for k in range(self._k)]
        return np.array(ci)
            






class PermutationIndependentHT(ResamplingHT):
    '''
    Class for independent samples hypothesis testing using the permutation method.    
    '''

    def __init__(self, *arrays, num_samples=4_000, seed=2026):
        """Initialize a pooled-label permutation comparison of two or more groups.
        
        Parameters
        ----------
        *arrays : array-like
            Nonempty one-dimensional finite real samples.
            Groups may have different lengths.
        num_samples : int, default 4000
            Positive number of Monte Carlo resamples.
        seed : int, default 2026
            Reproducible seed for resampling.
        
        Raises
        ------
        ValueError
            Invalid samples, group count, or resample count.
        
        Notes
        -----
        Permutation inference requires exchangeable labels under the null."""
        self._k = len(arrays)
        self._arrays = _validate_groups(arrays)
        self._num_samples = _positive_count(num_samples)
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
        """Return the first minus second group means for each null resample."""
        # Getting the two sample objects
        #arr0 = self._arrays_list[0]
        #arr1 = self._arrays_list[1]
        arr0 = self._sample_list[0]   
        arr1 = self._sample_list[1]
        # Computing the difference in means
        return arr0.mean(axis=0) - arr1.mean(axis=0)        
            
        
    def get_sample_distribution_k(self) -> np.ndarray:    
        """Return the null statistic for every resample for more than two groups.
        
        Use the sample-size-weighted root mean square deviation of group means
        from the weighted grand mean."""
        #sample_dist = sum([((self._arrays_list[k].mean(axis=0) - self._overall_mean)**2) *  
        #                    (self._sample_sizes[k] / self._overall_size) 
        #                    for k in range(self._k)])        
        sample_dist = sum([((self._sample_list[k].mean(axis=0) - self._overall_mean)**2) *  
                            (self._sample_sizes[k] / self._overall_size) 
                            for k in range(self._k)])                          
        return np.sqrt(sample_dist)


    def get_sample_distribution(self) -> np.ndarray:
        """Calculate and return the null distribution for two or more groups.
        
        Dispatch to the two-group or multiple-group statistic. The constructor
        caches this result; null_distribution accesses the cache without recalculation."""
        return self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()
                

        
    def get_observed_stat_2(self) -> float:
        """Return the first group mean minus the second group mean."""
        return self._arrays[0].mean() - self._arrays[1].mean()                         
        
    
    def get_observed_stat_k(self) -> float:
        """Return the observed statistic for more than two groups.
        
        Use the sample-size-weighted root mean square deviation of group means
        from the weighted grand mean."""
        stat = sum([(self._arrays[k].mean() - self._overall_mean)**2 * 
                            (self._sample_sizes[k] / self._overall_size) 
                            for k in range(self._k)])                
        return np.sqrt(stat) 
    

    def get_observed_stat(self) -> float:
        """Calculate and return the observed statistic for two or more groups."""
        return self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()        
            

  
    def bonferroni_correction(self):
        """Return the cached Bonferroni-adjusted matrix of pairwise p-values.
        
        Each pair uses its own two-group resampling null and a reproducible seed.
        Multiply each p-value by k*(k-1)/2 and cap at one. The returned matrix
        is symmetric with diagonal entries equal to one; callers receive a copy."""
        if hasattr(self, "_bonferroni_cache"):
            return self._bonferroni_cache.copy()
        adjusted = np.ones((self._k, self._k))
        pairs = list(combinations(range(self._k), 2))
        for offset, (i, j) in enumerate(pairs):
            pair_model = type(self)(self._arrays[i], self._arrays[j], num_samples=self._num_samples, seed=self._seed + 100 + offset)
            adjusted[i, j] = adjusted[j, i] = min(1, pair_model.get_p_value("two-sided") * len(pairs))
        self._corrected_p_values = adjusted
        self._bonferroni_cache = adjusted.copy()
        return adjusted.copy()
        
    

    def confidence_intervals(self):
        """Return percentile-bootstrap confidence intervals for group population means.
        
        Returns
        -------
        numpy.ndarray
            Shape (k, 2): lower and upper limits in original group order.
        
        Notes
        -----
        Use unshifted bootstrap observations and self.confidence. These intervals
        are for individual group means, not differences between means.
        For this permutation class, generate a separate bootstrap with seed + 200."""
        bootstrap = BootstrapIndependentHT(*self._arrays, num_samples=self._num_samples, seed=self._seed + 200)
        bootstrap.confidence = self.confidence
        return bootstrap.confidence_intervals()
    





class PermutationRelatedHT(ResamplingHT):
    '''    
    Class for related samples hypothesis testing using the permutation method.    
    '''

    def __init__(self, *arrays, num_samples=4_000, seed=2026):
        """Initialize a within-subject permutation comparison of two or more groups.
        
        Parameters
        ----------
        *arrays : array-like
            Nonempty one-dimensional finite real samples. Equal lengths are required;
            corresponding positions must identify the same subject.
        num_samples : int, default 4000
            Positive number of Monte Carlo resamples.
        seed : int, default 2026
            Reproducible seed for resampling.
        
        Raises
        ------
        ValueError
            Invalid samples, group count, or resample count.
        
        Notes
        -----
        Permutation inference requires exchangeable labels under the null."""
        self._k = len(arrays)
        self._arrays = _validate_groups(arrays, related=True)
        self._num_samples = _positive_count(num_samples)
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
        """Return the first minus second group means for each null resample."""
        # Getting the two sample objects
        arr0 = self._sample_list[0]
        arr1 = self._sample_list[1]        
        # Computing the difference in means
        return arr0.mean(axis=0) - arr1.mean(axis=0)    
        

    def get_sample_distribution_k(self) -> np.ndarray:
        """Return the null statistic for every resample for more than two groups.
        
        Use the mean absolute difference across all pairs of condition means."""
        # Getting the means of the k samples
        sample_list_mean = [self._sample_list[i].mean(axis=0) for i in range(self._k)]
        # Getting all possible pairs of samples
        pairs = np.array([np.abs(sample_list_mean[i]-sample_list_mean[j]) 
                        for i in range(self._k) for j in range(i+1, self._k)])     
        return pairs.mean(axis=0)  
              


    def get_sample_distribution(self) -> np.ndarray:
        """Calculate and return the null distribution for two or more groups.
        
        Dispatch to the two-group or multiple-group statistic. The constructor
        caches this result; null_distribution accesses the cache without recalculation."""
        return self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()
              
        

    def get_observed_stat_2(self) -> float:
        """Return the first group mean minus the second group mean."""
        return self._arrays[0].mean() - self._arrays[1].mean()                         
        
        
    
    def get_observed_stat_k(self) -> float: 
        """Return the observed statistic for more than two groups.
        
        Use the mean absolute difference across all pairs of condition means."""
        # Get the means of the k samples
        sample_mean = [self._arrays[i].mean() for i in range(self._k)]           
        # Getting all possible pairs of samples
        pairs = np.array([np.abs(sample_mean[i] - sample_mean[j]) 
                 for i in range(self._k) for j in range(i+1, self._k)])              
        return pairs.mean()
    
    

    def get_observed_stat(self) -> float:
        """Calculate and return the observed statistic for two or more groups."""
        return self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()           


    def bonferroni_correction(self):
        """Return the cached Bonferroni-adjusted matrix of pairwise p-values.
        
        Each pair uses its own two-group resampling null and a reproducible seed.
        Multiply each p-value by k*(k-1)/2 and cap at one. The returned matrix
        is symmetric with diagonal entries equal to one; callers receive a copy."""
        if hasattr(self, "_bonferroni_cache"):
            return self._bonferroni_cache.copy()
        adjusted = np.ones((self._k, self._k))
        pairs = list(combinations(range(self._k), 2))
        for offset, (i, j) in enumerate(pairs):
            pair_model = type(self)(self._arrays[i], self._arrays[j], num_samples=self._num_samples, seed=self._seed + 100 + offset)
            adjusted[i, j] = adjusted[j, i] = min(1, pair_model.get_p_value("two-sided") * len(pairs))
        self._corrected_p_values = adjusted
        self._bonferroni_cache = adjusted.copy()
        return adjusted.copy()
        
    

    def confidence_intervals(self):
        """Return percentile-bootstrap confidence intervals for group population means.
        
        Returns
        -------
        numpy.ndarray
            Shape (k, 2): lower and upper limits in original group order.
        
        Notes
        -----
        Use unshifted bootstrap observations and self.confidence. These intervals
        are for individual group means, not differences between means.
        For this permutation class, generate a separate bootstrap with seed + 200."""
        bootstrap = BootstrapRelatedHT(*self._arrays, num_samples=self._num_samples, seed=self._seed + 200)
        bootstrap.confidence = self.confidence
        return bootstrap.confidence_intervals()
       
