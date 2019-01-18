import numpy as np
import sys

"""
	Arithmetic mean can be large when any or both of the elements are large. 
	However, Harmonic mean would be large only 
	when both of the numbers are individually large.
	(i.e. Harmonic mean is less affected by outlier value)


	Harmonic mean gets penalized more 
	when there is more gap between the participating numbers.

	Hence HMean/F1-score/F-score(more generally) considers both 
	precision and recall values instead of just one, and penalizes when one
	of them is low.

	The below code computes arithmetic / h-mean for an array of random numbers

"""


def arithmetic_mean(arr):
	return np.floor(np.mean(arr));

def harmonic_mean(arr):
	lcm = np.lcm.reduce(arr);
	denom = np.sum(lcm/arr);
	hmean = len(arr)*(lcm/denom)

	return np.floor(hmean);


N = int(sys.argv[1]);

# randomly choose N numbers
arr = np.random.randint(1, 100, size=(N,));
print(f'List -- {arr}');

# arithmetic mean
print(f'Arithmetic mean -- {arithmetic_mean(arr)}');

# harmonic mean
print(f'Harmonic mean -- {harmonic_mean(arr)}');
