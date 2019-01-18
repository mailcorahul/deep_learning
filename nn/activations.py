import numpy as np
import matplotlib.pyplot as plt

def visualize_activation(a):
	act = globals()[a];
	x = [];
	y = [];
	for i in np.arange(-5.0, 5.0, 0.1):
		x.append(i);
		y.append(act(i));

	plt.plot(x, y, 'ro');
	plt.axis([-5, 5, 0, 1]);
	plt.show();


def sigmoid(z):
	return 1./(1+np.exp(-z));

if __name__ == '__main__':
	visualize_activation('sigmoid');