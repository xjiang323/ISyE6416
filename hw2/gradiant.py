import numpy as np
import matplotlib.pyplot as plt

def get_error(b, m, x, y):
	total_error = 0
	for i in range(0, len(x)):
		total_error += (y[i] - (m * x[i] + b)) ** 2
	return total_error/float(len(x))

def step_gradient(b_current, m_current, x, y, learning_rate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(x))
	for i in range(0, len(x)):
		b_gradient += -(2/N) * (y[i] - (m_gradient * x[i] + b_current))
		m_gradient += -(2/N) * x[i] * (y[i] - (m_current * x[i] + b_current))
	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b, new_m]

def runner(x, y, starting_b, strating_m, learning_rate, num_iterations):
	b = starting_b
	m = strating_m
	for i in range(num_iterations):
		b,m = step_gradient(b, m, x, y, learning_rate)
	return [b, m]

def run():
	y = np.genfromtxt('ry.dat', dtype = None, delimiter = ',')
	x = np.genfromtxt('rx.dat', dtype = None, delimiter = ',')
	plt.scatter(x, y, color = "m")
	learning_rate = 0.0001
	initial_m = 0
	initial_b = 0
	num_iterations = 200
	print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, get_error(initial_b, initial_m, x, y)))
	print("Running...")
	[b, m] = runner(x, y, initial_b,initial_m, learning_rate, num_iterations)
	y_pred = m * x + b
	plt.plot(x, y_pred, color = "g")
	
	print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, get_error(b, m, x, y)))

if __name__ == "__main__":
	run()
