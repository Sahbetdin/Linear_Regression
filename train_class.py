from config import Parameters, get_config, bcolors
from sys import exit
from re import search, compile
from random import sample, seed, random
import matplotlib.pyplot as plt

class LinRegression:
	def __init__(self, prms):
		self.file_path = prms.file
		self.read_file()
		self.is_header = prms.is_header
		self.get_header()
		self.strip_lines()
		if self.check_if_lines_good() is None: exit()
		self.convert_to_int()
		self.test_ratio = prms.test_ratio
		self.split_indices()
		self.split_dataset()
		self.method = prms.use_method
		self.n_epochs = prms.n_epochs
		self.lr = prms.learning_rate
		self.plot_losses = prms.plot_losses
		self.losses_train = None
		assert 2.e-3 < self.lr < 0.1, "Adjust learning rate"
		assert 4 < self.n_epochs < 10001, "Adjust n_epochs"
		self.print_results = prms.print_results
		self.plot_graphes = prms.plot_graphes
		self.apply_method()
		self.print_results_at_the_end()
		self.save_theta_to_file()

	def read_file(self):
		#open file
		try:
			f = open(self.file_path, "r")
		except FileNotFoundError:   #file is not found
			print("File does not exist")
			exit()
		except:
			print("Other error")
			exit()
		#read file
		self.lines = f.readlines()

	def save_theta_to_file(self):
		with open("thetas.txt", "w") as f:
			f.write(f"{self.theta[0]}\n{self.theta[1]}")

	def get_header(self):
		#verify data in file
		if self.is_header: 
			pattern_header = compile("[A-Za-z]+,[A-Za-z]+")
			is_header_good = search(pattern_header, self.lines[0])
			if is_header_good is None or is_header_good.group(0) != self.lines[0].strip():
				print("Header should consist of only letters and comma")
				exit()
			self.feat_label, self.target_label = self.lines[0].split(',')
			self.target_label = self.target_label.strip()
			self.lines = self.lines[1:]
		else:
			self.feat_label, self.target_label = "km", "price"

	def strip_lines(self):
		#strip \n
		for i, l in enumerate(self.lines):
			self.lines[i] = self.lines[i].strip()

	def check_if_lines_good(self):
		#check that other lines after header consist of only digits
		pattern_body = compile("[+-]?\d+,\d+")
		for i, l in enumerate(self.lines):
			is_line_good = search(pattern_body, l)
			if is_line_good is None or is_line_good.group(0) != l:
				print(f"This line is bad: {bcolors.WARNING} {l} {bcolors.ENDC}")
				return None
		return True

	def convert_to_int(self):
		self.n = len(self.lines)
		self.X = [None] * self.n
		self.Y = [None] * self.n
		for i, l in enumerate(self.lines):
			self.X[i], self.Y[i] = map(int, l.split(','))
		assert len(self.X) == len(self.Y)

	def apply_method(self):
		if self.method == "gd":
			self.GD_method()
		elif self.method == "ls":
			self.LSE_method()
		else:
			raise NotImplementedError(f"{bcolors.FAIL}Please provide one method: \
gd/ls{bcolors.ENDC}")

	def split_indices(self): 
		#split to train and test datasets
		assert 0.1 <= self.test_ratio <= 0.3, "Adjust 0.1 <= test_ratio <= 0.3"
		seed(21)
		self.n_test = int(self.n * self.test_ratio)
		self.n_train = self.n - self.n_test
		self.idx_train = sample(range(self.n), self.n_train)   #train indices
		self.idx_test = list(set(range(self.n)) - set(self.idx_train))

	def split_dataset(self):
		self.X_train = [self.X[i] for i in self.idx_train]
		self.Y_train = [self.Y[i] for i in self.idx_train]
		self.X_test = [self.X[i] for i in self.idx_test]
		self.Y_test = [self.Y[i] for i in self.idx_test]

	def print_results_at_the_end(self):
		if self.print_results:
			print("Least Square Error", end='') if self.method == 'ls' else print("Gradient Descent", end='')
			print(" method.")
			print(f"theta[0] = {self.theta[0]}")
			print(f"theta[1] = {self.theta[1]}")
			print(f"R2-train: {self.r2_train}")	
			print(f"R2-test:  {self.r2_test}")

	@staticmethod
	def linear_function(lst, theta):
		assert len(theta) == 2, "linear_function: theta should be length of 2"
		pred = [None] * len(lst) #predictions
		for i, el in enumerate(lst):
			pred[i] = theta[0] + theta[1] * el
		return pred

	@staticmethod
	def MSE(lst1, lst2): 
	#mean square error - atually not "mean", but the idea is clear
		assert len(lst1) == len(lst2), "sub_piecewise_squares_sum: Lists \
are not equal length"
		s = 0
		for i, j in zip(lst1, lst2):
			s += (i - j) ** 2
		return s

	@staticmethod
	def average_of_list(lst):
		s = 0
		for el in lst:
			s += el
		return s / len(lst)

	@staticmethod
	def sub_two_lists(lst1, lst2):
		assert len(lst1) == len(lst2), "sub_two_lists: Lists are not equal length"
		sub = [None] * len(lst1)
		for i in range(len(lst1)):
			sub[i] = lst1[i] - lst2[i]
		return sub
	
	@staticmethod
	def total_sum_of_squares(lst, av): 
		s = 0
		for el in lst:
			s += (el - av) ** 2
		return s

	@staticmethod
	def average_cov(lst1, lst2):
		assert len(lst1) == len(lst2), "average_cov: Lists are not equal length"
		s = 0
		for i, j in zip(lst1, lst2):
			s += i * j
		return s / len(lst1)
	
	@staticmethod
	def r2(y_pred, y):
		assert len(y_pred) == len(y), f"{bcolors.WARNING}r2{bcolors.ENDC}:\
 Lists are not equal length"
		y_av = LinRegression.average_of_list(y)
		r2 = 1 - LinRegression.MSE(y_pred, y) / \
			LinRegression.total_sum_of_squares(y, y_av)
		return r2
	
	def LSE_method(self):
		"""
		least squares method
		"""
		class LS:
			def __init__(self, x_tr):
				self.x_tr = x_tr
				self.x2_tr_av = self.average_of_squares(x_tr)

			@staticmethod
			def average_of_squares(lst):
				s = 0
				for el in lst:
					s += el ** 2
				return s / len(lst)

		a = LS(self.X_train)
		self.theta = [None] * 2
		self.X_train_av = self.average_of_list(self.X_train)
		self.Y_train_av = self.average_of_list(self.Y_train)
		self.XY_train_av = self.average_cov(self.X_train, self.Y_train)

		self.theta[1] = (self.XY_train_av - self.X_train_av * self.Y_train_av) \
			/ (a.x2_tr_av - self.X_train_av ** 2)
		self.theta[0] = self.Y_train_av - self.theta[1] * self.X_train_av
		self.caclulate_r2()

	@staticmethod
	def get_scaled_input(x_tr, y_tr):
		x_max = max(x_tr)
		y_max = max(y_tr)
		X_train_scaled = [x / x_max for x in x_tr]
		Y_train_scaled = [y / y_max for y in y_tr]
		return X_train_scaled, Y_train_scaled, x_max, y_max

	def caclulate_r2(self):
		self.y_train_pred = self.linear_function(self.X_train, self.theta)
		self.r2_train = self.r2(self.y_train_pred, self.Y_train)
		self.y_test_pred = self.linear_function(self.X_test, self.theta)
		self.r2_test = self.r2(self.y_test_pred, self.Y_test)

	def GD_method(self):
		"""
		linear regression by gradient descent
		"""
		X_train_scaled, Y_train_scaled, x_max, y_max = \
			self.get_scaled_input(self.X_train, self.Y_train)
		self.theta = [random(), random()]
		if self.plot_losses:
			self.losses_train = [None] * self.n_epochs
	
		for i in range(self.n_epochs):
			y_pred = self.linear_function(X_train_scaled, self.theta)
			error = self.sub_two_lists(Y_train_scaled, y_pred)
			error_av = self.average_of_list(error)
			loss_tr = self.MSE(y_pred, Y_train_scaled) / len(X_train_scaled)
			self.losses_train[i] = loss_tr
			theta0_grad = -2 * error_av
			theta1_grad = -2 * self.average_cov(error, X_train_scaled)
			self.theta[0] -= self.lr * theta0_grad
			self.theta[1] -= self.lr * theta1_grad
		self.theta[0] = self.theta[0] * y_max
		self.theta[1] = self.theta[1] * y_max / x_max
		self.caclulate_r2()
		if self.plot_graphes:
			self.plot_graph()
		if self.plot_losses:
			plt.plot(range(self.n_epochs), self.losses_train, linewidth=1, color='b')
			plt.title("Losses as MSE")
			plt.xlabel("epochs")
			plt.ylabel("Losses on train set")
			plt.show()


			

	def plot_graph(self):
		plt.scatter(self.X_train, self.Y_train, color='b', label="Train")
		plt.scatter(self.X_test, self.Y_test, color='r', label="Test")
		plt.plot(self.X_train, self.y_train_pred, color='g', marker='o',markerfacecolor='k')
		plt.title("Price of cars vs. mileage.")
		plt.xlabel("Mileage, km")
		plt.ylabel("Price, arb.units")
		plt.xticks(rotation=90)
		plt.legend()
		plt.show()



