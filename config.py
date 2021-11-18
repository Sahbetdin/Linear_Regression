from dataclasses import dataclass
# import dataclasses
# import os


@dataclass
class Parameters:
	use_method = "gd" #gd/ls  #CHOOSE: gradient descent or least square method
	file: str = "data.csv"
	is_header: bool = True
	test_ratio: float = 0.2
	learning_rate: float = 0.03
	n_epochs: int = 4000
	print_results: bool = True #prints thetas and r2 in-out of sample
	plot_losses: bool = True #just for "gd"
	plot_graphes: bool = True #plot dataset and predicted points

def get_config():
	return Parameters()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'