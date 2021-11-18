from config import Parameters, get_config, bcolors
from sys import exit, argv
from re import search, compile
from os.path import isfile

if __name__ == "__main__":
	assert len(argv) == 2,  f"Please pass {bcolors.WARNING}one float{bcolors.ENDC} argument."
	pattern_float = compile("[+-]?\d+.\d+")
	is_arg_good = search(pattern_float, argv[1])
	if is_arg_good is None or is_arg_good.group(0) != argv[1]:
		print(f"Please be sure the argument is {bcolors.WARNING}float{bcolors.ENDC}")
		exit()
	prms = get_config()
	path = "thetas.txt"
	if isfile(path) != True:
		print(f"Please run {bcolors.FAIL}train.py{bcolors.ENDC} first")
		exit()
	f = open(path, "r")
	theta = f.readlines()
	assert len(theta) == 2, f"{bcolors.WARNING}thetas.txt{bcolors.ENDC} \
should contain only two lines"
	theta[0] = theta[0].strip()
	theta[1] = theta[1].strip()
	for i, l in enumerate(theta):
		is_line_good = search(pattern_float, l)
		if is_line_good is None or is_line_good.group(0) != l:
			print(f"This line is bad: {bcolors.WARNING} {l} {bcolors.ENDC}")
		theta[i] = float(theta[i])
	argv[1] = float(argv[1])
	assert argv[1] > 0, f"Please be sure that {bcolors.WARNING}positive\
{bcolors.ENDC} value was passed"
	assert argv[1] < -theta[0]/theta[1], "Please provide less mileage."
	print(f"{round(theta[0] + theta[1] * argv[1], 2)} units of price")
