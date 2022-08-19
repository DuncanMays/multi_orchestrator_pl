import sys
from types import SimpleNamespace

arg_list = sys.argv[1:]

arg_dict = {}
for index, arg in enumerate(arg_list):
	if (arg[0] == '-'):
		arg_dict[arg[1:]] = arg_list[index+1]

args = SimpleNamespace(**arg_dict)

print(args)