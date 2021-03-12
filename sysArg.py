#!/usr/bin/env python3


import sys

# It's easy to print the list of all the arguments.
print (sys.argv)

for i in range(len(sys.argv)):
    if i == 0:
        print("function name : %s " % sys.argv[0])

    else:
        print("%d. argument: %s" % (i, sys.argv[i]))


if __name__ == '__main__':
    pass


