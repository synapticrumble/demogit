def Pythag(a,b):
    """ Pythagorean theorem which takes command arguments"""

    return (a*a + b*b ) ** .5

import sys
if len(sys.argv) >= 3:
    a = float(sys.argv[1])
    b = float(sys.argv[2])
else:
    a = b = 1

print(Pythag(a,b))

