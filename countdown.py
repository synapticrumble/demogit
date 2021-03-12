#!/usr/bin/env python3


import sys
import time
def countdown(minutes):
    timer = minutes*60
    for i in range(timer):

        print(str(timer)+' secs left !')
        timer -= 1
        i += 1
        time.sleep(1)
    return 0

if __name__ == '__main__':

    minutes =int(sys.argv[1])
    countdown(minutes)