<<<<<<< HEAD
print("hello from python on ubuntu on windows!")
=======

# program to keep asking for valid file name to open

# code will go on forever loop until the valid file is provided
import os
def readFile(filename, mode='r'):
    ''' read all the non-empty lines until end of file, and print it. 
        The code must meet the following requirements:
    -> Each line of the file must be read and printed
    -> If a blank line is encountered, it must be ignored
    -> When all lines have been read, the file must be closed
    You create the following code. Line numbers are included for reference only.
    '''
    inventory = open(filename, 'r')
    eof = False
    while eof == False:
        line = inventory.readline()
        if line !='':
            if line != "\n":
                print(line)
        else:
            print("end of file")
            eof = True
            inventory.close()


# Running the function below:

a = readFile('people-example.csv')
print(a)



'''
my_list = [4,7,[2,4,6,8]]

c = 0


for i in range(3):
    if type(my_list[i]) == list:
        for j in my_list[i]:
            if j%2 == 0:
                c +=1
            else:
                c -=1
print(c)




while True:
    try:
        fname = input('enter file: ')
        if not fname:
            break
        f = open(fname)
        print(f.read())
        f.close()
        break
    except FileNotFoundError:
        print('file not found, re-enter. ')
        

try:
    f = open('stuff.txt', 'r')
    print(f.read())
except:
    print("file not found error")
else:
    print('operation success')
finally:
    f.close()
    '''
    
>>>>>>> 0bdd2f2ede771d41fffad454f8d5ebafcbea182f
