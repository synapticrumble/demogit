import os
def get_first_line(filename, mode='r'):
    with open(filename, 'r') as file:
        if os.path.isfile(filename):
        #with open(filename, 'r') as file:
            return file.readline()
        else:
            return None

# Running the function below:

a = get_first_line('people-example.csv')
print(a)




