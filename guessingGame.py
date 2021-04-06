import random
n = random.randint(1,100)
while True:
    
    try:
        guess = int(input("enter guess? "))
        if guess > n:
            print('too high, guess again! ')
        elif guess < n:
            print('too low, guess again! ')
        else:
            print('Success!')
            break

    except ValueError:
        print("incorrect value entered, please enter numeric value")
