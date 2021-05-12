'''
DISTANCE STUDY OF BIKES

DAYS

    DISTANCE COVERED IN KMS

        ENFIELD     HONDA   YAHAMA  KAWASAKI

DAY 1	50	80	70	90
DAY 2	40	20	20	50
DAY 3	70	20	60	15
DAY 4	80	50	40	70
DAY 5	20	60	60	45

'''

import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y = [50,40,70,80,20]
y2 = [80,20,20,50,60]
y3 = [70,20,60,40,60]
y4 = [90,50,15,70,45]
plt.plot(x,y,'g',label='Enfield', linewidth=5)
plt.plot(x,y2,'c',label='Honda',linewidth=5)
plt.plot(x,y3,'k',label='Yahama',linewidth=5)
plt.plot(x,y4,'y',label='KTM',linewidth=5)
plt.title('bike details in line plot')
plt.ylabel(' Distance in kms')
plt.xlabel('Days')
plt.legend()
plt.show()