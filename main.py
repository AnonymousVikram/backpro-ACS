import numpy as np
import random

dataSet = np.array([])


def insertData():
    int1 = random.randint(0, 1)
    int2 = random.randint(0, 1)
    return int1, int2, (int1 or int2)

print(insertData())