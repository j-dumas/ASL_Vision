import math

def is_higher_than(i, j):
    return i > j

def is_lower_than(i, j):
    return i < j

def distance_between(x, y, targetX, targetY):
    distanceX = targetX - x
    distanceY = targetY - y

    distance = math.sqrt(pow(distanceX, 2) + pow(distanceY, 2)) 
    return distance

def positive(x):
    if (x < 0):
        x = x * -1
    return x