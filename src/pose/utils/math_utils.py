import math
from numba import njit

def dist(first_point: list, second_point: list) -> float:
    '''Already exists, but it's to simplify other classes that implements math_utils'''
    return math.dist(first_point, second_point)

def avg(list_of_values: list) -> float:
    '''Average of all values in the list'''
    if list_of_values: # check if the length is above 0 and if it exists
        return sum(list_of_values) / len(list_of_values)
    return 0

def keep(current_val: float, min_val: float, max_val: float) -> float:
    '''
    Keeps the value between two bounds. Cannot exceed nor be lower of the boundaries
    if a < min = min
    if a > max = max
    else: a
    '''
    calc = [min_val, max_val][current_val > min_val]
    return [calc, current_val][min_val <= current_val <= max_val]

def max(first: float, second: float) -> float:
    return [first, second][first < second]

@njit
def round(value: float, buffer: int) -> float:
    '''Returns a value with specific numbers after the decimal'''
    if buffer == 0:
        return value
    keep = 10 ** buffer
    return math.floor(value * keep) / keep