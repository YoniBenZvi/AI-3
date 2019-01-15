

def distance_euclidean(list1, list2):
    sum = 0
    for x, y in zip(list1, list2):
        sum = sum + (x-y)**2
    return sum**(.5)
