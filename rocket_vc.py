import pandas
import numpy
import matplotlib
import pdb



def dot(sparse1, sparse2):
    # length = max(max(sparse1[0]), max(sparse2[0]))
    # print(length)
    results = [[], []]
    summ = 0
    for ind in sparse1[0]:
        if ind in sparse2[0]:
            # results[0].append(ind)
            # results[1].append(sparse1[1][ind] * sparse2[1][ind])
            summ += sparse1[1][ind] * sparse2[1][ind]
            print(summ)
    return results


if __name__ == "__main__":
    sparse1 = [[0,1,2], [9, 7, 5]]
    sparse2 = [[0,1,3], [1, 2, 3]]
    dot(sparse1, sparse2)