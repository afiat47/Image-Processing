import cv2
import numpy as np

def find_extremes(matrix):
    smallest_abs_nonzero = None
    largest_value = None

    for row in matrix:
        for value in row:
            if value != 0:
                abs_value = abs(value)
                if smallest_abs_nonzero is None or abs_value < abs(smallest_abs_nonzero):
                    smallest_abs_nonzero = value
                if largest_value is None or value > largest_value:
                    largest_value = value

    return smallest_abs_nonzero, largest_value

def normalizedKernel(matrix):
    smallest_abs_nonzero, largest_value = find_extremes(matrix)

    sm=(smallest_abs_nonzero/smallest_abs_nonzero)//1
    mx=(largest_value/smallest_abs_nonzero)//1

    cv2.normalize(matrix, matrix,sm, mx, cv2.NORM_MINMAX)

    out = np.round(matrix).astype(np.uint8)
    print(out)
    print("done")
    return out