from cython cimport cdivision, boundscheck
from libc.stdlib cimport malloc, free


@cdivision
@boundscheck(False)
cdef class DamerauLevenshtein:
    cdef long** array
    cdef long insertion_cost
    cdef long deletion_cost
    cdef long substitution_cost
    cdef long transposition_cost
    def __init__(
        self,
        max_left_length: long,
        max_right_length: long,
        transposition_cost: long = 1,
        substitution_cost: long = 1,
        insertion_cost: long = 1,
        deletion_cost: long = 1,
    ):
        cdef long i, j
        cdef long rows = max_left_length + 2
        cdef long cols = max_right_length + 2
        self.insertion_cost = insertion_cost
        self.deletion_cost = deletion_cost
        self.substitution_cost = substitution_cost
        self.transposition_cost = transposition_cost
        cdef long** array = <long**> malloc(rows * sizeof(long*))
        for i in range(0, rows):
            array[i] = <long*> malloc(cols * sizeof(long))
            for j in range(0, cols):
                array[i][j] = 0
        array[0][0] = 0
        for i from 1 <= i < rows by 1:
            array[i][0] = i
        for i from 1 <= i < cols by 1:
            array[0][i] = i
        for i in range(1, rows):
            for j in range(1, cols):
                array[i][j] = 0

        self.array = array

    def distance(self, left: str, right: str):
        cdef bytes left_bytes = left.encode()
        cdef bytes right_bytes = right.encode()
        cdef char* left_c_string = left_bytes
        cdef char* right_c_string = right_bytes
        cdef long left_length = len(left)
        cdef long right_length = len(right)

        return self._distance(left_c_string, right_c_string, left_length, right_length)

    def similarity(self, left: str, right: str):
        cdef bytes left_bytes = left.encode()
        cdef bytes right_bytes = right.encode()
        cdef char* left_c_string = left_bytes
        cdef char* right_c_string = right_bytes
        cdef long left_length = len(left)
        cdef long right_length = len(right)

        return self._similarity(left_c_string, right_c_string, left_length, right_length)

    cdef long _distance(self, char* left, char* right, long left_length, long right_length):
        array = self.array
        cdef char left_char
        cdef char right_char
        cdef int unchanged
        cdef long insertion
        cdef long deletion
        cdef long substitution
        cdef long minimum
        cdef long insertion_cost = self.insertion_cost
        cdef long deletion_cost = self.deletion_cost
        cdef long substitution_cost = self.substitution_cost
        cdef long transposition_cost = self.transposition_cost
        for i from 0 <= i < left_length:
            for j from 0 <= j < right_length:
                left_char = left[i]
                right_char = right[j]
                unchanged = left_char == right_char
                array[i + 1][j + 1] = min(
                    array[i + 1][j] + insertion_cost,
                    array[i][j + 1] + deletion_cost,
                    array[i][j] + (0 if unchanged else substitution_cost),
                )
                minimum = min(insertion, deletion, substitution)
                if i and j and left_char == right[j - 1] and left[i - 1] == right_char:
                    if not unchanged: cost = transposition_cost
                    array[i + 1][j + 1] = min(array[i + 1][j + 1], array[i - 1][j - 1] + cost)

        return array[left_length][right_length]

    cdef double _similarity(self, char* left, char* right, long left_length, long right_length):
        cdef double distance = self._distance(left, right, left_length, right_length)
        cdef double max_length = max(left_length, right_length)

        return 1.0 - distance / max_length

    def close(self):
        array = self.array
        free(array)
