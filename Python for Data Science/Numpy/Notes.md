# NUMPY

NumPy stands for numeric python which is a python package for the computation and processing of the multidimensional and single dimensional array elements.
It is an extension module of Python which is mostly written in C.

`` Why is Numpy Array so Fast?  (Numpy VS List)``

- Homogeneous Data: NumPy arrays store elements of the same data type, making them more compact and memory-efficient than lists.

- Fixed Data Type: NumPy arrays have a fixed data type, reducing memory overhead by eliminating the need to store type information for each element.

- Contiguous Memory: NumPy arrays store elements in adjacent memory locations, reducing fragmentation and allowing for efficient access.

## NumPy Arrays

We can create a NumPy ndarray object by using the array() function.

```# 1D array
arr_1d = np.array([element1, element2, element3, ...])
```

-  In NumPy, you can use square brackets [] to create multi-dimensional arrays directly. 

### Array Properties:

1. Shape (shape):

- Returns a tuple representing the dimensions of the array.

```
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Shape:", arr.shape)  # Output: (2, 3)
```
2. Number of Dimensions (ndim):

- Returns the number of dimensions (axes) of the array.

```
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Number of Dimensions:", arr.ndim)  # Output: 2
```

3. Size (size):

- Returns the total number of elements in the array.

```
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Size:", arr.size)  # Output: 6
```

4. Data Type (dtype):

- Returns the data type of the elements in the array.

```
import numpy as np

arr = np.array([1, 2, 3], dtype=np.float64)
print("Item Size:", arr.itemsize)  # Output: 8 (for float64)
```

5. Item Size (itemsize):

- Returns the size (in bytes) of each element in the array.
```
import numpy as np

arr = np.array([1, 2, 3], dtype=np.float64)
print("Item Size:", arr.itemsize)  # Output: 8 (for float64)
```

6. Byte Size (nbytes):

- Returns the total number of bytes consumed by the array's elements.

```
import numpy as np

arr = np.array([1, 2, 3], dtype=np.float64)
print("Byte Size:", arr.nbytes)  # Output: 24 (3 elements * 8 bytes for float64)
```

### Array Indexing and Slicing

``Array Indexing``

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# Accessing elements
print(arr[0])  # Output: 1
print(arr[-1])  # Output: 5
print(arr_2d[0, 1])  # Output: 2 (row 0, column 1)
print(arr_2d[2, 2])  # Output: 9 (row 2, column 2)
```

``Array Slicing``

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])


# Slicing
print(arr[1:4])  # Output: [2, 3, 4]
print(arr_2d[0:2, 1:3])
# Output:
# [[2, 3],
#  [5, 6]]
print(arr_2d[::2, ::2])
# Output:
# [[1, 3],
#  [7, 9]]
```

###  Array Manipulation

1. reshape()

```arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape to a 2D array
arr_reshaped = arr.reshape((2, 3))
print(arr_reshaped)
# Output:
# [[1, 2, 3],
#  [4, 5, 6]]
```
2. append()

```
arr = np.array([1, 2, 3])

# Append elements
arr_appended = np.append(arr, [4, 5, 6])
print(arr_appended)
# Output: [1, 2, 3, 4, 5, 6]
```

3.. delete()

```
arr = np.array([1, 2, 3, 4, 5])

# Delete elements at indices 1 and 3
arr_deleted = np.delete(arr, [1, 3])
print(arr_deleted)
# Output: [1, 3, 5]
```

4. concatenate()

```arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate along an existing axis
arr_concatenated = np.concatenate([arr1, arr2])
print(arr_concatenated)
# Output: [1, 2, 3, 4, 5, 6]

```

5. split()

```
arr = np.array([1, 2, 3, 4, 5, 6])

# Split the array at indices 2 and 4
arr_split = np.split(arr, [2, 4])
print(arr_split)
# Output: [array([1, 2]), array([3, 4]), array([5, 6])]
```

6. transpose()

```
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Transpose the array
arr_transposed = arr.transpose()
print(arr_transposed)
# Output:
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

7. stack()

```
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Stack arrays vertically (along a new axis)
arr_stacked = np.stack([arr1, arr2])
print(arr_stacked)
# Output:
# [[1, 2, 3],
#  [4, 5, 6]]
```

### Statistical and Mathematical Operations

1. Basic Statistics

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Mean of the array
mean_value = np.mean(arr,axis=0)
print(mean_value)
# Output: 3.0

median_value = np.median(arr,axis=0)
print(median_value)
# Output: 3.0

# Standard deviation of the array
std_deviation = np.std(arr,axis=0)
print(std_deviation)
# Output: 1.4142135623730951

# compute the 25th percentile of the array
result1 = np.percentile(arr, 25)
print("25th percentile:",result1)

# compute the 75th percentile of the array
result2 = np.percentile(arr, 75)
print("75th percentile:",result2)

# find the minimum value of the array
min_val = np.min(arr)

# find the maximum value of the array
max_val = np.max(arr)

sum_along = np.sum(arr,axis=0)
print(sum_along)
```

NOTE:
```
In NumPy, many functions support the axis parameter, which determines the axis or axes along which the operation is performed. The axis is specified as an integer or a tuple of integers.

For a 1D array, the only valid axis is axis=0.
For a 2D array, axis=0 refers to operations along columns, and axis=1 refers to operations along rows.
For a higher-dimensional array, the axis parameter can be used to specify the dimension along which the operation should be applied.
```

### NumPy Matrix Operations

![Alt text](<Screenshot 2023-12-24 152336.png>)


```
import numpy as np

# Create a matrix using array()
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Display the original matrix
print("Original Matrix:")
print(matrix)

# Perform matrix multiplication using dot()
matrix_mult = np.dot(matrix, matrix)
print("\nMatrix Multiplication:")
print(matrix_mult)

# Transpose the matrix using transpose()
matrix_transposed = np.transpose(matrix)
print("\nTransposed Matrix:")
print(matrix_transposed)

# Calculate the inverse of the matrix using linalg.inv()
matrix_inverse = np.linalg.inv(matrix)
print("\nInverse Matrix:")
print(matrix_inverse)

# Calculate the determinant of the matrix using linalg.det()
matrix_determinant = np.linalg.det(matrix)
print("\nDeterminant of the Matrix:", matrix_determinant)

# Flatten the matrix into a 1D array using flatten()
matrix_flattened = matrix.flatten()
print("\nFlattened Matrix:")
print(matrix_flattened)
```

### Trigonometric and exponential functions in NumPy

```
import numpy as np
import matplotlib.pyplot as plt

# Generate an array of angles from 0 to 2*pi
angles = np.linspace(0, 2*np.pi, 100)

# Trigonometric Functions
sin_values = np.sin(angles)
cos_values = np.cos(angles)
tan_values = np.tan(angles)

# Exponential Functions
exp_values = np.exp(angles)
log_values = np.log(angles + 1)  # Adding 1 to avoid log(0)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(angles, sin_values, label='sin(x)')
plt.legend()
plt.title('Sine Function')

plt.subplot(2, 2, 2)
plt.plot(angles, cos_values, label='cos(x)')
plt.legend()
plt.title('Cosine Function')

plt.subplot(2, 2, 3)
plt.plot(angles, tan_values, label='tan(x)')
plt.legend()
plt.title('Tangent Function')

plt.subplot(2, 2, 4)
plt.plot(angles, exp_values, label='exp(x)')
plt.plot(angles, log_values, label='log(x+1)')
plt.legend()
plt.title('Exponential and Logarithmic Functions')

plt.tight_layout()
plt.show()

```

### Random module in Numpy

```
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate a random integer between 0 and 9
random_integer = np.random.randint(0, 10)
print("Random Integer:", random_integer)

# Generate a random floating-point number between 0 and 1
random_float = np.random.rand()
print("Random Float:", random_float)

# Generate a 1D array of 5 random integers between 0 and 9
random_integer_array = np.random.randint(0, 10, 5)
print("1D Random Integer Array:", random_integer_array)

# Generate a 1D array of 5 random numbers between 0 and 1
random_float_array = np.random.rand(5)
print("1D Random Float Array:", random_float_array)

# Generate a 2D array of shape (3, 4) with random integers between 0 and 9
random_2d_array = np.random.randint(0, 10, (3, 4))
print("2D Random Integer Array:")
print(random_2d_array)

# Choose a random number from an array
array_to_choose_from = np.array([10, 20, 30, 40, 50])
random_choice = np.random.choice(array_to_choose_from)
print("Random Choice from Array:", random_choice)

```
