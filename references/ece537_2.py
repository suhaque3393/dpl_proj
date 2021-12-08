# NumPy has become the de facto standard package for general scientific programming in Python. Its core object is the ndarray , a multidimensional array of a single data type, which can be sorted, reshaped, subject to mathematical operations and statistical analysis, written to and read from files, and much more.

import numpy as np

a = np.array([1, 2, 3])
print(a)

print(type(a))

print(a.shape) 

print(a[0], a[1], a[2])

a[0] = 5 
print(a)

b = np.array([[1,2,3],[4,5,6]])
print(b)

print(b.shape)

print(b[0, 0], b[0, 1], b[1, 0])

# Numpy also provides many functions to create arrays: 

a = np.zeros((2,2))
print(a)

b = np.ones((1,2))    
print(b) 

c = np.full((2,2), 7)
print(c) 

d = np.eye(3)         
print(d)  

e = np.random.random((2,2))  
print(e)

# [NumPy link for Array Creation](https://numpy.org/doc/stable/user/basics.creation.html#arrays-creation)

# Numpy offers several ways to index into arrays.

# Slicing: Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

b = a[:2, 1:3]
print(b)

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])
b[0, 0] = 77
print(a[0, 1]) 

# You can also mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the original array. Note that this is quite different from the way that MATLAB handles array slicing:

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
row_r1 = a[1, :]
row_r2 = a[1:2, :]
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  
print(col_r2, col_r2.shape) 

# Integer array indexing: When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array. Here is an example:

a = np.array([[1,2], [3, 4], [5, 6]])
print(a)

print(a[[0, 1, 2], [0, 1, 0]]) 

print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

print(a[[0, 0], [1, 1]]) 

print(np.array([a[0, 1], a[0, 1]]))

# One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:



a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)

b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])

a[np.arange(4), b] += 10
print(a)

# Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition. Here is an example:



a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = (a > 2)
print(bool_idx)

print(a[bool_idx])

print(a[a > 2]) 

# [NumPy Link for indexinglink](https://numpy.org/doc/stable/reference/arrays.indexing.html)

# Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype. Here is an example:

x = np.array([1, 2])   
print(x.dtype)  

x = np.array([1.0, 2.0])   
print(x.dtype)             


x = np.array([1, 2], dtype=np.int32)   
print(x.dtype)   

# [NumPy Link for Data types](https://numpy.org/doc/stable/reference/arrays.dtypes.html)

# Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module:

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))

# Note that unlike MATLAB, * is elementwise multiplication, not matrix multiplication. We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w))
print(np.dot(v, w))

print(x.dot(v))
print(np.dot(x, v))

print(v.dot(x))
print(np.dot(v, x))

print(x.dot(y))
print(np.dot(x, y))

print(y.dot(x))
print(np.dot(y, x))

# Numpy provides many useful functions for performing computations on arrays; one of the most useful is sum:

x = np.array([[1,2],[3,4]])
print(x)
print(np.sum(x)) 

print(x)
print(np.sum(x, axis=0))

print(x)
print(np.sum(x, axis=1)) 

# [NumPy link for mathematical functions ](https://numpy.org/doc/stable/reference/routines.math.html)

# Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays. The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:

x = np.array([[1,2], [3,4]])
print(x)    
print(x.T)

v = np.array([1,2,3])
print(v)   
print(v.T)

# [NumPy link for array manupulation](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)

# Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.

# For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:



x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(x)
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

for i in range(4):
    y[i, :] = x[i, :] + v

print(y)

# This works; however when the matrix x is very large, computing an explicit loop in Python could be slow. Note that adding the vector v to each row of the matrix x is equivalent to forming a matrix vv by stacking multiple copies of v vertically, then performing elementwise summation of x and vv. We could implement this approach like this:

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv) 

y = x + vv  
print(y)

# Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. Consider this version, using broadcasting:

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  
print(y)

# Broadcasting two arrays together follows these rules:

# 1. If the arrays do not have the same rank, prepend the shape of the lower rank 
# array with 1s until both shapes have the same length.
# 2. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
# 3. The arrays can be broadcast together if they are compatible in all dimensions.
# 4. After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
# 5. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension

# [NumPy link for Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

# Matplotlib is a plotting library. In this section give a brief introduction to the matplotlib.pyplot module, which provides a plotting system similar to that of MATLAB.

The most important function in matplotlib is plot, which allows you to plot 2D data. Here is a simple example:

import matplotlib.pyplot as plt
# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

import tensorflow as tf
mnist_train,mnist_test = tf.keras.datasets.mnist.load_data()


first_image = mnist_train[0][0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

# [Link for matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html)

