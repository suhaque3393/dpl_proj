import sys
print(sys.version)

# Like most languages, Python has a number of basic types including integers, floats, booleans, and strings. These data types behave in ways that are familiar from other programming languages.

x = 1000
print(type(x))

y = 1000.0
print(type(y))

a= (7==8)
print(a)
print(type(a))

dept='ECE'
course="537"
print(dept)      
print(len(dept))
my_course=dept+course
print(my_course)
unit=3
f"This course {my_course} has {unit} units."

s = "deep learning"
print(s.capitalize())  
print(s.upper())       
print(s.replace('deep', 'machine'))  

# Python includes several built-in container types: lists, dictionaries, sets, and tuples.

# A list is the Python equivalent of an array, but is resizeable and can contain elements of different types.

xs = [3, 2.5, 1e5, 4+4j]    
print(xs, xs[2])  
print(xs[-1])     
xs[2] = 'foo'     
print(xs)         
xs.append('bar')  
print(xs)         
x = xs.pop()      
print(x, xs)      

# Slicing: In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing.



nums = list(range(10))     
print(nums)               
print(nums[2:8])          
print(nums[2:])           
print(nums[:2])           
print(nums[:])            
print(nums[:-1])          
nums[2:4] = [8, 9]        
print(nums)   

# Loops: You can loop over the elements of a list like this:

courses = ['ECE240', 'ECE351', 'ECE537']
for course in courses:
    print(course)

# If you want access to the index of each element within the body of a loop, use the built-in enumerate function:

for idx, course in enumerate(courses):
    print(f'#{idx}: {course}')

# A list comprehension in Python is a construct for creating a list based on another iterable object in a single line of code.

nums = [0, 1, 2, 3, 4, 5]
cubes = [x ** 3 for x in nums]
print(cubes)

# List comprehensions can also contain conditions:

even_cubes = [x ** 3 for x in nums if x % 2 == 0]
print(even_cubes)

# A dictionary in Python is a type of “associative array” (also known as a “hash” in some languages). A dictionary can contain any objects as its values, but unlike sequences such as lists and tuples, in which the items are indexed by an integer starting at 0, each item in a dictionary is indexed by a unique key, which may be any immutable object. The dictionary therefore exists as a collection of key-value pairs; dictionaries themselves are
# mutable objects.

height = {'Burj Khalifa ': 828. , 'One World Trade Center ': 541.3 ,
 'Q1': 323. , 'Carlton Centre ': 223. , 'Gran Torre Santiago ': 300. ,
'Mercury City Tower ': 339.}
height

print(height ['One World Trade Center '])

height ['Empire State Building '] = 381.
height ['The Shard '] = 306.
height

An alternative way of defining a dictionary is to pass a sequence of (key, value) pairs to the dict constructor. If the keys are simple strings (of the sort that could be used as variable names), the pairs can also be specified as keyword arguments to this constructor:

ordinal = dict ([(1 , 'First '), (2, 'Second '), (3, 'Third ')])
mass = dict( Mercury =3.301e23 , Venus =4.867e24 , Earth =5.972e24)

ordinal [2] # NB 2 here is a key , not an index

mass['Earth']

# A for -loop iteration over a dictionary returns the dictionary keys (in order of key insertion):

for c in ordinal :
  print(c, ordinal [c])

# If you want access to keys and their corresponding values, use the items method:

d = {'crab': 10, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print(f'A {animal} has {legs} legs')

# Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries. 

nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)

# A set is an unordered collection of unique items. As with dictionary keys, elements of a set must be hashable objects. A set is useful for removing duplicates from a sequence and for determining the union, intersection and difference between two collections. Because they are unordered, set objects cannot be indexed or sliced, but they can be iterated over, tested for membership and they support the len built-in. A set is created by listing its elements between braces ( {...} ) or by passing an iterable to the set()
# constructor:

s = set ([1, 1, 4, 3, 2, 2, 3, 4, 1, 3, 'surprise !'])
s

len(s)

2 in s, 6 not in s

s.add(7)
s

s.remove('surprise !')
s

Loops: Iterating over a set has the same syntax as iterating over a list; however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:

for item in s:
  print (item)

C, D = set ((3 , 4, 5, 6)) , set ((6,7 , 8, 9))
print(C | D)
print(C & D)

# Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:

from math import sqrt
nums = {int(sqrt(x)) for x in range(100)}
print(nums)  

# A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:

d = {(x, x + 1): x for x in range(10)}  
t = (5, 6)        
print(type(t))    
print(d[t])       
print(d[(1, 2)])  

# Python functions are defined using the def keyword. For example:

def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

# We will often define functions to take optional keyword arguments, like this:

def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)


hello('Bob') 
hello('Fred', loud=True) 

# A class is defined using the class keyword and indenting the body of statements
# (attributes and methods) in a block following this declaration. It is a good idea to follow the class statement with a docstring describing what it is that the class does. Class methods are defined using the familiar def keyword, but the first argument to each method should be a variable named self – this name is used to refer to the object itself when it wants to call its own methods or refer to attributes

class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  
g.greet()            
g.greet(loud=True)