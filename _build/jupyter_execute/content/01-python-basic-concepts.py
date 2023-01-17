#!/usr/bin/env python
# coding: utf-8

# # Python: Basic Concepts

# ## 1. Variables and objects

# ### 1.1. Simple objects

#  Each object has a unique data type. Use the `type()` function to check the class of the object.

# In[1]:


type(100)


# In[2]:


type('python')


# The equal sign `=` is used to assign the value of an object to a variable. The variable is created at the first time an object is assigned to it.

# In[3]:


a = 100
a


# In[4]:


b = 'python'
print(b)


# Variables do not need to be declared with any particular type. The value of a variable can be overwritten later.

# In[5]:


a = 5
a = 9
a


# There are some other ways to assign value to a variable as well. Python executes the right side of the equal sign `=` first, then assigns it to the left side.

# In[6]:


a = 5
b = a
print(a)
print(b)


# In[7]:


a = 7
a = a + 2
a


# ### 1.2. Object identity
# The `id()` function displays the address of an object.

# In[8]:


pi = 3.14
id(pi)


# To check whether if two variables have the same address, try `is` or `is not` identity statement. If two variables point to the same small string or small integer, they will share the same address. Small strings (strings without spaces and have less than 20 characters) and small integers (integers from -5 to +255) are frequently used objects, so reusing them may save memory.

# In[9]:


x = 'Hello'
y = 'Hello'
x is y


# In[10]:


x = 'Hello World'
y = 'Hello World'
x is y


# In[11]:


x = 10e5
y = 10e5
y is x


# ## 2. Operators

# ### 2.1. Math operators

# In[12]:


# addition
123 + 532


# In[13]:


# subtraction
555 - 143


# In[14]:


# multiplication
124 * 936


# In[15]:


# division
48 / 12


# In[16]:


# remainder of a division
16 % 5 == 1


# In[17]:


# integer division or floor division
16 // 5 == 3


# In[18]:


# exponentiation
2 ** 5 == 32


# ### 2.2. Assignment operators
# Assignment operators assign the value to a variable.

# In[19]:


x = 7


# Mathematical operators can combine with the equal sign `=` to form a new assignment operator.

# In[20]:


x = 7
x += 3
x


# In[21]:


x = 10
x -= 8
x


# In[22]:


x = 5
x *= 4
x


# In[23]:


x = 60
x /= 5
x


# Multiple values can be assigned to multiple variables at once.

# In[24]:


x, y, z = 3, 8, 1
print(x, y, z)


# ### 2.3. Comparison operators
# Comparison operators check if the two values are equal or not; returns `True` or `False`.

# In[25]:


7 == 7


# In[26]:


'a' != 'A'


# In[27]:


7 > 4


# In[28]:


2 < 5


# In[29]:


10 >= 10


# In[30]:


4 <= 0


# Python also supports multiple comparison operators in a single statement. It only returns `True` when all operators are `True`. For example, these two statements are equivalent:

# In[31]:


1 < 5 < 10 != 11


# In[32]:


(1 < 5) and (5 < 10) and (10 != 11)


# ```{note}
# Comparison operators are very similar to identity operators, but they check the values instead of addresses.
# ```

# In[1]:


0.8**10


# ## 3. Other concepts

# ### 3.1. Functions

# In[33]:


print('Hello world')


# `print()` is a Python function. It takes the argument `'Hello world'` as input.

# In[34]:


get_ipython().run_line_magic('pinfo', 'print')


# The `?` command shows a function's docstrings (documentation string). `value`, `sep` and `end` are the parameters of the `print()` function. Parameters work as the names of arguments.

# In[35]:


print('Anaconda', 'Python', 'Jupyter', sep='\n')


# ### 3.2. Pipe
# The idea of `sspipe` is inspired by the *pipe operator* `%>%` from `magrittr`, a library of R. It has the ability to transform a complicated expression with nested parentheses to a sequence of simple expressions, which improves the human readability.
# 
# The whole functionality of this library is exposed by two objects, `p` (as a *wrapper* for functions to be called on the piped object) and `px` (as a *placeholder* for piped object). By default, the function will take the piped object as its first argument.

# In[1]:


from sspipe import p, px


# In[2]:


# using Python
print('Hello', end='!\n')

# using sspipe, default
'Hello' | p(print, end='!\n')

# using sspipe, positional argument
'!\n' | p(print, 'Hello', end=px)


# In[3]:


from math import sqrt, sin, ceil, pi

# using Python
print(int(sqrt(abs(ceil(pi)))))

# using sspipe
pi | p(ceil) | p(abs) | p(sqrt) | p(int) | p(print)


# ### 3.3. Special punctuations
# This section introduces three punctuations having special functionality in Python
# - The hash `#` indicates the rest of the line is commented content and will not be executed. Comments can be used to explain thinking process and mark some code lines for late execution.
# - The backslash `\` works as a continuation character, it tells Python that the line should continue.
# - The semicolon `;` allows writing multiple statements on a single line.

# In[39]:


print("Hello, world!")
# do not execute this line


# In[40]:


print(1); print(2)


# In[41]:


print('Hello, \
world!')


# ### 3.4. Magic commands

# In[42]:


# show all magic commands
get_ipython().run_line_magic('lsmagic', '')


# In[43]:


# notebook inline chart
get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


# time of execution
get_ipython().run_line_magic('timeit', 'L = [n**2 for n in range(1000)]')


# In[45]:


# all created variable names
get_ipython().run_line_magic('whos', '')


# ## 4. Libraries
# A library (also known as library or module) is a `.py` suffix file containing a set of functions and objects used for specific purposes.

# ### 4.1. Accessing a library

# The examples below give access to the `pi` constant and the `degrees()` function of the `math` module, then convert the angle $\pi$ from radian to degree.

# In[46]:


# basic import
import math
math.degrees(math.pi)


# In[47]:


# import the math module under the alias m
import math as m
m.degrees(m.pi)


# In[48]:


# import specific functions and constants
from math import pi, degrees
degrees(pi)


# In[49]:


# give alias to functions and constants
from math import pi as pi_number, degrees as radian_to_degree
radian_to_degree(pi_number)


# ### 4.2. Libraries help
# 
# ```python
# # list of functions and constants
# import math
# dir(math)
# 
# # help
# import math
# help(math)
# ```

# ### 4.3. Libraries management
# This section introduces `pip`, the packages manager for Python.
# 
# ```python
# # install a library
# pip install pygame
# 
# # the alternative command if the "module object is not callable" error raises
# python -m pip install pygame --user
# 
# # install a specific version
# pip install seaborn==0.10.0
# 
# # upgrade
# pip install --upgrade seaborn
# 
# # install all necessary libraries listed in a file
# pip install -r utilities/requirements.txt
# 
# # reinstall the latest version 
# pip install --upgrade --force-reinstall pyspark
# 
# # reinstall a specific version
# pip install --force-reinstall pyspark==2.4.7
# 
# # display the version of all libraries
# pip freeze
# 
# # active R to be used in Jupyter
# conda install -c r r-irkernel
# ```

# ## 5. Control flow statements
# In Python, indentation is a requirement, not just to make the code look pretty.

# ### 5.1. The if statement
# If the condition is `True`, then the body of `if` (recognized by indentation) gets executed. If the condition is `False`, then the body of `else` gets executed instead.

# In[51]:


x = 10
if x > 0:
    print('x is positive')
else:
    print('x is at most 0')


# Nested `if... else...` statements allow adding more branches.

# In[52]:


x = 10
if x > 0:
    print('x is positive')
else:
    if x == 0:
        print('x equals 0')
    else:
        print('x is negative')


# Or use `elif` instead.

# In[53]:


x = 10
if x > 0:
    print('x is positive')
elif x == 0:
    print('x equals 0')
else:
    print('x is negative')


# Python also supports short-hand `if... else...` statement.

# In[54]:


x = 10
print('x is positive') if x > 0 else print('x is at most 0')


# ### 5.2. The for loop
# A `for` loop runs through every item of an iterable and executes its body part with that value.

# In[55]:


for i in [1, 2, 3, None, 5]:
    print(i)


# The `break` statement terminates the current loop.

# In[56]:


for i in [1, 2, 3, 4, 5]:
    if i == 4:
        break
    print(i)


# The `continue` statement skips the remaining of the current iteration and continues to the next one.

# In[57]:


total = 0
for i in [1, 2, 3, None, 5]:
    if i == None:
        continue
    total += i
total


# ### 5.3. The while loop
# A `while` loop statement repeatedly executes its body as long as the condition is `True`.

# In[2]:


x = 0
while x < 10:
    print(x, end=' ')
    x += 1


# Add one more constraint to the loop condition.

# In[3]:


x = 0
while x < 10 and x != 5:
    print(x, end=' ')
    x += 1


# The `break` and `continue` statements also work with the `while` loop.

# In[4]:


x = 0
while True:
    x += 1
    print(x, end=' ')
    if x >= 10:
        break


# In[6]:


x = 0
while x < 10:
    x += 1
    if x == 5:
        continue
    print(x, end=' ')


# ### 5.4. Errors handling

# #### Type of errors
# - Syntax errors: caused by improper syntax of Python
# - Logical errors (or exceptions): occur at runtime

# In[62]:


if 1 > 0:
    print(True)


# In[63]:


1/0


# In[64]:


print(no_name)


# #### Exceptions handling
# While syntax errors are mostly done by beginners, experienced Data Scientist would always want to handle exceptions in practice. A code block which can raise exception is placed inside the `try` clause. The code that handles the exception is written in the `except` clause.

# In[65]:


try:
    print(6/0)
except:
    print('Cannot execute')


# Use `sys.exc_info()` to get the type of exception. This function only works with `try` and `except`.

# In[66]:


import sys
for entry in ['a', 10, 0, 2]:
    print('Entry:', entry)
    try:
        print('The reciprocal:', 1/entry)
    except:
        print(sys.exc_info()[0])
    print('')


# Each type of error can have its own solution.

# In[67]:


for entry in ['a', '5', 0, 2.5, 10]:
    try:
        print('Entry:', entry)
        print('The reciprocal:', 1/entry)
    except TypeError:
        try:
            print('The reciprocal:', 1/float(entry))
        except ValueError:
            print(ValueError)
    except ZeroDivisionError:
        print('The reciprocal: infinity')
    print('')

