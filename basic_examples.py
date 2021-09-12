"""
Some code snippets are taken from the tutorial of the Autumn 2019 course CS-C3160 Data Science
"""

##################################################################################
# PYTHON BASICS: See also PFDA CHAPTER 3                                         #
##################################################################################

# Strings

str1 = 'Hello'
str2 = 'Python'

# Access to characters
print(str1[0])  # H
print(str1[-1])  # o
print(str2[1:5])  # ytho

# String concatenation
greetings = str1 + ' ' + str2
greetings += '!'
print(greetings)

# Bring to lowercase
print(greetings.lower())

# Number of characters in a string
print(len(greetings))

# Split by space characters
greetings = greetings.split()
print(greetings)

###############################

# Lists

# Create a list
test_list = []  # []

# Appending elements
test_list.append(1)  # [1]
test_list.append('two')  # [1, 'two']

# Extending the list
test_list.extend([8, 6, 23])  # [1, 'two', 8, 6, 23]
print(test_list)

# Length of the list
print(len(test_list))

# Indexing
print(test_list[0])
print(test_list[-2])
print(test_list[2:4])

# Matrix (i.e. multidimensional lists)

test_matrix = [[1, 2, 3], [2, 4, 6]]
print(test_matrix[0][2])
print(test_matrix[0][:])

# List comprehension

squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x ** 2)
print(squares)

# VS.

squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(squares)

###############################################

# Tuples

test_tuple = ('Aalto', 'University', 'Data Science Project')

# Length of tuple
print(len(test_tuple))

# Indexing
print(test_tuple[0])

#####################################################


# Empty dictionary
test_dic = {}

# Non-empty dictionary
test_dic = {'Ellison': 5, 'Mark': 7, 'Josh': 2}

# Indexing
print(test_dic['Mark'])  # 7

# Iterating
for key, value in test_dic.items():
    print(f'For key {key} the value is {value}')

######################################################

# Conditions and loops

# IF
weight = 50
if weight > 100:
    print('That\'s too heavy!')
elif 50 >= weight > 10:
    print('That\'s reasonable!')
else:
    print('Are you sure you didn\'t forget your luggage?')

# FOR loop example:
for x in range(3):
    print(x)

# WHILE loop example:
n = 0
while n < 10:
    # Example of complex indentation
    if n % 2 == 0:
        n += 1
    n += 1
    print(n)


#######################################################

# Functions

# Define a function
def plus(a, b):
    return a + b


# Call a function
c = plus(5, 11)
print(c)

#######################################################
