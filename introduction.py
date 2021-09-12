import math


def print_intro():
    """
    This function prints some lines of text.
    :return: None
    """
    print("\n" +
          "According to PYPL (https://pypl.github.io/PYPL.html), Python is the most popular programming language.\n" +
          "This script presents some examples of Python using popular libraries such as numpy, pandas and matplotlib.\n" +
          "Instructions for installing Python and the basic libraries can be found in Section 1.4 of PFDA\n" +
          "Numerous Python tutorials are also available on-line.\n"
          )


def welcome_student(name):
    """
    This function takes as input a student's name as a string, and prints out a personalized welocome message.
    :param name: str, the name of a student
    :return: None
    """
    print(f"Hi, {name}! Welcome to this Python presentation.")


def ask_for_input(input_request):
    """
    This function asks for input. The input is described in the string 'input_request'
    :param input_request: str, a description of the input requested to the user
    :return: user_input, the user's input
    """
    user_input = input(input_request)
    return user_input


def continue_when_ready():
    ask_for_input(">>> Press Enter to continue...")


def explain_variables():
    """
    This function prints some lines of text.
    :return: None
    """
    print("VARIABLES\n" +
          "\tVariable names can contain letters, numbers and the special character '_'.\n" +
          "\tVariable names cannot start with a number, and are CASE SENSITIVE.\n")

    # Examples of variables types
    a_boolean = True
    an_integer = 50
    a_float = 12.34
    a_complex = 2 + 3j
    a_string = "Hello World"
    a_list = [1.5, 3.3, 7.12]
    a_tuple = (1, 0)
    a_dictionary = {'Ellison': 5, 'Mark': 7, 'Josh': 2}

    print(f"\tAn example of boolean: {a_boolean}")
    print(f"\tAn example of integer: {an_integer}")
    print(f"\tAn example of float: {a_float}")
    print(f"\tAn example of complex: {a_complex}")
    print(f"\tAn example of string: '{a_string}'")
    print(f"\tAn example of list: {a_list}")
    print(
        "\t\tLists are ordered collections of values. They are mutable and can contain the same value more than once.")
    print("P.S. Strings are actually lists of characters!")

    print(f"\tAn example of tuple: {a_tuple}")
    print("\t\tTuples are immutable versions of lists.")

    print(f"\tAn example of dictionary: {a_dictionary}")
    print("\t\tDictionaries are unordered collections of key-value pairs. " +
          "Keys are unique, you can't have the same key twice.")

    print("\n")
    print("\tSee Section 3.1 in PFDA book to know more.")
    print("\n")

    print("\tYou can use the command 'type(...)' to check the type of your variable.")
    print("\tThe result of 'type(2 + 3j)' is here below:")
    print(type(a_complex))


def explain_operators():
    """
    This function prints some lines of text.
    :return: None
    """
    print("OPERATORS")
    print("\tYou can add with +, subtract with -, multiply with *, divide with /")
    print("\tRemember that ** does exponent and // does floor division!")
    print("\tMore advanced math operations might require the math module.")

    print("\nEXAMPLES")
    a = 3
    b = 2
    print(f"\t{a} to the power of {b} is {a ** b}")

    number = 28.567
    print(f"\tThe square root of {number} is {math.sqrt(number)}")
    print("\tAnd the value of Pi is ", math.pi)


def final_tips():
    """
    This function prints some lines of text.
    :return: None
    """
    print("\nFinally, remember that you can google for the documentation or examples, " +
          "and use 'help(...)' when you are stuck.")
    print("As an example, see here below the output of help(math.factorial):")
    print(help(math.factorial))


def main():
    """
    This functions calls all the other functions defined above and combines them to build the tutorial.
    Run this function to run the tutorial.
    :return: None
    """
    print_intro()
    name = ask_for_input(">>> What's your name? ")
    welcome_student(name)

    explain_variables()
    continue_when_ready()

    explain_operators()
    continue_when_ready()

    final_tips()
    continue_when_ready()

    ask_for_input(f"\n\nThat's all for now, {name}. Press Enter to finish.")


if __name__ == '__main__':
    main()
