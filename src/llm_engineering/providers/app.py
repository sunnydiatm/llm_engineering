import math
import random
from pstats import SortKey


# def reverse_a_number(a):
#     print("Revering the input number")
#     for x in reversed(a):
#         print(x)
#
#
# number = input("Enter a number to reverse: ")
# reverse_a_number(number)

# def is_a_prime_number(n):
#     if n <= 1:
#         print("Not a prime number")
#     for x in range (2, int(math.sqrt(n) + 1)):
#         if n % x == 0:
#             print(str(x) + " is not a prime number")
#             break
#     print(str(n) + " is a PRIME number")
#
#
# prime_number = input("Enter a number to check if it's a prime number: ")
# is_a_prime_number(int(prime_number))

# def print_fib_series(n):
#     a, b = 0, 1
#     series = []
#     while len(series) < n:
#        series.append(a)
#        a, b = b, a + b
#     print(series)
#
#
# fibo_number = input("Enter a number until which you want fibonacci number: ")
# print_fib_series(int(fibo_number))


# def count_digits(n):
#     input_digit = str(n)
#     print(len(input_digit))
#
#
# digit_number = input("Enter a number for which you want to count digits: ")
# count_digits(int(digit_number))

# def smallest_larget_in_list(n):
#    print("starting smallest largest")
#    sorted_list = sorted(n)
#    print(n)
#    n.sort()
#
#    print("sorted list")
#    print(n)
#    print("smallest number is:", n[0])
#    print("Largest number is:", n[-1])
#
#
#
# input_list = input("Enter a comma separated list of numbers: ")
# list_converted = input_list.split(",")
# smallest_larget_in_list(list_converted)


# guess_limit = 3
# guess_secret = 9
# guess_count = 0
#
# while guess_count < guess_limit:
#    guess_number = int(input("Guess the number:"))
#    guess_count += 1
#    if guess_number == guess_secret:
#       print("You guessed right!")
#       break
# else:
#    print("Sorry, That was a wrong guess!")

# def reverseTheString(palindrom_number):
#    input_char = ""
#    number_len = len(str(palindrom_number))
#    count = 0
#    for x in range(len(str(palindrom_number))):
#       # print(number_len)
#       # print(count)
#       input_char = input_char + palindrom_number[number_len - count - 1]
#       count = count + 1
#
#    print("reverse String ->" + input_char)
#    return input_char
#
# palindrom_number= input("Enter a number to see if it is a palindrome:")
#
# output_char = reverseTheString(palindrom_number)
# if palindrom_number.lower() == output_char.lower():
#    print("It's a Palindrome!")
# else:
#    print("Not a Palindrome!")

# price_list = [100,2,3,11,109,119]
# total_price = 0
# for price in price_list:
#    total_price += price
#    #print(total_price)
# print(total_price)

# numbers = [2, 2, 2, 2 ,5]
#
# # for number in numbers:
# #    for x in range(number):
# #       print("x" * number)
# #       break;
#
# for number in numbers:
#    count = ''
#    for x in range(number):
#       count = count + 'x'
#    print(count)

# number_list = [2, 2 ,5,12,9,44,3,5,8]
# max = number_list[0]
# for number in number_list:
#    if number > max:
#       max = number
# print(max)

# number_list = [2, 2 ,5, 12, 9, 44, 3, 5, 8]
#
# max = 42
# min = 3
# for number in number_list:
#     if number > max:
#         max = number
#     elif number < min:
#         min = number
# print("Max is:" + str(max))
# print("Min is:" + str(min))

# string = "123456"
# count = 0
# reversed_string = ""
# for char in string:
#    reversed_string = reversed_string + string[len(string) - 1 - count]
#    count += 1
#
# print(reversed_string)

# number_list = [2, 2 ,5, 12, 9, 44, 44, 3, 5, 80, 80]
#
# largest = None
# second_largest = None
# for number in number_list:
#     if largest is None or number > largest:
#         if largest != number:
#             second_largest = largest
#         largest = number
#     elif number != largest and (second_largest is None or number > second_largest):
#         second_largest = number
#
#
# print("Largest is:" + str(largest))
# print("Second Largest is:" + str(second_largest))


# tuple = [11,22,1]
# x,y,z = tuple
# print(x)
# x = 50
# print(x)



# class Point:
#     def move(self):
#         print("move")
#
#     def move2(self):
#         print("move2")
#
#
# point = Point()
# point.x = 10
# point.y = 20
# point.z = 30
# point.move()
# point.move2()
# print(point.x)
# print(point.y)
# print(point.z)



# class Person:
#
#     def talk(self):
#         print(f'Person {self.name} is talking')
#
#     def __init__(self, name):
#         self.name = name
#
#
# try:
#     sunny = Person("Sunny")
#     sunny.age = 30
#     print(sunny.age)
#     sunny.talk()
#     prachi = Person("Prachi")
#     prachi.talk()
#     print(prachi.age)
# except Exception as e:
#     print("Something went wrong: {}", e)



class Dice:

    def roll(self):
        number_1 = random.randint(1, 6)
        number_2 = random.randint(1, 6)
        list = [number_1, number_2]
        return tuple(list)


dice = Dice()
roll = dice.roll()
print(roll)
