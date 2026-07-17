# string s is given, and we have to find out number of palindromic substring.
def number_of_palindromes(s):
    count = 0   # count for storing number of palindromic substrings.
    for i in range(len(s)):   # time comp: O(n**2)
        for j in range(i, len(s)):
            # check whether the substring is palindromic or not.
            if s[i:j + 1] == s[i:j + 1][::-1]:   # is yes then increase the count.
                count += 1

    return count   # return the count.
