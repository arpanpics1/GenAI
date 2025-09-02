Q1. There are n couples sitting in 2n seats arranged in a row and want to hold hands.

The people and seats are represented by an integer array row where row[i] is the ID of the person sitting in the ith seat. The couples are numbered in order, the first couple being (0, 1), the second couple being (2, 3), and so on with the last couple being (2n - 2, 2n - 1).

Return the minimum number of swaps so that every couple is sitting side by side. A swap consists of choosing any two people, then they stand up and switch seats.

 

Example 1:

Input: row = [0,2,1,3]
Output: 1
Explanation: We only need to swap the second (row[1]) and third (row[2]) person.
Example 2:

Input: row = [3,2,0,1]
Output: 0
Explanation: All couples are already seated side by side.


Q2. I'll create a Python program to validate whether a given bracket sequence is valid or not, using a stack-based approach to handle complex sequences with parentheses (), curly braces {}, and square brackets []. The program will check the provided examples from the previous response and determine their validity. It will also include comments to explain the logic clearly.
Example 1: '{[()()]()[]{}}' -> Valid
Example 2: '{[()](}{)}' -> Invalid
Example 3: '{{[(){[]}]()[()]}}' -> Valid
Example 4: '([{}])[)' -> Invalid
Example 5: '{[(){}]}}(])' -> Invalid


Q3. Given a string s, return whether s is a valid number.

For example, all the following are valid numbers: "2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789", while the following are not valid numbers: "abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53".
