# String s is given. you need to find the longest substring without repeating character.
def longest_substring(s):
    max_len=0   # max_len is for maximum_length.

    left=0   #apply sliding window approach.
    mp={}  # map for map the frequency of the characters.
    for i in range(len(s)):
        mp[s[i]]=mp.get(s[i],0)+1  # count the frequency of the character.
        while s[i] in mp and mp[s[i]]>1:  # while there is repeating character move left to the right or shrink the window.
            mp[s[left]]-=1
            left+=1
        max_len=max(max_len,i-left+1)   # calculate the max_len for each and every iteration.
    return max_len  # return max_len.

