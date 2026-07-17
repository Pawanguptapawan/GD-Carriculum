from collections import defaultdict
# Given an array of strings strs, group the anagrams together. You can return the answer in any order.
def groupAna(strs):
    mp=defaultdict(list)    #map for storing anagram corresponding to their sorted_word.
    for s in strs:    # check for each and every word and store them to their corresponding key.
        ss=''.join(sorted(s))
        mp[ss].append(s)

    result=[]   #  resultant array.
    for _,val in mp.items():  # store all values corresponding to their keys.
        result.append(val)
    return result  # return result.
