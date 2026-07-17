#You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
# Find two lines that together with the x-axis form a container, such that the container contains the most water.
# Return the maximum amount of water a container can store.
# Notice that you may not slant the container.
def container_with_most_water(nums):
    maxArea=0    # maxArea is a variable for storing the maximum value.
    left=0
    right=len(nums)-1    # using two pointer approach from left to right. TE=O(n).
    while left<right:
        # area= distance between two pillars and minimum height
        maxArea=max(maxArea,(right-left)*min(nums[left],nums[right]))
        if nums[left]>nums[right]:    # if height of right pillar is less than the left pillar then slide the right pointer towards left.
            right-=1
        else:
            left+=1   # otherwise slide the left pointer towards right.
    return maxArea  # return maximum area.
