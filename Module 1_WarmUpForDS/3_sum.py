# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
# Notice that the solution set must not contain duplicate triplets.
def sum_3(nums,target):
    result=[]   # result for storing the ans arrays
    nums.sort()  # sort the array such that we can easily check for target
    # time comp: O( nlog(n) )
    for i in range(len(nums)-2):
        left=i+1        # here we apply two pointer concept for optimization.
        right=len(nums)-1    # left pointer and right pointer.
        if i>0 and nums[i]==nums[i-1]:   # check for duplicates.
            continue
        while left<right:     # left should always less than right.
            s=nums[i]+nums[left]+nums[right]
            if s==target:      # check whether the calculated res is equal to target or not.
                result.append([nums[i],nums[left],nums[right]])  # if yes then store it in result array.
                while left<right and nums[left]==nums[left+1]:   # again  check for duplicates.
                    left+=1
                while left<right and nums[right]==nums[right-1]:   # again check for duplicates
                    right-=1

                left+=1
                right-=1     # slide the pointers.
            elif s<target:
                left+=1     # if calculated res is less than target then slide left pointer to the right
            else:
                right-=1    # else slide right pointer towards left.
    return result   # return the result array.








