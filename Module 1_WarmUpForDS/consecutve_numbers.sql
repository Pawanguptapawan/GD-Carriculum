-- consecutive numbers
-- Find all numbers that appear at least three times consecutively.
-- Return the result table in any order.
 
-- means a number should be same for at least threee consecutive ids.


select distinct l1.num as Consecutive
from Logs l1
join Logs l2 on l1.id=l2.id+1
join Logs l3 on l2.id=l3.id+1
where l1.num=l2.num and l2.num=l3.num;