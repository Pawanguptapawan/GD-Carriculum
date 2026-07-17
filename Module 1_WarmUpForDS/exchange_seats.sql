

-- Exchange Seats

-- Write a solution to swap the seat id of every two consecutive students. If the number of students is odd, the id of the last student is not swapped.
-- Return the result table ordered by id in ascending order.

select
    case
        when id%2=1 and id+1<=(select count(*) from seat)
                then id+1
        when id%2=0
                then id-1
        else id
    End as id,
    student
from seat
order by id;
 
