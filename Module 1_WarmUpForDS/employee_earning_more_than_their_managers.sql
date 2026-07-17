-- Employee Earning more than their managers

-- select those employees whose salary is greater than their managers.salary by joining Employee table with itself.
select e1.name as Employee
from Employee e1
join Employee e2 on e1.managerId=e2.id
where e1.salary>e2.salary;