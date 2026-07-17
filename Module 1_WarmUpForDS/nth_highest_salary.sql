-- Find the nth Highest Salary

-- create a function to get the nth highest salary.

CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
    -- write the sql query to find the salary of the nth employee.so select the offset of n-1 peoples and select the n-1th salary.
      SELECT DISTINCT Salary
      FROM Employee
      ORDER BY Salary DESC
      LIMIT 1 OFFSET N - 1
  );
END;













