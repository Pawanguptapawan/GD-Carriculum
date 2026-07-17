
-- Customer who never order
-- means select those customers from Customers table whose id is not present in Orders table's customerId column.
select name as Customers
from Customers
where id not in (
select customerId
from Orders
);
