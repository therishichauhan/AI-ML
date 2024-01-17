## SQL Queries

### Data Definition Language (DDL) Statements

**CREATE Statement**

- Used to create database objects like tables, views, and indexes.

- Syntax: CREATE TABLE table_name (column1 datatype, column2 datatype, ...);

**ALTER Statement**

- Used to modify existing database objects.

- Syntax: ALTER TABLE table_name ADD column_name datatype;

**DROP Statement**

- Used to delete database objects.

- Syntax: DROP TABLE table_name;
Data Manipulation Language (DML) Statements

**INSERT Statement**

- Used to insert new records into a table.

- Syntax: INSERT INTO table_name (column1, column2, ...) VALUES (value1, value2, ...);

**UPDATE Statement**

- Used to modify existing records in a table.

- Syntax: UPDATE table_name SET column1 = value1, column2 = value2 WHERE condition;

**DELETE Statement**

- Used to delete records from a table.

- Syntax: DELETE FROM table_name WHERE condition;

**Simple Queries with WHERE Clause**

WHERE Clause

- Used to filter records based on a specified condition.

- Example: SELECT * FROM employees WHERE department = 'HR';

**Compound WHERE Clause with Multiple AND & OR Conditions**

AND Condition

- Used to combine multiple conditions where all must be true.

- Example: SELECT * FROM products WHERE category = 'Electronics' AND price < 500;

OR Condition

- Used to combine multiple conditions where at least one must be true.

- Example: SELECT * FROM orders WHERE status = 'Shipped' OR status = 'Processing';

**SQL | Join**

SQL Join statement is used to combine data or rows from two or more tables based on a common field between them.

**INNER JOIN**

The INNER JOIN keyword selects all rows from both the tables as long as the condition is satisfied. This keyword will create the result-set by combining all rows from both the tables where the condition satisfies i.e value of the common field will be the same. 

```Syntax: 

SELECT table1.column1,table1.column2,table2.column1,....
FROM table1 
INNER JOIN table2
ON table1.matching_column = table2.matching_column;
```

Note: 

```We can also write JOIN instead of INNER JOIN. JOIN is same as INNER JOIN. ```


**LEFT JOIN**

This join returns all the rows of the table on the left side of the join and matches rows for the table on the right side of the join. For the rows for which there is no matching row on the right side, the result-set will contain null. LEFT JOIN is also known as LEFT OUTER JOIN.

Syntax: 
```
SELECT table1.column1,table1.column2,table2.column1,....
FROM table1 
LEFT JOIN table2
ON table1.matching_column = table2.matching_column;
```

Note:
``` We can also use LEFT OUTER JOIN instead of LEFT JOIN, both are the same.```


**RIGHT JOIN**

RIGHT JOIN is similar to LEFT JOIN. This join returns all the rows of the table on the right side of the join and matching rows for the table on the left side of the join. For the rows for which there is no matching row on the left side, the result-set will contain null. RIGHT JOIN is also known as RIGHT OUTER JOIN. 

Syntax: 
```
SELECT table1.column1,table1.column2,table2.column1,....
FROM table1 
RIGHT JOIN table2
ON table1.matching_column = table2.matching_column;
```
Note:
``` We can also use RIGHT OUTER JOIN instead of RIGHT JOIN, both are the same.```.


**Sub-queries**


Sub-queries

- A query nested within another query.

Example:
```
SELECT product_name
FROM products
WHERE category_id IN (SELECT category_id FROM categories WHERE category_name = 'Electronics'); 
```

**Correlated Sub-queries**

- A sub-query that depends on values from the outer query.

Example:
```
SELECT employee_name
FROM employees e
WHERE salary > (SELECT AVG(salary) FROM employees WHERE department = e.department);
```

**Data Control Language (DCL) Statements**

**GRANT Statement**

- Used to grant permissions to users or roles.

- Syntax: GRANT privilege ON object TO user_or_role;

**REVOKE Statement**

- Used to revoke permissions previously granted.

- Syntax: REVOKE privilege ON object FROM user_or_role;

