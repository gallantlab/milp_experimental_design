#  Design of Complex Neuroscience Experiments using Mixed Integer Linear Programming
This repository provides example implementations for the four case studies in the article ["Design of Complex Experiments using Mixed Integer Linear Programming"](https://arxiv.org/abs/2012.02361). Each case study aims to demonstrate how Mixed Integer Linear Programming (MILP) can be used to address real-world experimental design challenges. Full details of each case study can be found in the main article. The article also contains an introduction to the mathematical foundations of MILP.

## Code Organization

Each case study is solved by creating mixed integer linear programs. The included `milp` python package is used to create and solve these programs. This package can be installed using the included `setup.py` file. A jupyter notebook in the `notebooks` directory gives a guided walkthrough of each case study and reproduces the figures from the main article.

## `milp` Workflow

Each mixed integer linear program is constructed using the same basic workflow:

#### 1. Initialize Program
`milp.program.initialize_program()` initializes a dictionary that represents a mixed integer linear program. This dictionary will be updated to include the program's variables, linear constraints, and cost function terms.

```python
import milp
program = milp.program.initialize_program()```

#### 2. Add Variables

`milp.program.add_variable()` adds a variable to a program. The name given to each variable will be used to specify its constraints and cost function coefficients. Whether a varaible is real, integer, or boolean can be specified by setting `variable_type` to `float`, `int`, or `bool`.

```python
milp.program.add_variable(
    program=program,
    name='a',
    variable_type=bool,
)
milp.program.add_variable(
    program=program,
    name='b',
    variable_type=int,
    lower_bound=float('-inf'),
    upper_bound=float('inf'),
)
milp.program.add_variable(
    program=program,
    name='c',
    variable_type=float,
)
```

#### 3.  Add Linear Constraints

`milp.program.add_constraint()` adds a linear constraint to program. An equality constraint can be specified by using arguments `A_eq` and `b_eq`. The value of `A_eq` should be a dictionary whose keys are variable names and whose values are coefficients of those variables. Inequality constraints can be specified by using either `A_lt` and `b_lt`, or `A_gt` and `b_gt`. 

```python
milp.program.add_constraint(
    program=program,
    A_eq={'a': 1, 'b': 1},
    b_eq=0,
)
milp.program.add_constraint(
    program=program,
    A_lt={'a': 1, 'b': 2, 'c': 3},
    b_lt=3,
)
```

#### 4. Specify Cost Function
`milp.program.add_cost_terms` is used to specify the program's cost function. The value of `coefficients` should be a dictionary whose keys are variable names and whose values are coefficients of those variables.

```python
milp.program.add_cost_terms(
    program=program,
    coefficients={'a': -1, 'b': 1},
)
```

#### 5. Solve Program
`milp.program.solve_program()` solves the program using an external library (by default uses [Gurobi](https://www.gurobi.com)). It does this by 1) converting the program into the library-compatible representation, 2) running its solver, and 3) returning the solution. This solution specifies an experimental design that optimally conforms to the design constraints of the program.

```python
solution = milp.program.solve_MILP(program=program)
print(solution['variables'])
```
```{'a': True, 'b': -1, 'c': 0.0}```

Taken together these code snippets have represented and solved the following simple program:
 
<img src="https://render.githubusercontent.com/render/math?math=\min{b - a}">
 
<img src="https://render.githubusercontent.com/render/math?math=a %2B b = 0">
 
<img src="https://render.githubusercontent.com/render/math?math=a %2B 2 b %2B 3 c \leq 3">

<img src="https://render.githubusercontent.com/render/math?math=a \in \mathbb{B}, b \in \mathbb{Z}, c \in \mathbb{R}^{%2B}">

Some additional functions are used to implement the design patterns discussed in **Section 3.2** of the main article. For further details refer to the source code and docstrings of `milp`.


## Software Licensing
- All code in this repository, including the code in the `milp` package and the code in this Jupyter notebook, is licensed under a BSD 2-clause license.
- By default the `milp` package solves programs using the external library [Gurobi](https://www.gurobi.com). Gurobi offers free academic licenses, available [here](https://www.gurobi.com/downloads/end-user-license-agreement-academic/). Instructions for installing Gurobi can be found on the Gurobi website.
- Alternative solvers can be used instead of Gurobi. The `milp` also contains an adapter for [CPLEX](https://www.ibm.com/analytics/cplex-optimizer), used by specifying `solve_MILP(..., solver='cplex')`.
