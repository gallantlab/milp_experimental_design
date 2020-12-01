"""simple utilities for constructing and solving mixed integer programs"""

import time

import numpy as np


def initialize_program():
    """return a dict representation of a MILP program"""
    return {
        'variables': {},
        'constants': {},
        'constraints': {
            'A_eq': [],
            'b_eq': [],
            'A_lt': [],
            'b_lt': [],
        },
        'cost_function': {},
    }


def add_variable(
    program, name, variable_type, lower_bound=None, upper_bound=None,
):
    """add variable to program

    # Inputs
    - program: dict of MIP program
    - name: str of variable name
    - variable_type: should be one of {bool, int, float}
    - lower_bound: number lower bound of variable
    - upper_bound: number upper bound of variable
    """
    if name in program['variables']:
        raise Exception('variable already exists:' + str(name))
    else:
        variable = {
            'type': variable_type,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
        }
        program['variables'][name] = variable


def add_constraint(program, A_eq=None, b_eq=None, A_lt=None, b_lt=None):
    """add a constraint to program (in place)

    - use (A_eq, b_eq) to add an equality constraint
    - use (A_lt, b_lt) to add an equality constraint
    - A_eq / A_lt are the lhs of the equation
    - b_eq / b_lt are the rhs of the equation
    - A_X should be in the form of {variable_name: coefficient_value}
    """
    if A_eq is not None and b_eq is not None and A_lt is None and b_lt is None:
        program['constraints']['A_eq'].append(A_eq)
        program['constraints']['b_eq'].append(b_eq)
    elif (
        A_lt is not None and b_lt is not None and A_eq is None and b_eq is None
    ):
        program['constraints']['A_lt'].append(A_lt)
        program['constraints']['b_lt'].append(b_lt)
    else:
        raise Exception('specify (A_eq and b_eq) or (A_lt and b_lt)')


def add_cost_terms(program, coefficients):
    """add cost terms to program"""
    for variable_name, coefficient in coefficients.items():
        program['cost_function'].setdefault(variable_name, 0)
        program['cost_function'][variable_name] += coefficient


def add_abs_cost_term(program, coefficients, constant=0):
    """add a term to the cost function (in place)

    - term will look like abs(variables.dot(coefficients) + constant)
    - coefficients + constant <= name
        --> coefficients - name <= -constant
    - -coefficients + -constant <= name
        --> -coefficients - name <= constant
    """

    # create new variable for absolute value term
    ordered = [
        key + '__' + str(coefficients[key])
        for key in sorted(coefficients.keys())
    ]
    name = 'abs__' + '__'.join(ordered)
    name = str(np.abs(hash(name)))
    add_variable(program, name, float)

    # variable >= linear_expression
    coefficients_1 = {name: -1}
    coefficients_1.update(coefficients)
    add_constraint(program, A_lt=coefficients_1, b_lt=-constant)

    # variable >= -linear_expression
    coefficients_2 = {name: -1}
    coefficients_2.update({k: -v for k, v in coefficients.items()})
    add_constraint(program, A_lt=coefficients_2, b_lt=constant)

    # add term to cost function
    add_cost_terms(program, {name: 1})

    return {
        'variable_name': name,
    }


def store_constant(program, name, value):
    """store constant for later reference"""
    if name in program['constants']:
        raise Exception('constant already exists: ' + str(name))
    else:
        program['constants'][name] = value


def solve_MILP(program, solver='gurobi', verbose=True, **kwargs):
    """solve MILP"""

    # print program information
    if verbose:
        n_constraints = (
            len(program['constraints']['A_eq'])
            + len(program['constraints']['A_lt'])
        )
        print('program size:')
        print('- n_variables:', len(program['variables']))
        print('- n_constraints:', n_constraints)
        print('- n_cost_function_terms:', len(program['cost_function']))

    if solver == 'gurobi':
        return solve_MILP_gurobi(program, verbose=verbose, **kwargs)
    elif solver == 'cplex':
        return solve_MILP_cplex(program, verbose=verbose, **kwargs)
    else:
        raise Exception('solver unrecognized: ' + str(solver))


def solve_MILP_cplex(program, verbose=True):
    """solve MILP using cplex"""

    import docplex.mp.model

    start = time.time()

    model = docplex.mp.model.Model(name='model')

    vtypes = {
        bool: model.binary_var,
        int: model.integer_var,
        float: model.continuous_var,
    }

    # add variables
    variables = {}
    for variable_name, variable in program['variables'].items():
        variable_type = variable['type']
        kwargs = {}
        if variable['lower_bound'] is not None:
            kwargs['lb'] = variable['lower_bound']
        if variable['upper_bound'] is not None:
            kwargs['ub'] = variable['upper_bound']
        variables[variable_name] = vtypes[variable_type](
            name=variable_name,
            **kwargs
        )

    # add constraints
    for A, b, operator in [
        [program['constraints']['A_lt'], program['constraints']['b_lt'], '<='],
        [program['constraints']['A_eq'], program['constraints']['b_eq'], '='],
    ]:
        for coefficients, constant in zip(A, b):
            expression = model.sum(
                coefficient * variables[variable_name]
                for variable_name, coefficient in coefficients.items()
            )

            if operator == '<=':
                model.add_constraint(expression <= constant)
            elif operator == '=':
                model.add_constraint(expression == constant)
            else:
                raise Exception(operator)

    # add objective function
    expression = model.sum(
        coefficient * variables[variable_name]
        for variable_name, coefficient in program['cost_function'].items()
    )
    model.minimize(expression)

    # solve
    model.solve()

    return {
        'variables': {
            variable_name: variable.solution_value
            for variable_name, variable in variables.items()
        },
        'objective': model.objective_value,
        'program': program,
        'model': model,
        'time': time.time() - start,
    }


def solve_MILP_gurobi(program, verbose=True, parameters=None):
    """solve MILP using gurobi"""

    import gurobipy

    start = time.time()

    model = gurobipy.Model('model')
    model.setParam('OutputFlag', 0)

    if parameters is not None:
        for parameter, value in parameters.items():
            model.setParam(parameter, value)

    vtypes = {
        bool: gurobipy.GRB.BINARY,
        int: gurobipy.GRB.INTEGER,
        float: gurobipy.GRB.CONTINUOUS,
    }

    # add variables
    variables = {}
    for variable_name, variable in program['variables'].items():
        variable_type = variable['type']
        kwargs = {}
        if variable['lower_bound'] is not None:
            kwargs['lb'] = variable['lower_bound']
        if variable['upper_bound'] is not None:
            kwargs['ub'] = variable['upper_bound']
        variable = model.addVar(
            vtype=vtypes[variable_type],
            name=variable_name,
            **kwargs
        )
        variables[variable_name] = variable
    model.update()

    # add constraints
    for A, b, operator in [
        [program['constraints']['A_lt'], program['constraints']['b_lt'], '<='],
        [program['constraints']['A_eq'], program['constraints']['b_eq'], '='],
    ]:
        for coefficients, constant in zip(A, b):
            expression = gurobipy.LinExpr()
            for variable_name, coefficient in coefficients.items():
                expression += coefficient * variables[variable_name]
            model.addConstr(expression, operator, constant)

    # add objective function
    expression = gurobipy.LinExpr()
    for variable_name, coefficient in program['cost_function'].items():
        expression += coefficient * variables[variable_name]
    model.setObjective(expression)

    # solve
    model.optimize()
    if model.Status != gurobipy.GRB.OPTIMAL:
        if verbose:
            feasible = str(model.Status != gurobipy.GRB.Status.INFEASIBLE)
            print('feasible: ' + feasible)
            print('optimal: ' + str(model.Status == gurobipy.GRB.OPTIMAL))
        raise Exception('solution could not be found')

    variables = {
        variable_name: program['variables'][variable_name]['type'](variable.X)
        for variable_name, variable in variables.items()
    }
    return {
        'variables': variables,
        'objective': model.objVal,
        'program': program,
        'model': model,
        'time': time.time() - start,
    }


def get_variables(program, name, *parameters):
    """get all variables matching an indexed variable"""

    variables = []
    for variable in program['variables'].keys():
        if variable.startswith(name + '_'):
            variable_parameters = variable.split('_')[-1]
            variable_parameters = [
                int(index) for index in variable_parameters.split(',')
            ]

            for parameter, variable_parameter in zip(
                parameters,
                variable_parameters,
            ):
                if parameter != variable_parameter and parameter is not None:
                    break
            else:
                variables.append(variable)

    return variables


def get_solution_variable(solution, name):
    """get variable within a solution"""
    return get_named_quantity(solution['variables'], name)


def get_solution_constant(solution, name):
    """get constant within a solution"""
    return get_named_quantity(solution['program']['constants'], name)


def get_named_quantity(some_dict, name):
    """get named quantity within a solution"""

    # scalar variable
    if name in some_dict:
        return some_dict[name]

    # tensor variable
    else:

        entries = {}

        for variable, value in some_dict.items():
            if variable.startswith(name + '_'):
                multiindex = variable.split(name + '_', 1)[1]
                multiindex = tuple(
                    int(number) for number in multiindex.split(',')
                )
                entries[multiindex] = value

        if len(entries) == 0:
            raise Exception('variable not found')
        else:
            example_index, example_entry = next(iter(entries.items()))

        n_dims = len(example_index)
        max_indices = [0] * n_dims

        for index in entries.keys():
            for d in range(n_dims):
                if index[d] > max_indices[d]:
                    max_indices[d] = index[d]

        output = np.zeros(
            [max_index + 1 for max_index in max_indices],
            dtype=type(example_entry),
        )
        for index, entry in entries.items():
            output[index] = entry

        return output


def initialize_license():
    """run null program to get license message out of the way"""
    program = initialize_program()
    solve_MILP(program, verbose=False)

