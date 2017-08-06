```@meta
CurrentModule = JuMP
```

Variables
=========

Variables, also known as columns or decision variables, are the results of the optimization.

Constructors
------------

The primary way to create variables is with the `@variable` macro. The first argument will always be a `Model`. In the examples below we assume `m` is already defined. The second argument is an expression that declares the variable name and optionally allows specification of lower and upper bounds. Adding variables "column-wise", e.g., as in column generation, is supported as well; see the syntax discussed in the [Problem Modification](@ref) section.

```julia
@variable(m, x )              # No bounds
@variable(m, x >= lb )        # Lower bound only (note: 'lb <= x' is not valid)
@variable(m, x <= ub )        # Upper bound only
@variable(m, lb <= x <= ub )  # Lower and upper bounds
@variable(m, x == fixedval )  # Fixed to a value (lb == ub)
```

All these variations create a new local variable, in this case `x`. The names of your variables must be valid Julia variable names. Integer and binary restrictions can optionally be specified with a third argument, `Int` or `Bin`. For advanced users, `SemiCont` and `SemiInt` may be used to create [semicontinuous](http://orinanobworld.blogspot.com/2011/03/semicontinuous-variables.html) or [semi-integer](http://www.gams.com/help/topic/gams.doc/userguides/mccarl/semi-integer_variables.htm) variables, respectively.

To create arrays of variables we append brackets to the variable name.

```julia
@variable(m, x[1:M,1:N] >= 0 )
```

will create an `M` by `N` array of variables. Both ranges and arbitrary iterable sets are supported as index sets. Currently we only support ranges of the form `a:b` where `a` is an explicit integer, not a variable. Using ranges will generally be faster than using arbitrary symbols. You can mix both ranges and lists of symbols, as in the following example:

```julia
s = ["Green","Blue"]
@variable(m, x[-10:10,s] , Int)
x[-4,"Green"]
```

Bounds can depend on variable indices:

```julia
@variable(m, x[i=1:10] >= i )
```

And indices can have dependencies on preceding indices (e.g. "triangular indexing"):

```julia
@variable(m, x[i=1:10,j=i:10] >= 0)
```

Note the dependency must be on preceding indices, going from left to right. That is, `@variable(m, x[i=j:10,i=1:10] >= 0)` is not valid JuMP code.

Conditions can be placed on the index values for which variables are created; the condition follows the statement of the index sets and is separated with a semicolon:

```julia
@variable(m, x[i=1:10,j=1:10; isodd(i+j)] >= 0)
```

Note that only one condition can be added, although expressions can be built up by using the usual `&&` and `||` logical operators.

An initial value of each variable may be provided with the `start` keyword to `@variable`:

```julia
@variable(m, x[i=1:10], start=(i/2))
```

Is equivalent to:

```julia
@variable(m, x[i=1:10])
for i in 1:10
    setvalue(x[i], i/2)
end
```

For more complicated variable bounds, it may be clearer to specify them using the `lowerbound` and `upperbound` keyword arguments to `@variable`:

```julia
@variable(m, x[i=1:3], lowerbound=my_complex_function(i))
@variable(m, x[i=1:3], lowerbound=my_complex_function(i), upperbound=another_function(i))
```

Variable categories may be set in a more programmatic way by providing the appropriate symbol to the `category` keyword argument:

```julia
t = [:Bin,:Int]
@variable(m, x[i=1:2], category=t[i])
@variable(m, y, category=:SemiCont)
```

The constructor `Variable(m::Model,idx::Int)` may be used to create a variable object corresponding to an *existing* variable in the model (the constructor does not add a new variable to the model). The variable indices correspond to those of the internal MathProgBase model. The inverse of this operation is `linearindex(x::Variable)`, which returns the flattened out (linear) index of a variable as JuMP provides it to a solver. We guarantee that `Variable(m,linearindex(x))` returns `x` itself. These methods are only useful if you intend to interact with solver properties which are not directly exposed through JuMP.

!!! note
    `@variable` is equivalent to a simple assignment `x = ...` in Julia and therefore redefines variables. The following code will generate a warning and may lead to unexpected results:

    ```julia
    @variable(m, x[1:10,1:10])
    @variable(m, x[1:5])
    ```

    After the second line, the Julia variable `x` refers to a set of variables indexed by the range `1:5`. The reference to the first set of variables has been lost, although they will remain in the model. See also the section on anonymous variables.

Anonymous variables
-------------------

We also provide a syntax for constructing "anonymous" variables. In `@variable`, you may omit the name of the variable and instead assign the return value as you would like:

```julia
x = @variable(m) # Equivalent to @variable(m, x)
x = @variable(m, [i=1:3], lowerbound = i, upperbound = 2i) # Equivalent to @variable(m, i <= x[i=1:3] <= 2i)
```

The `lowerbound` and `upperbound` keywords must be used instead of comparison operators for specifying variable bounds within the anonymous syntax. For creating noncontinuous anonymous variables, the `category` keyword must be used to avoid ambiguity, e.g.:

```julia
x = @variable(m, Bin) # error
x = @variable(m, category = :Bin) # ok
```

Besides these syntax restrictions in the `@variable` macro, the **only** differences between anonymous and named variables are:

> 1.  For the purposes of printing a model, JuMP will not have a name for anonymous variables and will instead use `__anon__`. You may set the name of a variable for printing by using `setname` or the `basename` keyword argument described below.
> 2.  Anonymous variables cannot be retrieved by using `getindex` or `m[name]`.

If you would like to change the name used when printing a variable or group of variables, you may use the `basename` keyword argument:

```julia
i = 3
@variable(m, x[1:3], basename="myvariable-$i")
# OR:
x = @variable(m, [1:3], basename="myvariable-$i")
```

Printing `x[2]` will display `myvariable-3[2]`.

Semidefinite and symmetric variables
------------------------------------

JuMP supports modeling with [semidefinite variables](https://en.wikipedia.org/wiki/Semidefinite_programming). A square symmetric matrix ``X`` is positive semidefinite if all eigenvalues are nonnegative; this is typically denoted by ``X \succeq 0``. You can declare a matrix of variables to be positive semidefinite as follows:

```julia
@variable(m, X[1:3,1:3], SDP)
```

Note in particular the indexing: 1) exactly two index sets must be specified, 2) they must both be unit ranges starting at 1, 3) no bounds can be provided alongside the `SDP` tag. If you wish to impose more complex semidefinite constraints on the variables, e.g. ``X − I \succeq 0``, you may instead use the `Symmetric` tag, along with a semidefinite constraint:

```julia
@variable(m, X[1:n,1:n], Symmetric)
@SDconstraint(m, X >= eye(n))
```

Bounds can be provided as normal when using the `Symmetric` tag, with the stipulation that the bounds are symmetric themselves.

`@variables` blocks
-------------------

JuMP provides a convenient syntax for defining multiple variables in a single block:

```julia
@variables m begin
    x
    y >= 0
    Z[1:10], Bin
    X[1:3,1:3], SDP
    q[i=1:2], (lowerbound = i, start = 2i, upperbound = 3i)
    t[j=1:3], (Int, start = j)
end

# Equivalent to:
@variable(m, x)
@variable(m, y >= 0)
@variable(m, Z[1:10], Bin)
@variable(m, X[1:3,1:3], SDP)
@variable(m, q[i=1:2], lowerbound = i, start = 2i, upperbound = 3i)
@variable(m, t[j=1:3], Int, start = j)
```

The syntax follows that of `@variable` with each declaration separated by a new line. Note that unlike in `@variable`, keyword arguments must be specified within parentheses.

Methods
-------

**Bounds**

```@docs
setlowerbound
getlowerbound
setupperbound
getupperbound
```

**Variable Category**

```@docs
setcategory
getcategory
```

**Values**

```@docs
getvalue(v::Variable)
getvalue(arr::Array{Variable})
setvalue
getdual
```

!!! note
    The `getvalue` function always returns a floating-point value, even when a variable is constrained to take integer values, as most solvers only guarantee integrality up to a particular numerical tolerance. The built-in `round` function should be used to obtain integer values, e.g., by calling `round(Integer, getvalue(x))`.

**Names**

Variables (in the sense of columns) can have internal names (different from the Julia variable name) that can be used for writing models to file. This feature is disabled for performance reasons, but will be added if there is demand or a special use case.

```@docs
setname
getname
```


Helper functions
----------------
The following built-in functions are overloaded to provide easy construction of expressions from variables,

-   `sum(x)` - Operates on arrays of variables, efficiently produces an affine expression. Available in macros.
-   `dot(x, coeffs)` - Performs a generalized "dot product" for arrays of variables and coefficients up to three dimensions, or equivalently the sum of the elements of the Hadamard product. Available in macros, and also as `dot(coeffs, x)`.



Fixed variables
---------------

Fixed variables, created with the `x == fixedval` syntax, have slightly special semantics. First, it is important to note that fixed variables are considered optimization variables, not constants, for the purpose of determining the problem class. For example, in:

```julia
@variable(m, x == 5)
@variable(m, y)
@constraint(m, x*y <= 10)
```

the constraint added is a nonconvex quadratic constraint. For efficiency reasons, JuMP will *not* substitute the constant `5` for `x` and then provide the resulting *linear* constraint to the solver. Two possible uses for fixed variables are:

1.  For computing sensitivities. When available from the solver, the sensitivity of the objective with respect to the fixed value may be queried with `getdual(x)`.
2.  For solving a sequence of problems with varying parameters. One may call `JuMP.fix(x, val)` to change the value of a fixed variable or to fix a previously unfixed variable. For LPs in particular, most solvers are able to efficiently hot-start when solving the resulting modified problem.
