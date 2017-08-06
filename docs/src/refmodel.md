```@meta
CurrentModule = JuMP
```

Models
======

Constructor
-----------

`Model` is a type defined by JuMP. All variables and constraints are associated with a `Model` object. It has a constructor that has no required arguments:

```julia
m = Model()
```

The constructor also accepts an optional keyword argument, `solver`. You may specify a solver either here or later on by calling `setsolver`. JuMP will throw an error if you try to solve a problem without specifying a solver.

`solver` must be an `AbstractMathProgSolver` object, which is constructed as follows:

```julia
solver = solvername(Option1=Value1, Option2=Value2, ...)
```

where `solvername` is one of the supported solvers. See the solver table &lt;jump-solvertable&gt; for the list of available solvers and corresponding parameter names. All options are solver-dependent; see corresponding solver packages for more information.

!!! note
    Be sure that the solver provided supports the problem class of the model. For example `ClpSolver` and `GLPKSolverLP` support only linear programming problems. `CbcSolver` and `GLPKSolverMIP` support only mixed-integer programming problems.

As an example, we can create a `Model` object that will use GLPK's exact solver for LPs as follows:

```julia
m = Model(solver = GLPKSolverLP(method=:Exact))
```

Methods
-------

**General**

```@docs
MathProgBase.numvar
MathProgBase.numlinconstr
MathProgBase.numquadconstr
numsocconstr
numsosconstr
numsdconstr
numnlconstr
MathProgBase.numconstr
internalmodel
solve
build
setsolver
Base.getindex(m::Model, name::Symbol)
Base.setindex!(m::Model, value, name::Symbol)
```

**Objective**

```@docs
getobjective
getobjectivesense
setobjectivesense
getobjectivevalue
getobjectivebound
```

**Solver**

These functions are JuMP versions of the similarly named functions in MathProgBase.

```@docs
getsolvetime
getnodecount
getobjbound
getobjgap
getrawsolver
getsimplexiter
getbarrieriter
```


**Output**

```@docs
writeLP
writeMPS
```

Solve Status
------------

The call `status = solve(m)` returns a symbol recording the status of the optimization process, as reported by the solver. Typical values are listed in the table below, although the code can take solver-dependent values. For instance, certain solvers prove infeasibility or unboundedness during presolve, but do not report which of the two cases holds. See your solver interface documentation (as linked to in the solver table &lt;jump-solvertable&gt;) for more information.


| Status        | Meaning                                 |
| ------------- | --------------------------------------- |
| `:Optimal`    | Problem solved to optimality            |
| `:Unbounded`  | Problem is unbounded                    |
| `:Infeasible` | Problem is infeasible                   |
| `:UserLimit`  | Iteration limit or timeout              |
| `:Error`      | Solver exited with an error             |
| `:NotSolved`  | Model built in memory but not optimized |


Quadratic Objectives
--------------------

Quadratic objectives are supported by JuMP using a solver which implements the corresponding extensions of the MathProgBase interface. Add them in the same way you would a linear objective:

```julia
using Ipopt
m = Model(solver=IpoptSolver())
@variable(m, 0 <= x <= 2 )
@variable(m, 0 <= y <= 30 )

@objective(m, Min, x*x+ 2x*y + y*y )
@constraint(m, x + y >= 1 )

print(m)

status = solve(m)
```

Second-order cone constraints
-----------------------------

Second-order cone constraints of the form ``||Ax − b||_2 + a^Tx + c \leq 0`` can be added directly using the `norm` function:

```julia
@constraint(m, norm(A*x) <= 2w - 1)
```

You may use generator expressions within `norm()` to build up normed expressions with complex indexing operations in much the same way as with `sum(...)`:

```julia
@constraint(m, norm(2x[i] - i for i=1:n if c[i] == 1) <= 1)
```

Accessing the low-level model
-----------------------------

It is possible to construct the internal low-level model before optimizing. To do this, call the `JuMP.build` function. It is then possible to obtain this model by using the `internalmodel` function. This may be useful when it is necessary to access some functionality that is not exposed by JuMP. When you are ready to optimize, simply call `solve` in the normal fashion.
