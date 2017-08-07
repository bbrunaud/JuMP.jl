#  Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# JuMP
# An algebraic modeling language for Julia
# See http://github.com/JuliaOpt/JuMP.jl
#############################################################################
# src/indicator.jl
# Defines IndicatorConstraint of the type y = 1 => ax <= b
# Supported by CPLEX, Gurobi (since version 7.0), and SCIP
#############################################################################


type IndicatorConstraint <: AbstractConstraint
    binvar::Variable
    binvalue::Int32
    linconstraint::LinearConstraint
end
Base.copy(ic::IndicatorConstraint, new_model::Model) =
    IndicatorConstraint(copy(ic.binvar, new_model),
                    copy(ic.binval), copy(ic.linconstraint))


function addindicatorconstraint!(m::Model, c::IndicatorConstraint)
    push!(m.indconstr,c)
    if m.internalModelLoaded
        if method_exists(MathProgBase.addindconstr!, (typeof(m.internalModel),Int32, Int32, Vector{Int},Vector{Float64},Float64,Float64))
            assert_isfinite(c.linconstraint.terms)
            indices, coeffs = merge_duplicates(Cint, c.linconstraint.terms, m.indexedVector, m)
            MathProgBase.addindconstr!(m.internalModel, c.binvar.col, c.binvalue, indices,coeffs,c.linconstraint.lb,c.linconstraint.ub)
        else
            Base.warn_once("Solver does not appear to support adding constraints to an existing model. JuMP's internal model will be discarded.")
            m.internalModelLoaded = false
        end
    end
    # return LinConstrRef(m,length(m.indconstr))
end
addindicatorconstraint!(m::Model, c::Array{IndicatorConstraint}) =
    error("The operators <=, >=, and == can only be used to specify scalar constraints. If you are trying to add a vectorized constraint, use the element-wise dot comparison operators (.<=, .>=, or .==) instead")

function addVectorizedIndicatorConstraint(m::Model, v::Array{IndicatorConstraint})
    ret = Array(LinConstrRef, size(v))
    for I in eachindex(v)
        ret[I] = addindicatorconstraint(m, v[I])
    end
    ret
end

# two-argument constructconstraint! is used for one-sided constraints.
# Right-hand side is zero.
constructindconstraint!(indvar::Variable,indval::Number,v::Variable, sense::Symbol) = constructindconstraint!(indvar,indval,convert(AffExpr,v), sense)
function constructindconstraint!(indvar::Variable,indval::Number,aff::AffExpr, sense::Symbol)
    indval = convert(Int,indval)
    offset = aff.constant
    aff.constant = 0.0
    if sense == :(<=) || sense == :≤
        return IndicatorConstraint(indvar,indval,LinearConstraint(aff, -Inf, -offset))
    elseif sense == :(>=) || sense == :≥
        return IndicatorConstraint(indvar,indval,LinearConstraint(aff, -offset, Inf))
    elseif sense == :(==)
        return IndicatorConstraint(indvar,indval,LinearConstraint(aff, -offset, -offset))
    else
        error("Cannot handle ranged constraint")
    end
end

function constructindconstraint!(indvar::Variable,indval::Number,aff::AffExpr, lb, ub)
    indval = convert(Int,indval)
    offset = aff.constant
    aff.constant = 0.0
    IndicatorConstraint(indvar,indval,LinearConstraint(aff, lb-offset, ub-offset))
end


"""
    @indicatorconstraint(m::Model, ind=1/0, con)

add linear indicator constraints of the type y = 1 => ax <= b
Supported by CPLEX, Gurobi (since version 7.0), and SCIP.

    @constraint(m::Model, ref, ind=1/0, con)

add groups of indicator constraints.

"""
macro indicatorconstraint(args...)
    # Pick out keyword arguments
    if isexpr(args[1],:parameters) # these come if using a semicolon
        kwargs = args[1]
        args = args[2:end]
    else
        kwargs = Expr(:parameters)
    end
    kwsymbol = VERSION < v"0.6.0-dev.1934" ? :kw : :(=) # changed by julia PR #19868
    append!(kwargs.args, filter(x -> isexpr(x, kwsymbol), collect(args))) # comma separated
    args = filter(x->!isexpr(x, kwsymbol), collect(args))

    if length(args) < 3
        if length(kwargs.args) > 0
            constraint_error(args, "Not enough positional arguments")
        else
            constraint_error(args, "Not enough arguments")
        end
    end
    m = args[1]
    ind = args[2]
    x = args[3]
    extra = args[4:end]

    m = esc(m)
    # Two formats:
    # - @indicatorconstraint(m, y = 1, a*x <= 5)
    # - @indicatorconstraint(m, myref[a=1:5], y = 1, a*x <= 5)
    length(extra) > 1 && constraint_error(args, "Too many arguments.")
    # Canonicalize the arguments
    c = length(extra) == 1 ? ind        : gensym()
    ind = length(extra) == 1 ? x : ind
    x = length(extra) == 1 ? extra[1] : x

    anonvar = isexpr(c, :vect) || isexpr(c, :vcat) || length(extra) != 1
    variable = gensym()
    quotvarname = quot(getname(c))
    escvarname  = anonvar ? variable : esc(getname(c))

    if isa(x, Symbol)
        constraint_error(args, "Incomplete constraint specification $x. Are you missing a comparison (<=, >=, or ==)?")
    end

    (x.head == :block) &&
        constraint_error(args, "Code block passed as constraint. Perhaps you meant to use @indicatorconstraints instead?")

    ind.head != :call && constraint_error(args, "Need to pass a expression of type y == 1, ax <= b")
    indvar = ind.args[2]
    indvar = esc(indvar)
    indval = ind.args[3]
    !(indval in (1,0)) && constraint_error(args, "The indicator value must either 0 or 1")

    # Strategy: build up the code for non-macro addconstraint, and if needed
    # we will wrap in loops to assign to the ConstraintRefs
    refcall, idxvars, idxsets, idxpairs, condition = buildrefsets(c, variable)
    # JuMP accepts constraint syntax of the form @constraint(m, foo in bar).
    # This will be rewritten to a call to constructconstraint!(foo, bar). To
    # extend JuMP to accept set-based constraints of this form, it is necessary
    # to add the corresponding methods to constructconstraint!. Note that this
    # will likely mean that bar will be some custom type, rather than e.g. a
    # Symbol, since we will likely want to dispatch on the type of the set
    # appearing in the constraint.
    if isexpr(x, :call)
        if x.args[1] == :in
            @assert length(x.args) == 3
            newaff, parsecode = parseExprToplevel(x.args[2], :q)
            constraintcall = :(addconstraint($m, constructindconstraint!($indvar,$indval,$newaff,$(esc(x.args[3])))))
        else
            # Simple comparison - move everything to the LHS
            @assert length(x.args) == 3
            (sense,vectorized) = _canonicalize_sense(x.args[1])
            lhs = :($(x.args[2]) - $(x.args[3]))
            addindconstr = (vectorized ? :addVectorizedIndicatorConstraint : :addindicatorconstraint!)
            newaff, parsecode = parseExprToplevel(lhs, :q)
            constraintcall = :($addindconstr($m, constructindconstraint!($indvar,$indval,$newaff,$(quot(sense)))))
        end
        addkwargs!(constraintcall, kwargs.args)
        code = quote
            q = zero(AffExpr)
            $parsecode
            $(refcall) = $constraintcall
        end
    elseif isexpr(x, :comparison)
        # Ranged row
        (lsign,lvectorized) = _canonicalize_sense(x.args[2])
        (rsign,rvectorized) = _canonicalize_sense(x.args[4])
        if (lsign != :(<=)) || (rsign != :(<=))
            constraint_error(args, "Only ranged rows of the form lb <= expr <= ub are supported.")
        end
        ((vectorized = lvectorized) == rvectorized) || constraint_error("Signs are inconsistently vectorized")
        addconstr = (lvectorized ? :addVectorizedConstraint : :addconstraint)
        x_str = string(x)
        lb_str = string(x.args[1])
        ub_str = string(x.args[5])
        newaff, parsecode = parseExprToplevel(x.args[3],:aff)

        newlb, parselb = parseExprToplevel(x.args[1],:lb)
        newub, parseub = parseExprToplevel(x.args[5],:ub)

        constraintcall = :($addconstr($m, constructconstraint!($newaff,$newlb,$newub)))
        addkwargs!(constraintcall, kwargs.args)
        code = quote
            aff = zero(AffExpr)
            $parsecode
            lb = 0.0
            $parselb
            ub = 0.0
            $parseub
        end
        if vectorized
            code = quote
                $code
                lbval, ubval = $newlb, $newub
            end
        else
            code = quote
                $code
                CoefType = coeftype($newaff)
                try
                    lbval = convert(CoefType, $newlb)
                catch
                    constraint_error($args, string("Expected ",$lb_str," to be a ", CoefType, "."))
                end
                try
                    ubval = convert(CoefType, $newub)
                catch
                    constraint_error($args, string("Expected ",$ub_str," to be a ", CoefType, "."))
                end
            end
        end
        code = quote
            $code
            $(refcall) = $constraintcall
        end
    else
        # Unknown
        constraint_error(args, string("Constraints must be in one of the following forms:\n" *
              "       expr1 <= expr2\n" * "       expr1 >= expr2\n" *
              "       expr1 == expr2\n" * "       lb <= expr <= ub"))
    end
    return assert_validmodel(m, quote
        $(getloopedcode(variable, code, condition, idxvars, idxsets, idxpairs, :ConstraintRef))
        $(if anonvar
            variable
        else
            quote
                registercon($m, $quotvarname, $variable)
                $escvarname = $variable
            end
        end)
    end)
end
