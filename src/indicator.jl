#  Copyright 2016, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
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
    return LinConstrRef(m,length(m.linconstr))
end
addindicatorconstraint(m::Model, c::Array{IndicatorConstraint}) =
    error("The operators <=, >=, and == can only be used to specify scalar constraints. If you are trying to add a vectorized constraint, use the element-wise dot comparison operators (.<=, .>=, or .==) instead")

function addVectorizedIndicatorConstraint(m::Model, v::Array{IndicatorConstraint})
    ret = Array(LinConstrRef, size(v))
    for I in eachindex(v)
        ret[I] = addindicatorconstraint(m, v[I])
    end
    ret
end

