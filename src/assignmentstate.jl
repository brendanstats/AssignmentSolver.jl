"""
    mutable struct AssignmentState{G<:Integer, T<:Real}

Assignment type to track assignment and duel variable states.

# Fields

* `r2c::Array{G, 1}`: Mapping from row to column.  r2c[ii] = jj if row ii is linked to column jj.  If row ii is unlinked then r2c[ii] == 0.
* `c2r::Array{G, 1}`: Mapping from column to row.  c2r[jj] = ii if column jj is linked to row ii.  If column jj is unlinked then c2r[jj] == 0.
* `rowPrices::Array{T, 1}`: Row dual variables, prices for auction algorithms and costs for hungarian algorithms.
* `colPrices::Array{T, 1}`: Column dual variables, prices for auction algorithms and costs for hungarian algorithms.
* `nassigned::G`: Number of assignments in
* `nrow::G`: Number of rows equal to nrow == length(r2c) == length(rowPrices).
* `ncol::G`: Number of columns equal to ncol == length(c2r) == length(colPrices).
* `nexcesscols::G`: ncol - nrow
* `sym::Bool`: Indicator if nrow == ncol.

# Constructors

    AssignmentState(rewardMatrix::Array{T, 2}; maximize = true, assign = true, pad = false, dfltReward::T = zero(T)) where T <: AbstractFloat
    AssignmentState(rewardMatrix::SparseMatrixCSC{T, G}; maximize = true, assign = true, pad = false, dfltReward::T = zero(T)) where {G <: Integer, T <: AbstractFloat}
    AssignmentState(rowPrices::Array{T, 1}, colPrices::Array{T, 1}) where T <: AbstractFloat
    AssignmentState(r2c::Array{G, 1}, c2r::Array{G, 1}, rowPrices::Array{T, 1}, colsPrices::Array{T, 1})  where {G <: Integer, T <: AbstractFloat}

# Arguments

* `rewardMatrix::Union{Array{T, 2}, SparseMatrixCSC{T, G}}`:
* `maximize::Bool`: Is a maximal (auction algorithm) or minimal (hungarian algorithm) to be found.  This determines initialization of rowPrices and colPrices.
* `assign::Bool`: Should rules of thumb be used to set an initial assignment or should it be left empty.
* `pad::Bool`: Should padding be added to assure an feasible assignment.  If so dimension of problem is expanded to nrow x (ncol + nrow) with entries assumed in the rewardMatrix at ii, ii + ncol with rewards of dfltReward.
* `dfltReward::T = zero(T)`: Default reward to be used in padding.

"""
mutable struct AssignmentState{G<:Integer, T<:Real} 
    r2c::Array{G, 1}
    c2r::Array{G, 1}
    rowPrices::Array{T, 1}
    colPrices::Array{T, 1}
    nassigned::G
    nrow::G
    ncol::G
    nexcesscols::G
    sym::Bool
end

function AssignmentState(rewardMatrix::Array{T, 2}; maximize = true, assign = true, pad = false, dfltReward::T = zero(T)) where T <: Real
    nrow = size(rewardMatrix, 1)
    if pad
        ncol = nrow + size(rewardMatrix, 2)
    else
        ncol = size(rewardMatrix, 2)
    end
    
    if maximize
        if assign
            rowPrices, r2c = findrowmax(rewardMatrix)
        else
            rowPrices = vec(maximum(rewardMatrix, dims=2))
            r2c = zeros(Int, nrow)
        end
    else
        if assign
            rowPrices, r2c = findrowmin(rewardMatrix)
        else
            rowPrices = vec(minimum(rewardMatrix, dims=2))
            r2c = zeros(Int, nrow)
        end
    end

    if pad
        for ii in 1:nrow
            if rowPrices[ii] < dfltReward
                rowPrices[ii] = dfltReward
                r2c[ii] = ii + size(rewardMatrix, 2)
            end
        end
    end
    
    c2r = zeros(Int, ncol)
    colPrices = zeros(T, ncol)

    nassigned = 0
    if assign
        for ii in 1:length(r2c)
            if !iszero(r2c[ii]) && iszero(c2r[r2c[ii]])
                c2r[r2c[ii]] = ii
                nassigned += 1
            else
                r2c[ii] = 0
            end
        end
    end
    return AssignmentState(r2c, c2r, rowPrices, colPrices, nassigned, nrow, ncol, ncol - nrow, nrow == ncol)
end

function AssignmentState(rewardMatrix::SparseMatrixCSC{T, G}; maximize = true, assign = true, pad = false, dfltReward::T = zero(T)) where {G <: Integer, T <: AbstractFloat}
    nrow = G(size(rewardMatrix, 1))
    if pad
        ncol = nrow + G(size(rewardMatrix, 2))
    else
        ncol = G(size(rewardMatrix, 2))
    end
    
    if maximize
        if assign
            rowPrices, r2c = findrowmax(rewardMatrix)
        else
            rowPrices = vec(maximum(rewardMatrix, dims=2))
            r2c = zeros(G, nrow)
        end
    else
        if assign
            rowPrices, r2c = findrowmin(rewardMatrix)
        else
            rowPrices = vec(minimum(rewardMatrix, dims=2))
            r2c = zeros(G, nrow)
        end
    end

    if pad
        for ii in 1:nrow
            if rowPrices[ii] < dfltReward
                rowPrices[ii] = dfltReward
                r2c[ii] = ii + size(rewardMatrix, 2)
            end
        end
    end
    
    c2r = zeros(G, ncol)
    colPrices = zeros(T, ncol)

    nassigned = zero(G)
    if assign
        for ii in one(G):nrow
            if !iszero(r2c[ii]) && iszero(c2r[r2c[ii]])
                c2r[r2c[ii]] = ii
                nassigned += one(G)
            else
                r2c[ii] = zero(G)
            end
        end
    end
    return AssignmentState(r2c, c2r, rowPrices, colPrices, nassigned, nrow, ncol, ncol - nrow, nrow == ncol)
end

function AssignmentState(rowPrices::Array{T, 1}, colPrices::Array{T, 1}) where T <: Real
    return AssignmentState(zeros(Int, length(rowPrices)),
                           zeros(Int, length(colPrices)),
                           rowPrices,
                           colPrices,
                           zero(Int),
                           length(rowPrices),
                           length(colPrices),
                           length(colPrices) - length(rowPrices),
                           length(colPrices) == length(rowPrices))
end

function AssignmentState(r2c::Array{G, 1}, c2r::Array{G, 1}, rowPrices::Array{T, 1}, colsPrices::Array{T, 1})  where {G <: Integer, T <: Real}
    return AssignmentState(r2c,
                           c2r,
                           rowPrices,
                           colPrices,
                           G(count(.!iszero.(r2c))),
                           G(length(rowPrices)),
                           G(length(colPrices)),
                           G(length(colPrices) - length(rowPrices)),
                           length(colPrices) == length(rowPrices))
end

"""
    clear_assignment!(astate::AssignmentState{G, T}) where {G <: Integer, T <: Real} -> astate

Unassign all rows and columns in `astate`.
"""
function clear_assignment!(astate::AssignmentState{G, T}) where {G <: Integer, T <: Real}
    astate.r2c .= zero(G)
    for jj in 1:astate.ncol
        if !iszero(astate.c2r[jj])
            astate.c2r[jj] = zero(G)
        end
    end
    astate.nassigned = zero(G)
    return astate
end

"""
    function adjust_inf!(astate::Assignment{G, T}, lambda::T = zero(T), dfltReward::T = zero(T)) where {G <: Integer, T <: Real}

Finds rows for which `isinf(astate.rowPrices[row])`, sets row prices to `dfltReward - lambda` and of the assigned column prices to `lambda`.

This is intended to be used as a post-processing function for case where `dfltTwo = Inf` but finite prices are needed
for downstream applications.  The `Inf` is taken to indicate that only a single column was available to the row and only
a single row available to the column.  Only appropriate for use if algorithm was run with `pad = true`.
"""
function adjust_inf!(astate::AssignmentState{G, T}, lambda::T = zero(T), dfltReward::T = zero(T)) where {G <: Integer, T <: Real}
    for ii in one(G):astate.nrow
        if isinf(astate.rowPrices[ii])
            astate.rowPrices[ii] = dfltReward - lambda
            astate.colPrices[astate.r2c[ii]] = lambda
        end
    end
    return astate
end

"""
    remove_padded!(astate::Assignment{G, T}, ncol::Integer) where {G <: Integer, T <: Real}

Removes assignments to padded, implicitly added, columns where `ncol` is the original number of columns.

This is intended to be used as a post-processing function if padded columns are added to allow rows to remain unassigned.
"""
function remove_padded!(astate::AssignmentState{G, T}, ncol::Integer) where {G <: Integer, T <: Real}
    for ii in one(G):astate.nrow
        if astate.r2c[ii] > ncol
            astate.c2r[astate.r2c[ii]] = zero(G)
            astate.r2c[ii] = zero(G)
            astate.nassigned -= one(G)
        end
    end
    return astate
end

"""
    flip(astate::AssignmentState{G, T}) where {G <: Integer, T <: Real} -> AssignmentState{G, T}

Take transpose of `astate` 
"""
function flip(astate::AssignmentState{G, T}) where {G <: Integer, T <: Real}
    return AssignmentState(astate.c2r,
                           astate.r2c,
                           astate.colPrices,
                           astate.rowPrices,
                           astate.nassigned,
                           astate.ncol,
                           astate.nrow,
                           -astate.nexcesscols,
                           astate.sym)
end

"""
    compute_objective(astate::AssignmentState, rewardMatrix::Union{Array{T, 2}, SparseMatrixCSC{T, <:Integer}}, dfltReward::T = zero(T)) where {G <: Integer, T <: Real}
    compute_objective(r2c::Array{<:Integer, 1}, rewardMatrix::Union{Array{T, 2}, SparseMatrixCSC{T, <:Integer}}, dfltReward::T = zero(T)) where {G <: Integer, T <: Real}

Compute objective value of assignment.
"""
function compute_objective(astate::AssignmentState, rewardMatrix::Union{Array{T, 2}, SparseMatrixCSC{T, <:Integer}}, dfltReward::T = zero(T)) where {G <: Integer, T <: Real}
    tot = zero(T)
    for row in 1:astate.nrow
        if !iszero(astate.r2c[row])
            if row <= size(rewardMatrix, 1) && astate.r2c[row] <= size(rewardMatrix, 2)
                tot += rewardMatrix[row, astate.r2c[row]]
            else
                tot += dfltReward
            end
        end
    end
    return tot
end

function compute_objective(r2c::Array{<:Integer, 1}, rewardMatrix::Union{Array{T, 2}, SparseMatrixCSC{T, <:Integer}}, dfltReward::T = zero(T)) where {G <: Integer, T <: Real}
    tot = zero(T)
    for row in 1:length(r2c)
        if !iszero(r2c[row])
            if row <= size(rewardMatrix, 1) && r2c[row] <= size(rewardMatrix, 2)
                tot += rewardMatrix[row, r2c[row]]
            else
                tot += dfltReward
            end
        end
    end
    return tot
end

"""
    pad_matrix(rewardMatrix::Array{T, 2}, dfltReward::T = zero(T)) where T <: Real
    pad_matrix(rewardMatrix::SparseMatrixCSC{T, G}, dfltReward::T = zero(T)) where {G <: Integer, T <: Real}

Add a dummy column for each row with value `dfltRow` is sparse then entries are only added on the diagonal.
"""
function pad_matrix(rewardMatrix::Array{T, 2}, dfltReward::T = zero(T)) where T <: Real
    return hcat(rewardMatrix, fill(dfltReward, size(rewardMatrix, 1), size(rewardMatrix, 1)))
end

function pad_matrix(rewardMatrix::SparseMatrixCSC{T, G}, dfltReward::T = zero(T)) where {G <: Integer, T <: Real}
    return SparseMatrixCSC(rewardMatrix.m, rewardMatrix.n + rewardMatrix.m, [rewardMatrix.colptr;  collect(range(rewardMatrix.colptr[end] + one(G), length=rewardMatrix.m))], [rewardMatrix.rowval; collect(one(G):rewardMatrix.m)], [rewardMatrix.nzval; fill(dfltReward, rewardMatrix.m)])
end

"""
    reward2cost(rewardMatrix::Array{T, 2}, maxreward::T = maximum(rewardMatrix)) where T <: Real

Convert reward matrix to cost (vice versa), returned value is maximum .- rewardMatrix unless maxreward is set to a different value.
"""
function reward2cost(rewardMatrix::Array{T, 2}, maxreward::T = maximum(rewardMatrix)) where T <: Real    
    return maxreward .- rewardMatrix
end
