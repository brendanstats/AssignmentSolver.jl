"""
    adjusted_cost(ii::Integer, jj::Integer, costMatrix::Array{G, 2}, rowOffsets::Array{G, 1}, colOffsets::Array{G, 1}) where G <: Real -> adjustedCostMatrix[ii, jj]::G
    adjusted_cost(costMatrix::Array{G, 2}, rowOffsets::Array{G, 1}, colOffsets::Array{G, 1}) where G <: Real -> adjustedCostMatrix::Array{G, 2}

 Compute the cost adjusted for row and column offsets where:
`adjustedCostMatrix[ii, jj] = costMatrix[ii, jj] - rowOffsets[ii] - colOffsets[jj]`.  If
indicies are passed then only a single adjusted cost is returned.  Otherwise the entire
matrix is returned.
"""
function adjusted_cost(ii::Integer, jj::Integer,
                       costMatrix::Array{G, 2},
                       rowOffsets::Array{G, 1},
                       colOffsets::Array{G, 1}) where G <: Real
    return costMatrix[ii, jj] - (rowOffsets[ii] + colOffsets[jj])
end

function adjusted_cost(ii::Integer, jj::Integer,
                       costMatrix::Array{T, 2},
                       astate::AssignmentState{G, T}) where {G <: Integer, T <: Real}
    return costMatrix[ii, jj] - (astate.rowPrices[ii] + astate.colPrices[jj])
end

function adjusted_cost(costMatrix::Array{G, 2},
                       rowOffsets::Array{G, 1},
                       colOffsets::Array{G, 1}) where G <: Real
    out = Array{G}(size(costMatrix))
    for jj in 1:length(colOffsets), ii in 1:length(rowOffsets)
        out[ii, jj] = costMatrix[ii, jj] - (rowOffsets[ii] + colOffsets[jj])
    end
    return out
end

function adjusted_cost(costMatrix::Array{T, 2},
                       astate::AssignmentState{G, T}) where {G <: Integer, T <: Real}
    out = Array{T}(size(costMatrix))
    for jj in 1:astate.ncol, ii in 1:astate.nrow
        out[ii, jj] = costMatrix[ii, jj] - (astate.rowPrices[ii] + astate.colPrices[jj])
    end
    return out
end

"""
    zero_cost(ii::Integer, jj::Integer, costMatrix::Array{G, 2}, rowOffsets::Array{G, 1}, colOffsets::Array{G, 1}) where G <: Real -> Bool

Computes the adjusted  cost by calling the `adkisted_cost` function and then `iszero` if this returns false the function checks if the adjusted cost is within `max(eps(rowOffset[ii]), eps(colOffset[jj]))`.  The second check was added to deal with issues of numeric precision that arose in test cases.  
"""
function zero_cost(ii::Integer, jj::Integer, costMatrix::Array{G, 2},
                   rowOffsets::Array{G, 1},
                   colOffsets::Array{G, 1}) where G <: Real
    adjcost = adjusted_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
    if iszero(adjcost) || (abs(adjcost) < max(eps(rowOffsets[ii]), eps(colOffsets[jj])))
        return true
    else
        return false
    end
end
