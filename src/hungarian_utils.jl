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
    out = similar(costMatrix)
    for jj in 1:length(colOffsets), ii in 1:length(rowOffsets)
        out[ii, jj] = costMatrix[ii, jj] - (rowOffsets[ii] + colOffsets[jj])
    end
    return out
end

function adjusted_cost(costMatrix::Array{T, 2},
                       astate::AssignmentState{G, T}) where {G <: Integer, T <: Real}
    out = similar(costMatrix)
    for jj in 1:astate.ncol, ii in 1:astate.nrow
        out[ii, jj] = costMatrix[ii, jj] - (astate.rowPrices[ii] + astate.colPrices[jj])
    end
    return out
end

"""
    zero_cost(ii::Integer, jj::Integer, costMatrix::Array{G, 2}, rowOffsets::Array{G, 1}, colOffsets::Array{G, 1}) where G <: Real -> Bool

Computes the adjusted  cost by calling the `adkisted_cost` function and then `iszero` if this returns false the function checks if the adjusted cost is within `max(eps(rowOffset[ii]), eps(colOffset[jj]))`.  The second check was added to deal with issues of numeric precision that arose in test cases.  
"""
function zero_cost(ii::Integer, jj::Integer, costMatrix::Array{T, 2},
                   rowOffsets::Array{T, 1},
                   colOffsets::Array{T, 1}) where T <: AbstractFloat
    offsettot = rowOffsets[ii] + colOffsets[jj]
    return (costMatrix[ii, jj] == offsettot) || isapprox(costMatrix[ii, jj], offsettot)
    #return isapprox(adjusted_cost(ii, jj, costMatrix, rowOffsets, colOffsets), zero(T))
    #adjcost = adjusted_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
    #if iszero(adjcost) || (abs(adjcost) < max(eps(rowOffsets[ii]), eps(colOffsets[jj])))
    #    return true
    #else
    #    return false
    #end
end

function zero_cost(ii::Integer, jj::Integer, costMatrix::Array{G, 2},
                   rowOffsets::Array{G, 1},
                   colOffsets::Array{G, 1}) where G <: Integer
    return iszero(adjusted_cost(ii, jj, costMatrix, rowOffsets, colOffsets))
end
