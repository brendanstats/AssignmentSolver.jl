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

function AssignmentState(rewardMatrix::Array{T, 2}; maximize = true, assign = true, pad = false, dfltReward::T = zero(T)) where T <: AbstractFloat
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

function AssignmentState(rewardMatrix::SparseMatrixCSC{T, G}; maximize = true, assign = true, pad = false, dfltReward::T = zero(T)) where T <: AbstractFloat
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

function AssignmentState(rowPrices::Array{T, 1}, colPrices::Array{T, 1}) where T <: AbstractFloat
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

function AssignmentState(r2c::Array{G, 1}, c2r::Array{G, 1}, rowPrices::Array{T, 1}, colsPrices::Array{T, 1})  where {G <: Integer, T <: AbstractFloat}
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
    flip(astate::AssignmentState{G, T}) where {G <: Integer, T <: Real}
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
