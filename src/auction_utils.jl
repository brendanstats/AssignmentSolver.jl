"""
    maxtwo_column(col::Integer, rowPrices::Array{T, 1}, rewardMatrix::Array{T, 2}) where T <: AbstractFloat -> (maxidx::G, maxval::T, maxtwo::T)
    maxtwo_column(col::Integer, rowPrices::Array{T, 1}, rewardMatrix::Array{T, 2}, dfltTwo::T) where T <: AbstractFloat -> (maxidx::G, maxval::T, maxtwo::T)
    maxtwo_column(col::G, rowPrices::Array{T, 1}, rewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (maxidx::G, maxval::T, maxtwo::T)
    maxtwo_column(col::G, rowPrices::Array{T, 1}, rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (maxidx::G, maxval::T, maxtwo::T)

Search over `rewardMatrix[:, col]` - `rowPrices` and return the index of the largest value,
the largest value and the second largest value.

`dfltTwo` must be supplied if the reward matrix is sparse so allow for cases where
`rewardMatrix[row, col]` is nonzero for only a single row.  If `dfltReward` is supplied
then a padded entry is implicity added setting `rewardMatrix[size(rewardMatrix, 1) + col, col]`
to `dfltReward`.  This is typically used with the transpose of a sparse reward matrix to maximize
across a row as done in `maxtwo_row`.

# Arguments

* `col::G`: Row for which the forward bid is computed.
* `rowPrices::Array{T, 1}`: column (individual) prices in auction algorithm.
* `rewardMatrix::Union{Array{T, 2}, SparseMatrixCSC{T, 2}`: reward matrix for which a maximal
 assignment is being found.
* `dfltTwo::T`: default second largest value using if only one row has an entry containing
`col` if `rewardMatrix` is sparse.  Ignored if `rewardMatrix` is not sparse.
* 'dfltReward::T': Default reward value used for implicity added entries if a padded reward
matrix is being used.

where `G <: Integer` `T <: AbstractFloat`

See also: [`maxtwo_row`](@ref), [`forward_bid`](@ref), [`reverse_bid`](@ref)

# Examples

```julia
rewardMatrix = sparse([1, 2, 3, 4, 3], [7, 1, 3, 2, 2], [0.5, 0.6, 0.7, 0.8, 0.4])
maxtwo_column(2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rewardMatrix)
```
"""
function maxtwo_column(col::G, rowPrices::Array{T, 1}, rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}

    ##Loop through to find largest and second largest value
    maxrow = 1
    maxval = rewardMatrix[1, col] - rowPrices[1]
    maxtwo = rewardMatrix[2, col] - rowPrices[2]
    if maxtwo > maxval
        val = maxval
        maxval = maxtwo
        maxtwo = val
        maxrow = 2
    end

    if size(rewardMatrix, 1) > 2
        for row in 3:size(rewardMatrix, 1)
            val = rewardMatrix[row, col] - rowPrices[row]
            if val > maxval
                maxtwo = maxval
                maxval = val
                maxrow = row
            elseif val > maxtwo
                maxtwo = val
            end
        end
    end

    return maxrow, maxval, maxtwo
end

maxtwo_column(col::G, rowPrices::Array{T, 1}, rewardMatrix::Array{T, 2}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} = maxtwo_column(col, rowPrices, rewardMatrix)

function maxtwo_column(col::G, rowPrices::Array{T, 1}, rewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}

    ##Column edges and values
    rng = nzrange(rewardMatrix, col)
    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)

    ##Check if any matcehs are found
    if length(rng) == 0
        error("no edges for column $col")
    end

    ##Case where second largest value does not exist
    if length(rng) == 1
        maxidx = rng[1]
        maxrow = rows[maxidx]
        return maxidx, rewards[maxidx] - rowPrices[maxrow], dfltTwo
    end

    ##Initialize maxval and maxidx
    maxidx = rng[1]
    maxval = rewards[maxidx] - rowPrices[rows[maxidx]]

    ##Initialize maxtwo
    val = rewards[rng[2]] - rowPrices[rows[rng[2]]]
    if val > maxval
        maxtwo = maxval
        maxval = val
        maxidx = rng[2]
    else
        maxtwo = val
    end
    
    #Loop over remaining values
    for idx in rng[3:end]
        row = rows[idx]
        val = rewards[idx] - rowPrices[row]
        if val > maxval
            maxtwo = maxval
            maxval = val
            maxidx = idx
        elseif val > maxtwo
            maxtwo = val
        end
    end
    return maxidx, maxval, maxtwo
end

function maxtwo_column(col::G, rowPrices::Array{T, 1}, rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                       dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    if col > size(rewardMatrix, 2)
        maxidx = -one(G)
        return maxidx, dfltReward - rowPrices[col - size(rewardMatrix, 2)], dfltTwo
    else
        return maxtwo_column(col, rowPrices, rewardMatrix, dfltTwo)
    end
end

"""
    maxtwo_row(col::Integer, rowPrices::Array{T, 1}, rewardMatrix::Array{T, 2}) where T <: AbstractFloat -> (maxidx::G, maxval::T, maxtwo::T)
    maxtwo_row(col::Integer, rowPrices::Array{T, 1}, rewardMatrix::Array{T, 2}, dfltTwo::T) where T <: AbstractFloat -> (maxidx::G, maxval::T, maxtwo::T)
    maxtwo_row(col::G, rowPrices::Array{T, 1}, rewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (maxidx::G, maxval::T, maxtwo::T)
    maxtwo_row(col::G, rowPrices::Array{T, 1}, rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (maxidx::G, maxval::T, maxtwo::T)

Search over `rewardMatrix[:, col]` - `rowPrices` and return the index of the largest value,
the largest value and the second largest value.

`dfltTwo` must be supplied if the reward matrix is sparse so allow for cases where
`rewardMatrix[row, col]` is nonzero for only a single row.  If `dfltReward` is supplied
then a padded entry is implicity added setting `rewardMatrix[size(rewardMatrix, 1) + col, col]`
to `dfltReward`.  This is typically used with the transpose of a sparse reward matrix to maximize
across a row as done in `maxtwo_row`.

# Arguments

* `col::G`: Row for which the forward bid is computed.
* `rowPrices::Array{T, 1}`: column (individual) prices in auction algorithm.
* `rewardMatrix::Array{T, 2}`: reward matrix for which a maximal assignment is being found.
* `trewardMatrix::SparseMatrixCSC{T, 2}`: transpose of reward matrix for which a maximal
assignment is being found.
* `dfltTwo::T`: default second largest value using if only one row has an entry containing
`col` if `rewardMatrix` is sparse.  Ignored if `rewardMatrix` is not sparse.
* 'dfltReward::T': Default reward value used for implicity added entries if a padded reward
matrix is being used.

where `G <: Integer` `T <: AbstractFloat`

See also: [`maxtwo_column`](@ref), [`forward_bid`](@ref), [`reverse_bid`](@ref)

"""
function maxtwo_row(row::Integer, colPrices::Array{T, 1}, rewardMatrix::Array{T, 2}) where T <: AbstractFloat

    ##Loop through to find largest and second largest value
    maxcol = 1
    maxval = rewardMatrix[row, 1] - colPrices[1]
    maxtwo = rewardMatrix[row, 2] - colPrices[2]
    if maxtwo > maxval
        val = maxval
        maxval = maxtwo
        maxtwo = val
        maxcol = 2
    end
    
    for col in 3:size(rewardMatrix, 2)
        val = rewardMatrix[row, col] - colPrices[col]
        if val > maxval
            maxtwo = maxval
            maxval = val
            maxcol = col
        elseif val > maxtwo
            maxtwo = val
        end
    end
    
    return maxcol, maxval, maxtwo
end

function maxtwo_row(row::G, colPrices::Array{T, 1}, rewardMatrix::Array{T, 2}, dfltReward::T) where {G <: Integer, T <: AbstractFloat}

    ##Initialize maxval and maxidx
    padval = dfltReward - colPrices[row + size(rewardMatrix, 2)]
    maxcol, maxval, maxtwo = maxtwo_row(row, colPrices, rewardMatrix)
    if padval > maxval
        maxtwo = maxval
        maxval = padval
        maxcol = row + size(rewardMatrix, 2)
    elseif padval > maxtwo
        maxtwo = padval
    end
    return maxcol, maxval, maxtwo
end

maxtwo_row(row::G, colPrices::Array{T, 1}, trewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} = maxtwo_column(row, colPrices, trewardMatrix, dfltTwo)

function maxtwo_row(row::G, colPrices::Array{T, 1}, trewardMatrix::SparseMatrixCSC{T, G}, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}

    ##Row edges and values
    rng = nzrange(trewardMatrix, row)
    cols = rowvals(trewardMatrix)
    rewards = nonzeros(trewardMatrix)
    padval = dfltReward - colPrices[row + trewardMatrix.m]

    ##Initialize maxval and maxidx
    maxidx = -one(G)
    maxval = padval
    
    ##Case where padded value is the only value
    if length(rng) == 0
        #-1 index indicates that padded entry has been used
        return maxidx, maxval, dfltTwo
    else

        ##Initialize maxtwo
        idx = rng[1]
        col = cols[idx]
        val = rewards[idx] - colPrices[col]
        if val > maxval
            maxtwo = maxval
            maxval = val
            maxidx = idx
        else
            maxtwo = val
        end
        
        #Loop over remaining values
        for idx in rng[2:end]
            col = cols[idx]
            val = rewards[idx] - colPrices[col]
            if val > maxval
                maxtwo = maxval
                maxval = val
                maxidx = idx
            elseif val > maxtwo
                maxtwo = val
            end
        end
        return maxidx, maxval, maxtwo
    end
end

"""
    get_zeros(x::Array{G, 1}) where G <: Integer

Construct a Queue containing the indicies for which `iszero(x[ii])`

See also: [`get_openrows`](@ref), [`get_opencols`](@ref)

"""
function get_zeros(x::Array{G, 1}) where G <: Integer
    zeroIdx = Queue{G}()
    for (ii, xi) in pairs(x)
        if iszero(xi)
            enqueue!(zeroIdx, ii)
        end
    end
    return zeroIdx
end

"""
    get_openrows(astate::AssignmentState)

Construct  a Queue containing indicies for which `astate.r2c[idx] == 0`, wrapper around `get_zeros`

See also: [`get_zeros`](@ref), [`get_opencols`](@ref)
"""
get_openrows(astate::AssignmentState) = get_zeros(astate.r2c)

"""
    get_opencols(astate::AssignmentState)

Construct a Queue containing indicies for which `astate.c2r[idx] == 0`, wrapper around `get_zeros`

See also: [`get_zeros`](@ref), [`get_openrows`](@ref)

"""
get_opencols(astate::AssignmentState) = get_zeros(astate.c2r)

"""
    get_opencolsabove(c2r::Array{G, 1}, colPrices::Array{T, 1}, lambda::T) where {G <: Integer, T <: AbstractFloat} -> nbelow, openColsAbove
    get_opencolsabove(astate::AssignmentState{G, T}, lambda::T) -> nbelow, openColsAbove

Columns where `colPrices[jj] > lambda` and `iszero(c2r[jj])` are added to the queue `openColsAbove`.
"""
function get_opencolsabove(c2r::Array{G, 1}, colPrices::Array{T, 1}, lambda::T) where {G <: Integer, T <: AbstractFloat}
    openColsAbove = Queue{G}()
    for jj in 1:length(c2r)
        if iszero(c2r[jj]) && colPrices[jj] > lambda
            enqueue!(openColsAbove, jj)
        end
    end
    return openColsAbove
end

get_opencolsabove(astate::AssignmentState{G, T}, lambda::T) where {G <: Integer, T <: AbstractFloat} = get_opencolsabove(astate.c2r, astate.colPrices, lambda)

"""
    get_nbelow_opencolsabove(c2r::Array{G, 1}, colPrices::Array{T, 1}, lambda::T) where {G <: Integer, T <: AbstractFloat} -> nbelow, openColsAbove
    get_nbelow_opencolsabove(astate::AssignmentState{G, T}, lambda::T) -> nbelow, openColsAbove

Count the number of columns `jj` where `colPrices[jj] < lambda`, columns where `colPrices[jj] > lambda` and `iszero(c2r[jj])` are added to the queue `openColsAbove`.
"""
function get_nbelow_opencolsabove(c2r::Array{G, 1}, colPrices::Array{T, 1}, lambda::T) where {G <: Integer, T <: AbstractFloat}
    nbelow = zero(G)
    openColsAbove = Queue{G}()
    for jj in 1:length(c2r)
        if colPrices[jj] < lambda
            nbelow += one(G)
        elseif iszero(c2r[jj]) && colPrices[jj] > lambda
            enqueue!(openColsAbove, jj)
        end
    end
    return nbelow, openColsAbove
end

get_nbelow_opencolsabove(astate::AssignmentState{G, T}, lambda::T) where {G <: Integer, T <: AbstractFloat} = get_nbelow_opencolsabove(astate.c2r, astate.colPrices, lambda)

"""
    findrowmax(rewardMatrix::Array{T, 2}) where {T <: Real} -> rowMaximum, rowMaximumIdx
    findrowmax(rewardMatrix::SparseMatrixCSC{T, G}) where {G <: Integer, T <: Real} -> rowMaximum, rowMaximumIdx

Return vector of row maximums and corresponding column number, similar to `findmax(rewardMatrix, dims=1)`
but with only the columns recorded. For sparse matrix if no entry is observed in a row the
value is set to `-T(Inf)` and the index is set to zero.
"""
function findrowmax(rewardMatrix::Array{T, 2}) where {T <: Real}
    rowMaximum = rewardMatrix[:, 1]
    rowMaximumIdx = ones(Int, size(rewardMatrix, 1))
    for jj in 2:size(rewardMatrix, 2)
        for ii in 1:size(rewardMatrix, 1)
            if rewardMatrix[ii, jj] > rowMaximum[ii]
                rowMaximum[ii] = rewardMatrix[ii, jj]
                rowMaximumIdx[ii] = jj
            end
        end
    end
    return rowMaximum, rowMaximumIdx
end

function findrowmax(rewardMatrix::SparseMatrixCSC{T, G}) where {G <: Integer, T <: Real}
    
    rowMaximum = fill(T(-Inf), rewardMatrix.m)
    rowMaximumIdx = zeros(G, rewardMatrix.m)

    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)
    
    for jj in 1:rewardMatrix.n
        rng = nzrange(rewardMatrix, jj)
        for idx in rng
            ii = rows[idx]
            val = rewards[idx]
            if val > rowMaximum[ii]
                rowMaximum[ii] = val
                rowMaximumIdx[ii] = jj
            end
        end
    end
    return rowMaximum, rowMaximumIdx
end

"""
    findrowmin(rewardMatrix::Array{T, 2}) where {T <: Real} -> rowMinimum, rowMinimumIdx
    findrowmin(rewardMatrix::SparseMatrixCSC{T, G}) where {G <: Integer, T <: Real} -> rowMinimum, rowMinimumIdx

Return vector of row minimums and corresponding column number, similar to `findmin(rewardMatrix, dims=1)`
but with only the columns recorded. For sparse matrix if no entry is observed in a row the
value is set to `T(Inf)` and the index is set to zero.
"""
function findrowmin(rewardMatrix::Array{T, 2}) where {T <: Real}
    rowMinimum = rewardMatrix[:, 1]
    rowMinimumIdx = ones(Int, size(rewardMatrix, 1))
    for jj in 1:size(rewardMatrix, 2)
        for ii in 1:size(rewardMatrix, 1)
            if rewardMatrix[ii, jj] < rowMinimum[ii]
                rowMinimum[ii] = rewardMatrix[ii, jj]
                rowMinimumIdx[ii] = jj
            end
        end
    end
    return rowMinimum, rowMinimumIdx
end

function findrowmin(rewardMatrix::SparseMatrixCSC{T, G}) where {G <: Integer, T <: Real}
    
    rowMinimums = fill(G(Inf), rewardMatrix.m)
    rowMinimumIdx = zeros(G, rewardMatrix.m)

    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)
    
    for jj in 1:rewardMatrix.n
        rng = nzrange(rewardMatrix, jj)
        for idx in rng
            ii = rows[idx]
            val = rewards[idx]
            if val > rowMinimums[ii]
                rowMinimums[ii] = val
                rowMinimumIdx[ii] = jj
            end
        end
    end
    return rowMinimum, rowMinimumIdx
end


"""
    findcolmax(rewardMatrix::Array{T, 2}) where {T <: Real} -> colMaximum, colMaximumIdx
    findcolmax(rewardMatrix::SparseMatrixCSC{T, G}) where {G <: Integer, T <: Real} -> colMaximum, colMaximumIdx

Return vector of column maximums and corresponding row number, similar to `findmax(rewardMatrix, dims=2)`
but with only the rowss recorded.  For sparse matrix if no entry is observed in a column both the
value and the index are set to zero.
"""
function findcolmax(rewardMatrix::Array{T, 2}) where {T <: Real}
    colMaximum = rewardMatrix[1, :]
    colMaximumIdx = ones(Int, size(rewardMatrix, 2))    
    
    for jj in 1:size(rewardMatrix, 2)
        for ii in 2:size(rewardMatrix, 1)
            if rewardMatrix[ii, jj] > colMaximum[ii]
                colMaximum[ii] = rewardMatrix[ii, jj]
                colMaximumIdx[ii] = jj
            end
        end
    end
    return colMaximum, colMaximumIdx
end

function findcolmax(rewardMatrix::SparseMatrixCSC{T, G}) where {G <: Integer, T <: Real}
    
    colMaximum = zeros(T, rewardMatrix.n)
    colMaximumIdx = zeros(G, rewardMatrix.n)

    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)
    
    for jj in 1:rewardMatrix.n
        rng = nzrange(rewardMatrix, jj)
        if length(rng) > 0
            val, idx = findmax(rewards[rng])
            colMaximum = val
            colMaximumIdx = rows[idx]
        end
    end
    return colMaximum, colMaximumIdx
end

"""
    findcolmin(rewardMatrix::Array{T, 2}) where {T <: Real} -> colMinimum, colMinimumIdx
    findcolmin(rewardMatrix::SparseMatrixCSC{T, G}) where {G <: Integer, T <: Real} -> colMinimum, colMinimumIdx

Return vector of column minimums and corresponding row ids, similar to `findmin(rewardMatrix, dims=2)`
but with only the rowss recorded.  For sparse matrix if no entry is observed in a column both the
value and the index are set to zero.
"""
function findcolmin(rewardMatrix::Array{T, 2}) where {T <: Real}
    colMinimum = rewardMatrix[1, :]
    colMinimumIdx = ones(Int, size(rewardMatrix, 2))    
    
    for jj in 1:size(rewardMatrix, 2)
        for ii in 2:size(rewardMatrix, 1)
            if rewardMatrix[ii, jj] < colMinimum[ii]
                colMinimum[ii] = rewardMatrix[ii, jj]
                colMinimumIdx[ii] = jj
            end
        end
    end
    return colMinimum, colMinimumIdx
end

function findcolmin(rewardMatrix::SparseMatrixCSC{T, G}) where {G <: Integer, T <: Real}
    
    colMinimum = zeros(T, rewardMatrix.n)
    colMinimumIdx = zeros(G, rewardMatrix.n)

    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)
    
    for jj in 1:rewardMatrix.n
        rng = nzrange(rewardMatrix, jj)
        if length(rng) > 0
            val, idx = findmin(rewards[rng])
            colMinimum = val
            colMinimumIdx = rows[idx]
        end
    end
    return colMinimum, colMinimumIdx
end

"""
    dimmaximums(rewardMatrix::Array{T, 2}) where {G <: Integer, T <: Real} -> rowMaximums, colMaximums

Simultaneously calculate the maximum over both the rows and the columns of a matrix
`x = rand(4, 5)`
`rowMax, colMax = dimmaximums(x)`
"""
function dimmaximums(rewardMatrix::Array{T, 2}) where {T <: Real}

    rowMaximums = zeros(T, size(rewardMatrix, 1))
    colMaximums = zeros(T, size(rewardMatrix, 2))

    for jj in 1:size(rewardMatrix, 2)
        for ii in 1:size(rewardMatrix, 1)
            val = rewardMatrix[ii, jj]
            if val > rowMaximums[ii]
                rowMaximums[ii] = val
            end

            if val > colMaximums[jj]
                colMaximums[jj] = val
            end
        end
    end
    return rowMaximums, colMaximums
end

function dimmaximums(rewardMatrix::SparseMatrixCSC{T, G}) where {G <: Integer, T <: Real}
    rowMaximums = zeros(T, rewardMatrix.m)
    colMaximums = zeros(T, rewardMatrix.n)

    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)

    for jj in 1:rewardMatrix.n
        rng = nzrange(rewardMatrix, jj)
        for idx in rng
            ii = rows[idx]
            val = rewards[idx]
            if val > rowMaximums[ii]
                rowMaximums[ii] = val
            end

            if val > colMaximums[jj]
                colMaximums[jj] = val
            end
        end
    end
    return rowMaximums, colMaximums
end

"""
    forward_rewardmatrix(rewardMatrix::SparseMatrixCSC) -> trewardMatrix
    forward_rewardmatrix(rewardMatrix::Matrix) -> rewardMatrix

Tranpose (using permutedims) sparse matrix and return non-transformed full matrix.
"""
function forward_rewardmatrix(rewardMatrix::SparseMatrixCSC)
    return permutedims(rewardMatrix)
end

function forward_rewardmatrix(rewardMatrix::Matrix)
    return rewardMatrix
end

"""
    scale_assignment!(astate::AssignmentState{G, T}, epsi::T, epsiscale::T) -> (astate, newepsi)

Shrink `epsi` by a factor of `epsiscale` for next iteration of auction assignment algorithm and remove previous assignemtns from  `astate`.

`newepsi = epsi * epsiscale`.  All assigments are removed using `clear_assignment!` so that complimentary slackness is maintained which requires that `rowPrice[row] + colprice[col] >= rewardMatrix[row, col] - epsilon`
"""
function scale_assignment!(astate::AssignmentState{G, T}, epsi::T, epsiscale::T) where {G <: Integer, T <: AbstractFloat}
    newepsi = epsi * epsiscale
    deltaepsi = epsi - newepsi
    for ii in one(G):astate.nrow
        if !isinf(astate.rowPrices[ii])
            astate.rowPrices[ii] += deltaepsi
            astate.c2r[astate.r2c[ii]] = zero(G)
            astate.r2c[ii] = zero(G)
            astate.nassigned -= one(G)
        end
    end
    return astate, newepsi
end

"""
    min_assigned_colprice(astate::AssignmentState{G, T}) where {G <: Integer, T <: Real} -> lambda

Return the minimum `astate.colPrices` for which `astate.c2r` is nonzero.  Returns zero if `astate.nassigned == 0`
"""
function min_assigned_colprice(astate::AssignmentState{G, T}) where {G <: Integer, T <: Real}

    if iszero(astate.nassigned)
        return zero(T)
    end

    idx = findfirst(c -> !iszero(c), astate.r2c)
    lambda = astate.colPrices[astate.r2c[idx]]
    if astate.nassigned > one(G)
        for ii in (G(idx) + one(G)):astate.nrow
            if !iszero(astate.r2c[ii]) && astate.colPrices[astate.r2c[ii]] < lambda
                lambda = astate.colPrices[astate.r2c[ii]]
            end
        end
    end
    return lambda
end

"""
    check_epsilons(epsi0::T, epsitol::T, epsiscale::T) where T <: AbstractFloat -> nothing

Check that a workable set of epsilon values have been provided.
"""
function check_epsilons(epsi0::T, epsitol::T, epsiscale::T) where T <: AbstractFloat
    if epsiscale >= one(T)
        error("epsiscale set to $epsiscale, must be < 1.0 or tolerance will not shrink")
    elseif epsiscale <= zero(T)
        error("epsiscale set to $epsiscale, must be > 0.0 or tolerance will not stay positive")
    elseif epsitol > epsi0
        @warn "epsitol > epsi0, no assignment will be performed"
    elseif epsi0 < zero(T)
        "epsi0 < 0.0, epsi0 must be set to a positive value"
    end
    return nothing
end

"""
    check_epsilon_slackness(astate::AssignmentState{G, T}, rewardMatrix::Array{T, 2}, epsilon::T) where {G <: Integer, T <: AbstractFloat} -> (consistent::Bool, outrows::Array{Int, 1}, outcols::Array{Int, 1})
    check_epsilon_slackness(astate::AssignmentState{G, T}, rewardMatrix::SparseMatrixCSC{T, G}, epsilon::T) where {G <: Integer, T <: AbstractFloat} -> (consistent::Bool, outrows::Array{Int, 1}, outcols::Array{Int, 1})

Check epsilon-slackness conditions of rewardMatrix[ii, jj] - epsilon <= astate.rowPrices[ii] + astate.colPrices[jj].  If `consistent` is false then `outrows` and `outcols` will contain the entries for which epsilon-slackness is violated.
"""
function check_epsilon_slackness(astate::AssignmentState{G, T}, rewardMatrix::Array{T, 2}, epsilon::T) where {G <: Integer, T <: AbstractFloat}
    consistent = true
    outrows = Int[]
    outcols = Int[]
    for jj in 1:size(rewardMatrix, 2), ii in 1:size(rewardMatrix, 1)
        if (rewardMatrix[ii, jj] - epsilon) > (astate.rowPrices[ii] + astate.colPrices[jj])
            push!(outrows, ii)
            push!(outcols, jj)
            if consistent
                consistent = false
            end
        end
    end
    return consistent, outrows, outcols
end

function check_epsilon_slackness(astate::AssignmentState{G, T}, rewardMatrix::SparseMatrixCSC{T, G}, epsilon::T) where {G <: Integer, T <: AbstractFloat}
    consistent = true
    outrows = Int[]
    outcols = Int[]

    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)
    
    for jj in 1:size(rewardMatrix, 2)
        rng = nzrange(rewardMatrix, jj)
        for idx in rng[3:end]
            ii = rows[idx]
            if (rewards[idx] - epsilon) > (astate.rowPrices[ii] + astate.colPrices[jj])
                push!(outrows, ii)
                push!(outcols, jj)
                if consistent
                    consistent = false
                end
            end
        end
    end
    return consistent, outrows, outcols
end

"""
    tuple_r2c(r2c::Array{G, 1})
    tuple_r2c(astate::AssignmentState)

Construct a tuple (rows, cols) of the row-column assignments.
"""
function tuple_r2c(r2c::Array{G, 1})
    rows = G[]
    cols = G[]
    for (ii, jj) in pairs(r2c)
        if !izero(jj)
            push!(rows, ii)
            push!(cols, jj)
        end
    end
    return rows, cols
end

tuple_r2c(astate::AssignmentState) = tuple_r2c(astate.r2c)

