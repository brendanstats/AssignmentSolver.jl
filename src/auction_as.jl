########################################
#Forward Bids - Identical for Symmetric / Asymmetric
########################################

"""
    forward_bid(row::G, colPrices::Array{T, 1}, epsi::T, rewardMatrix::Array{T, 2}[, dfltTwo::T]) where {G <: Integer, T <: AbstractFloat} -> (maxcol, bid, twoMinusEpsilon)
    forward_bid(row::G, colPrices::Array{T, 1}, epsi::T, trewardMatrix::SparseMatrixCSC{T, 2}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (maxcol, bid, twoMinusEpsilon)
    forward_bid(row::G, colPrices::Array{T, 1}, epsi::T, trewardMatrix::SparseMatrixCSC{T, 2}, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (maxcol, bid, twoMinusEpsilon)

Compute forward bid for `row` in an auction algorithm forwrd iteration.

Find the maximal column assignment for `row` using `maxtwo_row` and then return the `maxcol`,
`bid`, and `twoMinusEpsilon`.  Where `bid = `rewardMatrix[row, maxcol] - twoMinusEpsilon` and
`twoMinusEpsilon = maxtwo - epsi` for `maxtwo` returned by `maxtwo_row`.

# Arguments

* `row::G`: Row for which the forward bid is computed.
* `colPrices::Array{T, 1}`: column (object) prices in auction algorithm.
* `epsi::T`: tolerance associated with auction algorithm iteration.
* `rewardMatrix::Array{T, 2}`: reward matrix for which a maximal assignment is being found.
* `trewardMatrix::SparseMatrixCSC{T, 2}`: transpose of reward matrix for which a maximal
assignment is being found.
* `dfltTwo::T`: default second largest value using if only one column has an entry containing
`row` if the rewardMatrix is sparse.  Ignored if the reward matrix is not sparse.
* `dfltReward::T`: Default reward value used for implicity added entries if a padded reward
matrix is being used.

See also: [`maxtwo_row`](@ref), [`maxtwo_column`](@ref), [`reverse_bid`](@ref), [`forward_update`](@ref), [`forward_iteration!`](@ref)
"""
function forward_bid(row::G,
                     colPrices::Array{T, 1},
                     epsi::T,
                     rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}
    
    ##Find maximum reward - object price
    maxcol, maxval, maxtwo = maxtwo_row(row, colPrices, rewardMatrix)

    ##Compute useful values
    twoMinusEpsilon = maxtwo - epsi
    bid = rewardMatrix[row, maxcol] - twoMinusEpsilon
    
    return maxcol, bid, twoMinusEpsilon
end

forward_bid(row::G, colPrices::Array{T, 1}, epsi::T, rewardMatrix::Array{T, 2}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} = forward_bid(row, colPrices, epsi, rewardMatrix)

function forward_bid(row::G, colPrices::Array{T, 1}, epsi::T,
                     trewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Find maximum reward - object price
    maxidx, maxval, maxtwo = maxtwo_row(row, colPrices, trewardMatrix, dfltTwo)

    ##Compute useful values
    maxcol = trewardMatrix.rowval[maxidx]
    twoMinusEpsilon = maxtwo - epsi
    bid = trewardMatrix.nzval[maxidx] - twoMinusEpsilon
    
    return maxcol, bid, twoMinusEpsilon
end

function forward_bid(row::G,
                     colPrices::Array{T, 1},
                     epsi::T,
                     trewardMatrix::SparseMatrixCSC{T, G},
                     dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Find maximum reward - object price
    maxidx, maxval, maxtwo = maxtwo_row(row, colPrices, trewardMatrix, dfltReward, dfltTwo)
    
    ##Compute useful values
    twoMinusEpsilon = maxtwo - epsi
    if maxidx > zero(G)
        maxcol = trewardMatrix.rowval[maxidx]
        bid = trewardMatrix.nzval[maxidx] - twoMinusEpsilon
    else
        maxcol = row + trewardMatrix.m
        bid = dfltReward - twoMinusEpsilon
    end
    
    return maxcol, bid, twoMinusEpsilon
end

########################################
#Reverse Bids - Symmetric / Asymmetric; Sparse / Dense Matrix
########################################

"""
    reverse_bid(col::G, rowPrices::Array{T, 1}, epsi::T, rewardMatrix::Array{T, 2}[, dfltTwo::T]) where {G <: Integer, T <: AbstractFloat} -> (maxrow, bid, twoMinusEpsilon)
    reverse_bid(col::G, rowPrices::Array{T, 1}, epsi::T, rewardMatrix::SparseMatrixCSC{T, 2}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (maxrow, bid, twoMinusEpsilon)
    reverse_bid(col::G, rowPrices::Array{T, 1}, lambda::T, epsi::T, rewardMatrix::Array{T, 2}[, dfltTwo::T]) where {G <: Integer, T <: AbstractFloat} -> (maxrow, bid, twoMinusEpsilon)
    reverse_bid(col::G, rowPrices::Array{T, 1}, lambda::T, epsi::T, rewardMatrix::SparseMatrixCSC{T, 2}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (maxrow, bid, twoMinusEpsilon)
    reverse_bid(col::G, rowPrices::Array{T, 1}, lambda::T], epsi::T, rewardMatrix::SparseMatrixCSC{T, 2}, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (maxrow, bid, twoMinusEpsilon)

Compute reverse bid for `col` in an auction algorithm reverse iteration.

Find the maximal row assignment for `col` using `maxtwo_column` and then return the `maxrow`,
`bid`, and `twoMinusEpsilon`.  Where `bid = `rewardMatrix[maxrow, col] - twoMinusEpsilon` and
`twoMinusEpsilon = maxtwo - epsi` for `maxtwo` returned by `maxtwo_column`.  If a `lambda`
parameter is included then the problem is assumed to be asymmetric, containing more objects
than people (the reward matrix contains more columns than rows). In the case where a
`dfltReward` parameter is set a padded reward matrix is assumed which will always contain
more rows and than columns.

# Arguments

* `col::G`: Row for which the forward bid is computed.
* `rowPrices::Array{T, 1}`: column (individual) prices in auction algorithm.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.
* `epsi::T`: tolerance associated with auction algorithm iteration.
* `rewardMatrix::Union{Array{T, 2}, SparseMatrixCSC{T, 2}`: reward matrix for which a maximal
 assignment is being found.
* `dfltTwo::T`: default second largest value using if only one row has an entry containing
`column` if `rewardMatrix` is sparse.  Ignored if `rewardMatrix` is not sparse.
* `dfltReward::T`: Default reward value used for implicity added entries if a padded reward
matrix is being used.

See also: [`maxtwo_row`](@ref), [`maxtwo_column`](@ref), [`forward_bid`](@ref), [`reverse_update!`](@ref), [`reverse_iteration!`](@ref)
"""
function reverse_bid(col::G,
                     rowPrices::Array{T, 1},
                     epsi::T,
                     rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}
    
    ##Find maximum reward - object price
    maxrow, maxval, maxtwo = maxtwo_column(col, rowPrices, rewardMatrix)
    
    ##Compute useful values
    twoMinusEpsilon = maxtwo - epsi
    bid = rewardMatrix[maxrow, col] - twoMinusEpsilon
    
    return maxrow, bid, twoMinusEpsilon
end

reverse_bid(col::G, rowPrices::Array{T, 1}, epsi::T, rewardMatrix::Array{T, 2}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} = reverse_bid(col, rowPrices, epsi, rewardMatrix)

#Symmetric reverse bid, no lambda
function reverse_bid(col::G,
                     rowPrices::Array{T, 1},
                     epsi::T,
                     rewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Find maximum reward - object price
    maxidx, maxval, maxtwo = maxtwo_column(col, rowPrices, rewardMatrix, dfltTwo)
    
    ##Compute useful values
    maxrow = rewardMatrix.rowval[maxidx]
    twoMinusEpsilon = maxtwo - epsi
    bid = rewardMatrix.nzval[maxidx] - twoMinusEpsilon
    
    return maxrow, bid, twoMinusEpsilon
end

#Asymmetric reverse bid, lambda
function reverse_bid(col::G,
                     rowPrices::Array{T, 1},
                     lambda::T, epsi::T,
                     rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}
    
    ##Find maximum reward - object price
    maxrow, maxval, maxtwo = maxtwo_column(col, rowPrices, rewardMatrix)

    ##Compute useful values
    maxMinusEpsilon = maxval - epsi
    if maxMinusEpsilon >= lambda
        twoMinusEpsilon = maxtwo - epsi
        if twoMinusEpsilon >= lambda 
            bid = rewardMatrix[maxrow, col] - twoMinusEpsilon
            return maxrow, bid, twoMinusEpsilon
        else
            bid = rewardMatrix[maxrow, col] - lambda
            return maxrow, bid, lambda
        end
    else
        bid = zero(T)
        return maxrow, bid, maxMinusEpsilon
    end
end

function reverse_bid(col::G,
                     rowPrices::Array{T, 1},
                     lambda::T, epsi::T,
                     rewardMatrix::Array{T, 2}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    return reverse_bid(col, rowPrices, lambda, epsi, rewardMatrix)
end

function reverse_bid(col::G,
                     rowPrices::Array{T, 1},
                     lambda::T, epsi::T,
                     rewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Find maximum reward - object price
    maxidx, maxval, maxtwo = maxtwo_column(col, rowPrices, rewardMatrix, dfltTwo)

    ##Compute useful values
    maxrow = rewardMatrix.rowval[maxidx]
    maxMinusEpsilon = maxval - epsi
    if maxMinusEpsilon >= lambda
        twoMinusEpsilon = maxtwo - epsi
        if twoMinusEpsilon >= lambda 
            bid = rewardMatrix.nzval[maxidx] - twoMinusEpsilon
            return maxrow, bid, twoMinusEpsilon
        else
            bid = rewardMatrix.nzval[maxidx] - lambda
            return maxrow, bid, lambda
        end
    else
        bid = zero(T)
        return maxrow, bid, maxMinusEpsilon
    end
end

function reverse_bid(col::G,
                     rowPrices::Array{T, 1},
                     lambda::T, epsi::T,
                     rewardMatrix::SparseMatrixCSC{T, G},
                     dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Find maximum reward - object price
    if col <= rewardMatrix.n
        return reverse_bid(col, rowPrices, lambda, epsi, rewardMatrix, dfltTwo)
    else
        ##Compute useful values
        maxrow = col - rewardMatrix.n
        maxtwo = dfltTwo
        maxMinusEpsilon = dfltReward - rowPrices[maxrow] - epsi    
        if maxMinusEpsilon >= lambda
            twoMinusEpsilon = maxtwo - epsi
            if twoMinusEpsilon >= lambda 
                bid = dfltReward - twoMinusEpsilon
                return maxrow, bid, twoMinusEpsilon
            else
                bid = dfltReward - lambda
                return maxrow, bid, lambda
            end
        else
            bid = zero(T)
            return maxrow, bid, maxMinusEpsilon
        end
    end
end

########################################
#Perform Forward Updates - Identical for sparse / dense
########################################

"""
    forward_update!(row::G, maxcol::G, bid::T, newRowPrice::T, openRows::Queue{G}, astate::AssignmentState{G, T}) where {G <: Integer, T <: AbstractFloat} -> (addassign, openRows, astate)
    forward_update!(row::G, maxcol::G, bid::T, newRowPrice::T, openRows::Queue{G}, astate::AssignmentState{G, T}, lambda::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, openRows, astate)

Perform forward update of assignment and prices for `row` in an auction algorithm forwrd iteration. 

Updates the assignement and prices in `astate`.  If the reward matrix is asymmetric then
`row` is always assigned to `maxcol` and a `lambda` parameter is not used.  For an
asymmetric reward matrix the assignment is only made if `bid >= lambda`.  If `maxcol`
is already assigned then the previous assignment is removed. `addassign::Bool` indicates if the
total number of assignments has increased.

# Arguments

* `row::G`: Row for which the forward update is performed.
* `maxcol::G`: Column (object) to which `row` may be assigned.
* `bid::T`: New price for `maxcol`.
* `newRowPrice::T`: New price for `row`.
* `openRows::Queue{G}`: Queue containing unassigned rows.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.

See also: [`forward_update_nbelow!`](@ref), [`forward_bid`](@ref), [`reverse_update!`](@ref), [`forward_iteration!`](@ref), [`AssignmentState`](@ref)
"""
function forward_update!(row::G, maxcol::G, bid::T, newRowPrice::T,
                         openRows::Queue{G}, astate::AssignmentState{G, T}) where {G <: Integer, T <: AbstractFloat}

    ##Set new row and column costs
    astate.rowPrices[row] = newRowPrice
    astate.colPrices[maxcol] = bid
    
    ##Check if column is already assigned, unassign if necessary
    if iszero(astate.c2r[maxcol])
        addassign = true
        astate.nassigned += one(G)
    else
        prevrow = astate.c2r[maxcol]
        astate.r2c[prevrow] = zero(G)
        
        ##add row to queue of unassigned rows
        enqueue!(openRows, prevrow)
        addassign = false
    end

    ##Assign row to maxcol
    astate.r2c[row] = maxcol
    astate.c2r[maxcol] = row
    
    return addassign, openRows, astate
end

function forward_update!(row::G, maxcol::G, bid::T, newRowPrice::T,
                         openRows::Queue{G}, astate::AssignmentState{G, T},
                         lambda::T) where {G <: Integer, T <: AbstractFloat}
    if bid >= lambda
        astate.colPrices[maxcol] = bid

        ##Check if column is already assigned, unassign if necessary
        if iszero(astate.c2r[maxcol])
            addassign = true
            astate.nassigned += one(G)
        else
            prevrow = astate.c2r[maxcol]
            astate.r2c[prevrow] = zero(G)
            
            ##add row to queue of unassigned rows
            enqueue!(openRows, prevrow)
            addassign = false
        end

        ##Assign row to maxcol
        astate.r2c[row] = maxcol
        astate.c2r[maxcol] = row
    else
        astate.colPrices[maxcol] = lambda
        enqueue!(openRows, row)
        addassign = false
    end

    ##Set column cost
    astate.rowPrices[row] = newRowPrice

    return addassign, openRows, astate
end

"""
    forward_update_nbelow!(row::G, maxcol::G, bid::T, newRowPrice::T, nbelow::G, openRows::Queue{G}, astate::AssignmentState{G, T}, lambda::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, nbelow, openRows, astate)

Perform forward update of assignment and prices for `row` in an auction algorithm forwrd iteration, identical to `forward_update!` but tracks the number of column prices below `lambda`.

Updates the assignement and prices in `astate`.  If the reward matrix is asymmetric then
`row` is always assigned to `maxcol` and a `lambda` parameter is not used.  For an
asymmetric reward matrix the assignment is only made if `bid >= lambda`.  If `maxcol`
is already assigned then the previous assignment is removed. `addassign::Bool` indicates if the
 total number of assignments has increased.

# Arguments

* `row::G`: Row for which the forward update is performed.
* `maxcol::G`: Column (object) to which `row` may be assigned.
* `bid::T`: New price for `maxcol`.
* `newRowPrice::T`: New price for `row`.
* `nbelow::G`: number of column prices (`astate.colPrices`) strictly less than `lambda`.
* `openRows::Queue{G}`: Queue containing unassigned rows.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.

See also: [`forward_update!`](@ref), [`forward_bid`](@ref), [`reverse_update!`](@ref), [`forward_iteration!`](@ref), [`AssignmentState`](@ref)
"""
function forward_update_nbelow!(row::G, maxcol::G, bid::T, newRowPrice::T,
                         nbelow::G, openRows::Queue{G}, astate::AssignmentState{G, T},
                         lambda::T) where {G <: Integer, T <: AbstractFloat}
    if bid >= lambda
        astate.colPrices[maxcol] = bid

        ##Check if column is already assigned, unassign if necessary
        if iszero(astate.c2r[maxcol])
            addassign = true
            astate.nassigned += one(G)
        else
            prevrow = astate.c2r[maxcol]
            astate.r2c[prevrow] = zero(G)
            
            ##add row to queue of unassigned rows
            enqueue!(openRows, prevrow)
            addassign = false
        end

        ##Assign row to maxcol
        astate.r2c[row] = maxcol
        astate.c2r[maxcol] = row
    else
        if astate.colPrices[maxcol] < lambda
            nbelow -= one(G)
        end
        astate.colPrices[maxcol] = lambda
        enqueue!(openRows, row)
        addassign = false
    end

    ##Set column cost
    astate.rowPrices[row] = newRowPrice

    return addassign, nbelow, openRows, astate
end

########################################
#Perform Reverse Updates
########################################

"""
    reverse_update!(maxrow::G, col::G, bid::T, newColPrice::T, openCols::Queue{G}, astate::AssignmentState{G, T}) where {G <: Integer, T <: AbstractFloat} -> (addassign, openCols, astate)
    reverse_update!(maxrow::G, col::G, bid::T, newColPrice::T, openColsAbove::Queue{G}, astate::AssignmentState{G, T}, lambda::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, openColsAbove, astate)

Perform reverse update of assignment and prices for `row` in an auction algorithm forwrd iteration. 

Updates the assignement and prices in `astate`.  If the reward matrix is asymmetric then
`col` is always assigned to `maxrow` and a `lambda` parameter is not used.  For an
asymmetric reward matrix the assignment is only made if `newColPrice` >= `lambda`.  If `maxrow`
is already assigned then the previous assignment is removed.  `addassign::Bool` indicates if the
 total number of assignments has increased.

# Arguments

* `maxrow::G`: Row (individual) to which `col` may be assigned.
* `col::G`: Column for which the reverse update is performed.
* `bid::T`: New price for `maxrow`.
* `newColPrice::T`: New price for `col`.
* `openCols::Queue{G}`: Queue containing unassigned columns.
* `openColsAbove::Queue{G}`: Queue containing unassigned columns where `astate.colPrices` > `lambda`.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.

See also: [`reverse_update_nbelow!`](@ref), [`reverse_bid`](@ref), [`forward_update!`](@ref), [`reverse_iteration!`](@ref), [`AssignmentState`](@ref)
"""
function reverse_update!(maxrow::G, col::G, bid::T, newColPrice::T,
                         openCols::Queue{G}, astate::AssignmentState{G, T}) where {G <: Integer, T <: AbstractFloat}

    ##Set new row and column costs
    astate.rowPrices[maxrow] = bid
    astate.colPrices[col] = newColPrice
    
    ##If maxrow already assigned, unassign
    if iszero(astate.r2c[maxrow])
        addassign = true
        astate.nassigned += one(G)
    else
        prevcol = astate.r2c[maxrow]
        astate.c2r[prevcol] = zero(G)

        ##add column to queue of unassigned columns
        enqueue!(openCols, prevcol)
        addassign = false
    end
    
    ##Add assignment
    astate.r2c[maxrow] = col
    astate.c2r[col] = maxrow
    
    return addassign, openCols, astate
end

#update for asymmetric problem, lambda parameter
function reverse_update!(maxrow::G, col::G, bid::T, newColPrice::T,
                         openColsAbove::Queue{G}, astate::AssignmentState{G, T},
                         lambda::T) where {G <: Integer, T <: AbstractFloat}
    
    if newColPrice >= lambda

        ##Set new row and column costs
        astate.rowPrices[maxrow] = bid
        astate.colPrices[col] = newColPrice

        ##If maxrow already assigned, unassign
        if iszero(astate.r2c[maxrow])
            addassign = true
            astate.nassigned += one(G)
        else
            prevcol = astate.r2c[maxrow]
            astate.c2r[prevcol] = zero(G)

            ##add back to queue of open columns if price is sufficiently high
            if astate.colPrices[prevcol] > lambda
                enqueue!(openColsAbove, prevcol)
            end
            addassign = false
        end
        
        ##Add assignment
        astate.c2r[col] = maxrow
        astate.r2c[maxrow] = col
    else
        astate.colPrices[col] = newColPrice #will be <= lambda so not added to queue
        addassign = false
    end

    return addassign, openColsAbove, astate
end

"""
    reverse_update_nbelow!(maxrow::G, col::G, bid::T, newColPrice::T, openColsAbove::Queue{G}, astate::AssignmentState{G, T}, lambda::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, nbelow, openColsAbove, astate)

Perform reverse update of assignment and prices for `row` in an auction algorithm forwrd iteration, identical to `reverse_update!` but tracks the number of column prices below `lambda`.

Updates the assignement and prices in `astate`.  If the reward matrix is asymmetric then
`col` is always assigned to `maxrow` and a `lambda` parameter is not used.  For an
asymmetric reward matrix the assignment is only made if `newColPrice` >= `lambda`.  If `maxrow`
is already assigned then the previous assignment is removed.  `addassign::Bool` indicates if the
 total number of assignments has increased.

# Arguments

* `maxrow::G`: Row (individual) to which `col` may be assigned.
* `col::G`: Column for which the reverse update is performed.
* `bid::T`: New price for `maxrow`.
* `newColPrice::T`: New price for `col`.
* `nbelow::G`: number of column prices (`astate.colPrices`) < `lambda`.
* `openCols::Queue{G}`: Queue containing unassigned columns.
* `openColsAbove::Queue{G}`: Queue containing unassigned columns where `astate.colPrices` > `lambda`.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.

See also: [`reverse_update!`](@ref), [`reverse_bid`](@ref), [`forward_update!`](@ref), [`reverse_iteration!`](@ref), [`AssignmentState`](@ref)
"""
function reverse_update_nbelow!(maxrow::G, col::G, bid::T, newColPrice::T,
                                nbelow::G, openColsAbove::Queue{G}, astate::AssignmentState{G, T},
                                lambda::T) where {G <: Integer, T <: AbstractFloat}
    
    if newColPrice >= lambda

        ##Set new row and column costs
        astate.rowPrices[maxrow] = bid
        astate.colPrices[col] = newColPrice

        ##If maxrow already assigned, unassign
        if iszero(astate.r2c[maxrow])
            addassign = true
            astate.nassigned += one(G)
        else
            prevcol = astate.r2c[maxrow]
            astate.c2r[prevcol] = zero(G)

            ##add back to queue of open columns if price is sufficiently high
            if astate.colPrices[prevcol] > lambda
                enqueue!(openColsAbove, prevcol)
            end
            addassign = false
        end
        
        ##Add assignment
        astate.c2r[col] = maxrow
        astate.r2c[maxrow] = col
    else
        astate.colPrices[col] = newColPrice #will be <= lambda so not added to queue
        if astate.colPrices[col] < lambda
            nbelow += one(G)
        end
        addassign = false
    end

    return addassign, nbelow, openColsAbove, astate
end

########################################
#Forward Iterations
########################################

"""
    forward_iteration!(openRows::Queue{G}, astate::AssignmentState{G, T}, epsi::T, frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, openRows, astate)
    forward_iteration!(openRows::Queue{G}, astate::AssignmentState{G, T}, lambda::T, epsi::T, frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, openRows, astate)
    forward_iteration!(openRows::Queue{G}, astate::AssignmentState{G, T}, lambda::T, epsi::T, frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, openRows, astate)

Find an unassigned row, compute the `forward_bid` and then perform the `foward_update`.

If the reward matrix is asymmetric then a `lambda` parameter is required. If the reward
matrix is sparse and extra entries are to be assumed (to ensure a feasible assignment)
then a `dfltReward` must be supplied. `addassign::Bool` indicates if the
 total number of assignments has increased.

# Arguments

* `openRows::Queue{G}`: Queue containing unassigned rows.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.
* `epsi::T`: tolerance associated with auction algorithm iteration.
* `frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: forward reward matrix.  If the
reward matrix is space then this is the transpose of the reward matrix.  Otherwise it is just
the reward matrix.
* `dfltTwo::T`: default second largest value passed to `forward_bid`.
`row` if the rewardMatrix is sparse.  Ignored if the reward matrix is not sparse.
* `dfltReward::T`: Default reward value used for implicity added entries if a padded reward
matrix is being used, passed to `forward_bid`.

See also: [`forward_iteration_nbelow!`](@ref), [`forward_bid`](@ref), [`forward_update!`](@ref), [`forward_update_nbelow!`](@ref), [`reverse_iteration!`](@ref), [`AssignmentState`](@ref)
"""
function forward_iteration!(openRows::Queue{G}, astate::AssignmentState{G, T},
                            epsi::T,
                            frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                            dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned row to attempt to match
    row = dequeue!(openRows)

    ##Row may have been assigned in a reverse iteration, assumes another free row since otherwise all rows are already assigned...
    if !iszero(astate.r2c[row])
        return forward_iteration!(openRows, astate, epsi, frewardMatrix, dfltTwo)
    end

    maxcol, bid, newRowPrice = forward_bid(row, astate.colPrices, epsi, frewardMatrix, dfltTwo)
    addassign, openRows, astate = forward_update!(row, maxcol, bid, newRowPrice, openRows, astate)

    return addassign, openRows, astate
end

#asymmetric, includes lambda
function forward_iteration!(openRows::Queue{G}, astate::AssignmentState{G, T},
                            lambda::T, epsi::T,
                            frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                            dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned row to attempt to match
    row = dequeue!(openRows)

    ##Row may have been assigned in a reverse iteration, assumes another free row since otherwise all rows are already assigned...
    if !iszero(astate.r2c[row])
        return forward_iteration!(openRows, astate, lambda, epsi, frewardMatrix, dfltTwo)
    end

    maxcol, bid, newRowPrice = forward_bid(row, astate.colPrices, epsi, frewardMatrix, dfltTwo)
    addassign, openRows, astate = forward_update!(row, maxcol, bid, newRowPrice, openRows, astate, lambda)

    return addassign, openRows, astate
end

#padded
function forward_iteration!(openRows::Queue{G}, astate::AssignmentState{G, T},
                            lambda::T, epsi::T,
                            frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                            dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned row to attempt to match
    row = dequeue!(openRows)

    ##Row may have been assigned in a reverse iteration, assumes another free row since otherwise all rows are already assigned...
    if !iszero(astate.r2c[row])
        return forward_iteration!(openRows, astate, lambda, epsi, frewardMatrix, dfltReward, dfltTwo)
    end

    maxcol, bid, newRowPrice = forward_bid(row, astate.colPrices, epsi, frewardMatrix, dfltReward, dfltTwo)
    addassign, openRows, astate = forward_update!(row, maxcol, bid, newRowPrice, openRows, astate, lambda)

    return addassign, openRows, astate
end

"""
    forward_iteration_nbelow!(nbelow::G, openRows::Queue{G}, astate::AssignmentState{G, T}, lambda::T, epsi::T, frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, nbelow, openRows, astate)
    forward_iteration_nbelow!(nbelow::G, openRows::Queue{G}, astate::AssignmentState{G, T}, lambda::T, epsi::T, frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, nbelow, openRows, astate)

Find an unassigned row, compute the `forward_bid` and then perform the `foward_update_nbelow`, identical to `reverse_update!` but tracks the number of column prices below `lambda`.

If the reward matrix is asymmetric then a `lambda` parameter is required. If the reward
matrix is sparse and extra entries are to be assumed (to ensure a feasible assignment)
then a `dfltReward` must be supplied.  `addassign::Bool` indicates if the
 total number of assignments has increased.

# Arguments

* `nbelow::G`: number of column prices (`astate.colPrices`) < `lambda`.
* `openRows::Queue{G}`: Queue containing unassigned rows.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.
* `epsi::T`: tolerance associated with auction algorithm iteration.
* `frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: forward reward matrix.  If the
reward matrix is space then this is the transpose of the reward matrix.  Otherwise it is just
the reward matrix.
* `dfltTwo::T`: default second largest value passed to `forward_bid`.
`row` if the rewardMatrix is sparse.  Ignored if the reward matrix is not sparse.
* `dfltReward::T`: Default reward value used for implicity added entries if a padded reward
matrix is being used, passed to `forward_bid`.

See also: [`forward_iteration!`](@ref), [`forward_bid`](@ref), [`forward_update!`](@ref), [`forward_update_nbelow!`](@ref), [`reverse_iteration!`](@ref), [`AssignmentState`](@ref)
"""
function forward_iteration_nbelow!(nbelow::G, openRows::Queue{G}, astate::AssignmentState{G, T},
                                   lambda::T, epsi::T,
                                   frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                                   dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned row to attempt to match
    row = dequeue!(openRows)

    ##Row may have been assigned in a reverse iteration, assumes another free row since otherwise all rows are already assigned...
    if !iszero(astate.r2c[row])
        return forward_iteration_nbelow!(nbelow, openRows, astate, lambda, epsi, frewardMatrix, dfltTwo)
    end

    maxcol, bid, newRowPrice = forward_bid(row, astate.colPrices, epsi, frewardMatrix, dfltTwo)
    addassign, nbelow, openRows, astate = forward_update_nbelow!(row, maxcol, bid, newRowPrice, nbelow, openRows, astate, lambda)

    return addassign, nbelow, openRows, astate
end

#padded
function forward_iteration_nbelow!(nbelow::G, openRows::Queue{G}, astate::AssignmentState{G, T},
                                   lambda::T, epsi::T,
                                   frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                                   dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned row to attempt to match
    row = dequeue!(openRows)

    ##Row may have been assigned in a reverse iteration, assumes another free row since otherwise all rows are already assigned...
    if !iszero(astate.r2c[row])
        return forward_iteration_nbelow!(nbelow, openRows, astate, lambda, epsi, frewardMatrix, dfltReward, dfltTwo)
    end

    maxcol, bid, newRowPrice = forward_bid(row, astate.colPrices, epsi, frewardMatrix, dfltReward, dfltTwo)
    addassign, nbelow, openRows, astate = forward_update_nbelow!(row, maxcol, bid, newRowPrice, nbelow, openRows, astate, lambda)

    return addassign, nbelow, openRows, astate
end

########################################
#Reverse Iterations
########################################

"""
    reverse_iteration!(openCols::Queue{G}, astate::AssignmentState{G, T}, epsi::T, rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, openRows, astate)
    reverse_iteration!(openColsAbove::Queue{G}, astate::AssignmentState{G, T}, lambda::T, epsi::T, rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, openRows, astate)
    reverse_iteration!(openColsAbove::Queue{G}, astate::AssignmentState{G, T}, lambda::T, epsi::T, rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, openRows, astate)

Find an unassigned column, compute the `reverse_bid` and then perform the `reverse_update!`.

If the reward matrix is asymmetric then a `lambda` parameter is required. If the reward
matrix is sparse and extra entries are to be assumed (to ensure a feasible assignment)
then a `dfltReward` must be supplied. `addassign::Bool` indicates if the
 total number of assignments has increased.

# Arguments

* `openCols::Queue{G}`: Queue containing unassigned columns.
* `openColsAbove::Queue{G}`: Queue containing unassigned columns where `astate.colPrices` > `lambda`.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.
* `epsi::T`: tolerance associated with auction algorithm iteration.
* `rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: reward matrix.
* `dfltTwo::T`: default second largest value passed to `reverse_bid` if the rewardMatrix is
sparse.  Ignored if the reward matrix is not sparse.
* `dfltReward::T`: Default reward value used for implicity added entries if a padded reward
matrix is being used, passed to `reverse_bid`.

See also: [`reverse_iteration_nbelow!`](@ref), [`reverse_bid`](@ref), [`reverse_update!`](@ref), [`reverse_update_nbelow!`](@ref), [`forward_iteration!`](@ref), [`AssignmentState`](@ref)
"""
function reverse_iteration!(openCols::Queue{G}, astate::AssignmentState{G, T},
                            epsi::T,
                            rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                            dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned col to attempt to match, needs to have price > lambda
    col = dequeue!(openCols)

    ##col price may not be > lambda since queue is not updated in forward iterations
    if !iszero(astate.c2r[col])

        ##Recursively call if additional columns above lambda are available
        if length(openCols) > 0
            return reverse_iteration!(openCols, astate, epsi, rewardMatrix, dfltTwo)
        else
            addassign = false
            return addassign, openCols, astate
        end
    end

    maxrow, bid, newColPrice = reverse_bid(col, astate.rowPrices, epsi, rewardMatrix, dfltTwo)
    addassign, openColsAbove, astate = reverse_update!(maxrow, col, bid, newColPrice, openCols, astate)

    return addassign, openCols, astate
end

#asymmetric, includes lambda
function reverse_iteration!(openColsAbove::Queue{G}, astate::AssignmentState{G, T},
                            lambda::T, epsi::T,
                            rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                            dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned col to attempt to match, needs to have price > lambda
    col = dequeue!(openColsAbove)

    ##col price may not be > lambda since queue is not updated in forward iterations
    if astate.colPrices[col] <= lambda || !iszero(astate.c2r[col])

        ##Recursively call if additional columns above lambda are available
        if length(openColsAbove) > 0
            return reverse_iteration!(openColsAbove, astate, lambda, epsi, rewardMatrix, dfltTwo)
        else
            addassign = false
            return addassign, openColsAbove, astate
        end
    end

    maxrow, bid, newColPrice = reverse_bid(col, astate.rowPrices, lambda, epsi, rewardMatrix, dfltTwo)
    addassign, openColsAbove, astate = reverse_update!(maxrow, col, bid, newColPrice, openColsAbove, astate, lambda)

    return addassign, openColsAbove, astate
end

#padded
function reverse_iteration!(openColsAbove::Queue{G}, astate::AssignmentState{G, T},
                            lambda::T, epsi::T,
                            rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                            dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned col to attempt to match, needs to have price > lambda
    col = dequeue!(openColsAbove)

    ##col price may not be > lambda since queue is not updated in forward iterations
    if astate.colPrices[col] <= lambda || !iszero(astate.c2r[col])

        ##Recursively call if additional columns above lambda are available
        if length(openColsAbove) > 0
            return reverse_iteration!(openColsAbove, astate, lambda, epsi, rewardMatrix, dfltReward, dfltTwo)
        else
            addassign = false
            return addassign, openColsAbove, astate
        end
    end

    maxrow, bid, newColPrice = reverse_bid(col, astate.rowPrices, lambda, epsi, rewardMatrix, dfltReward, dfltTwo)
    addassign, openColsAbove, astate = reverse_update!(maxrow, col, bid, newColPrice, openColsAbove, astate, lambda)

    return addassign, openColsAbove, astate
end

"""
    reverse_iteration_nbelow!(nbelow::G, openColsAbove::Queue{G}, astate::AssignmentState{G, T}, lambda::T, epsi::T, rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, nbelow, openRows, astate)
    reverse_iteration_nbelow!(nbelow::G, openColsAbove::Queue{G}, astate::AssignmentState{G, T}, lambda::T, epsi::T, rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat} -> (addassign, nbelow, openRows, astate)

Find an unassigned column, compute the `reverse_bid` and then perform the `reverse_update_nbelow!`, identical to `reverse_update!` but tracks the number of column prices below `lambda`.

If the reward matrix is asymmetric then a `lambda` parameter is required. If the reward
matrix is sparse and extra entries are to be assumed (to ensure a feasible assignment)
then a `dfltReward` must be supplied. `addassign::Bool` indicates if the
 total number of assignments has increased.

# Arguments

* `nbelow::G`: number of column prices (`astate.colPrices`) < `lambda`.
* `openCols::Queue{G}`: Queue containing unassigned columns.
* `openColsAbove::Queue{G}`: Queue containing unassigned columns where `astate.colPrices` > `lambda`.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.
* `epsi::T`: tolerance associated with auction algorithm iteration.
* `rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: reward matrix.
* `dfltTwo::T`: default second largest value passed to `reverse_bid` if the rewardMatrix is
sparse.  Ignored if the reward matrix is not sparse.
* `dfltReward::T`: Default reward value used for implicity added entries if a padded reward
matrix is being used, passed to `reverse_bid`.

See also: [`reverse_iteration!`](@ref), [`reverse_bid`](@ref), [`reverse_update!`](@ref), [`reverse_update_nbelow!`](@ref), [`forward_iteration!`](@ref), [`AssignmentState`](@ref)
"""
function reverse_iteration_nbelow!(nbelow::G, openColsAbove::Queue{G}, astate::AssignmentState{G, T},
                                   lambda::T, epsi::T,
                                   rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                                   dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned col to attempt to match, needs to have price > lambda
    col = dequeue!(openColsAbove)

    ##col price may not be > lambda since queue is not updated in forward iterations
    if astate.colPrices[col] <= lambda || !iszero(astate.c2r[col])

        ##Recursively call if additional columns above lambda are available
        if length(openColsAbove) > 0
            return reverse_iteration_nbelow!(nbelow, openColsAbove, astate, lambda, epsi, rewardMatrix, dfltTwo)
        else
            addassign = false
            return addassign, nbelow, openColsAbove, astate
        end
    end

    maxrow, bid, newColPrice = reverse_bid(col, astate.rowPrices, lambda, epsi, rewardMatrix, dfltTwo)
    addassign, nbelow, openColsAbove, astate = reverse_update_nbelow!(maxrow, col, bid, newColPrice, nbelow, openColsAbove, astate, lambda)

    return addassign, nbelow, openColsAbove, astate
end

#padded
function reverse_iteration_nbelow!(nbelow::G, openColsAbove::Queue{G}, astate::AssignmentState{G, T},
                                   lambda::T, epsi::T,
                                   rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}},
                                   dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}
    
    ##Choose unassigned col to attempt to match, needs to have price > lambda
    col = dequeue!(openColsAbove)

    ##col price may not be > lambda since queue is not updated in forward iterations
    if astate.colPrices[col] <= lambda || !iszero(astate.c2r[col])

        ##Recursively call if additional columns above lambda are available
        if length(openColsAbove) > 0
            return reverse_iteration_nbelow!(nbelow, openColsAbove, astate, lambda, epsi, rewardMatrix, dfltReward, dfltTwo)
        else
            addassign = false
            return addassign, nbelow, openColsAbove, astate
        end
    end

    maxrow, bid, newColPrice = reverse_bid(col, astate.rowPrices, lambda, epsi, rewardMatrix, dfltReward, dfltTwo)
    addassign, nbelow, openColsAbove, astate = reverse_update_nbelow!(maxrow, col, bid, newColPrice, nbelow, openColsAbove, astate, lambda)

    return addassign, nbelow, openColsAbove, astate
end

########################################
#Using Only Forward Scaling
########################################

"""
    scaling_as!(astate::AssignmentState{G, T}, epsi::T, rewardMatrix::A, frewardMatrix::A, dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}
    scaling_as!(astate::AssignmentState{G, T}, epsi::T, rewardMatrix::A, frewardMatrix::A, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}

Perform asymmetric scaling phase finding a feasible assignment (not necessarily optimal) for `rewardMatrix` and `epsi`.

If a `dfltReward` parameter is passed then `astate` is assumed to be 'padded' so that dummy
entries have been added so `rewardMatrix[row, row + size(rewardMatrix, 2)] = dfltReward`.

# Arguments

* `astate::AssignmentState{G, T}` State of assignment solution.
* `epsi::T`: tolerance associated with auction algorithm iteration.
* `rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: reward matrix.
* `frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: forward reward matrix.  If the
reward matrix is space then this is the transpose of the reward matrix.  Otherwise it is just
the reward matrix.
* `dfltTwo::T`: default second largest value passed to `reverse_bid` if the rewardMatrix is
sparse.  Ignored if the reward matrix is not sparse.
* `dfltReward::T`: Default reward value used for implicity added entries if a padded reward
matrix is being used, passed to `reverse_bid`.

See also: [`forward_iteration!`](@ref), [`auction_assignment_as`](@ref), [`auction_assignment_padas`](@ref), [`AssignmentState`](@ref)
"""
function scaling_as!(astate::AssignmentState{G, T}, epsi::T,
                    rewardMatrix::A, frewardMatrix::A,
                    dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}

    ##Count initially assign rows and add others to Queue
    openRows = get_openrows(astate)
    lambda = zero(T)
    
    ##Use forward iterations to find feasible assignment
    while astate.nassigned < astate.nrow
        addassign, openRows, astate = forward_iteration!(openRows, astate, lambda, epsi, frewardMatrix, dfltTwo)
    end
    
    return astate
end

#padded version
function scaling_as!(astate::AssignmentState{G, T}, epsi::T,
                    rewardMatrix::A, frewardMatrix::A,
                    dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}
    
    ##Count initially assign rows and add others to Queue
    openRows = get_openrows(astate)
    lambda = zero(T)
    
    ##Use forward iterations to find feasible assignment
    while astate.nassigned < astate.nrow
        addassign, openRows, astate = forward_iteration!(openRows, astate, lambda, epsi, frewardMatrix, dfltReward, dfltTwo)
    end
    
    return astate
end

"""
    auction_assignment_as(astate::AssignmentState{G, T}, rewardMatrix::A, frewardMatrix::A = forward_rewardmatrix(rewardMatrix); epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.2), dfltTwo::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}} -> (astate, lambda)

Return an approximately optimal assignment to maximize reward of assignment based on `rewardMatrix`.

Implementation of the 'AS' auction algorithm for asymmetric assignment problems (Bertsekas 1992).
Asymmetry assumes that `size(rewardMatrix, 1) <= size(rewardMatrix, 2)`.  For sparse problems
a feasible solution is also assumed to exist.  If this is not certain it is recommended that
`auction_assignment_padas` be used as this will guarantee a feasible solution exists.

# Arguments

* `rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: reward matrix.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: forward reward matrix.  If the
reward matrix is space then this is the transpose of the reward matrix.  Otherwise it is just
the reward matrix.
* `epsi0::T`: Starting tolerance associated with auction algorithm iteration.
* `epsitol::T`: Final tolerance, result with be within `size(rewardMatrix, 1) * epsitol` of optimal.
* `epsiscale::T`: Scaling rate of `epsi` at each interation `epsinew = epsi * epsiscale`.
* `dfltTwo::T`: default second largest value passed to `reverse_bid` if the rewardMatrix is
sparse.  Ignored if the reward matrix is not sparse.

See also: [`scaling_as!`](@ref), [`auction_assignment_padas`](@ref), [`auction_assignment_padas`](@ref), [`AssignmentState`](@ref)
"""
function auction_assignment_as(rewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}};
                               astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = false),
                               frewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}} = forward_rewardmatrix(rewardMatrix),
                               epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.2),
                               dfltTwo::T = -T(Inf)) where {T <: AbstractFloat}

    check_epsilons(epsi0, epsitol, epsiscale)
    epsi = epsi0
    abovetol = epsi >= epsitol
    
    while abovetol
        astate = scaling_as!(astate, epsi, rewardMatrix, frewardMatrix, dfltTwo)
        
        if epsi < epsitol
            abovetol = false
        else
            astate, epsi = scale_assignment!(astate, epsi, epsiscale)
        end
    end
    
    ##Set lambda to minimum assigned prices
    if astate.nexcesscols > zero(G)
        lambda = min_assigned_colprice(astate)
    else
        lambda = zero(T)
    end
    openColsAbove = get_opencolsabove(astate, lambda)
    
    ##Execute reverse iterations until termination
    while length(openColsAbove) > 0
        addassign, openColsAbove, astate = reverse_iteration!(openColsAbove, astate, lambda, epsi, rewardMatrix, dfltTwo)
    end
    
    return astate, lambda
end

function auction_assignment_padas(rewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}};
                                  astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = true),
                                  frewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}} = forward_rewardmatrix(rewardMatrix),
                                  epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.2),
                                  dfltReward::T = zero(T), dfltTwo::T = -T(Inf)) where {T <: AbstractFloat}

    check_epsilons(epsi0, epsitol, epsiscale)
    epsi = epsi0
    abovetol = epsi >= epsitol

    while abovetol
        astate = scaling_as!(astate, epsi, rewardMatrix, frewardMatrix, dfltReward, dfltTwo)
        
        if epsi < epsitol
            abovetol = false
        else
            astate, epsi = scale_assignment!(astate, epsi, epsiscale)
        end
    end
    
    ##Set lambda to minimum assigned prices
    if astate.nexcesscols > zero(G)
        lambda = min_assigned_colprice(astate)
    else
        lambda = zero(T)
    end
    openColsAbove = get_opencolsabove(astate, lambda)
    
    ##Execute reverse iterations until termination
    while length(openColsAbove) > 0
        addassign, openColsAbove, astate = reverse_iteration!(openColsAbove, astate, lambda, epsi, rewardMatrix, dfltReward, dfltTwo)
    end
    
    return astate, lambda
end

########################################
#Using Only Forward Scaling
########################################

"""
    scaling_asfr1!(astate::AssignmentState{G, T}, lambda::T, epsi::T, rewardMatrix::A, frewardMatrix::A, dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}
    scaling_asfr1!(astate::AssignmentState{G, T}, lambda::T, epsi::T, rewardMatrix::A, frewardMatrix::A, dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}

Perform asymmetric scaling phase finding a nearly optimal assignment for `rewardMatrix` and `epsi` for 'ASFR1' algorithm.

If a `dfltReward` parameter is passed then `astate` is assumed to be 'padded' so that dummy
entries have been added so `rewardMatrix[row, row + size(rewardMatrix, 2)] = dfltReward`.

# Arguments

* `astate::AssignmentState{G, T}` State of assignment solution.
* `lambda::T`: profitability threshold parameter for asymmetric auction algorithm.
* `epsi::T`: tolerance associated with auction algorithm iteration.
* `rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: reward matrix.
* `frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: forward reward matrix.  If the
reward matrix is space then this is the transpose of the reward matrix.  Otherwise it is just
the reward matrix.
* `dfltTwo::T`: default second largest value passed to `reverse_bid` if the rewardMatrix is
sparse.  Ignored if the reward matrix is not sparse.
* `dfltReward::T`: Default reward value used for implicity added entries if a padded reward
matrix is being used, passed to `reverse_bid`.

See also: [`forward_iteration!`](@ref), [`auction_assignment_asfr1`](@ref), [`auction_assignment_padasfr1`](@ref), [`AssignmentState`](@ref)
"""
function scaling_asfr1!(astate::AssignmentState{G, T},
                       lambda::T, epsi::T,
                       rewardMatrix::A, frewardMatrix::A,
                       dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}


    ##Count initially assign rows and add others to Queue
    openRows = get_openrows(astate)
    nbelow, openColsAbove = get_nbelow_opencolsabove(astate, lambda)
    
    while (astate.nassigned < astate.nrow) || length(openColsAbove) > 0

        ##Only execute forward iteration if not all rows are assigned
        if astate.nassigned < astate.nrow
            #println("Forward Iteration: $nassigned assigned")
            addassign = false
            while !addassign
                addassign, nbelow, openRows, astate = forward_iteration_nbelow!(nbelow, openRows, astate, lambda, epsi, frewardMatrix, dfltTwo)
            end
        end

        ##Only execute reverse iteration if open columns above lambda
        addassign = false
        while !addassign && length(openColsAbove) > 0
            #println("Reverse Iteration: $nassigned assigned")
            addassign, nbelow, openColsAbove, astate = reverse_iteration_nbelow!(nbelow, openColsAbove, astate, lambda, epsi, rewardMatrix, dfltTwo)
        end

        ## Adjust lambda after reverse iterations if necessary
        if nbelow > astate.nexcesscols
            if astate.nexcesscols > zero(G)
                colperm = sortperm(astate.colPrices)
                newlambda = astate.colPrices[colperm[astate.nexcesscols]]
                
                #find column prices above new lambda value but <= old lambda value and add to set of columns that can be assigned
                jj = astate.nexcesscols + one(G)
                while (jj <= astate.ncol) && (astate.colPrices[colperm[jj]] <= lambda)
                    if astate.colPrices[colperm[jj]] > newlambda
                        enqueue!(openColsAbove, colperm[jj])
                    end
                    jj += one(G)
                end
                lambda = newlambda
            else
                lambda = max(minimum(astate.colPrices), zero(T))
                nbelow, openColsAbove = get_nbelow_opencolsabove(astate, lambda)
            end
        end
    end

    return astate, lambda
end

function scaling_asfr1!(astate::AssignmentState{G, T},
                       lambda::T, epsi::T,
                       rewardMatrix::A, frewardMatrix::A,
                       dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}


    ##Count initially assign rows and add others to Queue
    openRows = get_openrows(astate)
    nbelow, openColsAbove = get_nbelow_opencolsabove(astate, lambda)
    
    while (astate.nassigned < astate.nrow) || length(openColsAbove) > 0

        ##Only execute forward iteration if not all rows are assigned
        if astate.nassigned < astate.nrow
            #println("Forward Iteration: $nassigned assigned")
            addassign = false
            while !addassign
                addassign, nbelow, openRows, astate = forward_iteration_nbelow!(nbelow, openRows, astate, lambda, epsi, frewardMatrix, dfltReward, dfltTwo)
            end
        end

        ##Only execute reverse iteration if open columns above lambda
        addassign = false
        while !addassign && length(openColsAbove) > 0
            #println("Reverse Iteration: $nassigned assigned")
            addassign, nbelow, openColsAbove, astate = reverse_iteration_nbelow!(nbelow, openColsAbove, astate, lambda, epsi, rewardMatrix, dfltReward, dfltTwo)
        end

        ## Adjust lambda after reverse iterations if necessary
        if nbelow > astate.nexcesscols
            colperm = sortperm(astate.colPrices)
            newlambda = astate.colPrices[colperm[astate.nexcesscols]]

            #find column prices above new lambda value but <= old lambda value and add to set of columns that can be assigned
            jj = astate.nexcesscols + one(G)
            while (jj <= astate.ncol) && (astate.colPrices[colperm[jj]] <= lambda)
                if astate.colPrices[colperm[jj]] > newlambda
                    enqueue!(openColsAbove, colperm[jj])
                end
                jj += one(G)
            end
            lambda = newlambda
        end
    end

    return astate, lambda
end

"""
    auction_assignment_asfr1(rewardMatrix::A; astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = false), frewardMatrix::A = forward_rewardmatrix(rewardMatrix), lambda0::T = zero(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1), dfltTwo::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}} -> astate, lambda

Find approximately optimal (maximal) assignment for `rewardMatrix` and `epsitol` employing epsilon scaling.

Implementation of the 'ASFR1' auction algorithm for asymmetric assignment problems (Bertsekas 1992).
Asymmetry assumes that `size(rewardMatrix, 1) <= size(rewardMatrix, 2)`.  For sparse problems
a feasible solution is also assumed to exist.  If this is not certain it is recommended that
`auction_assignment_padasfr1` be used as this will guarantee a feasible solution exists.

# Arguments

* `rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: reward matrix.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: forward reward matrix.  If the
reward matrix is space then this is the transpose of the reward matrix.  Otherwise it is just
the reward matrix.
* `lambda0::T`: Initial value of the profitability threshold parameter for asymmetric auction algorithm.
* `epsi0::T`: Starting tolerance associated with auction algorithm iteration.
* `epsitol::T`: Final tolerance, result with be within `size(rewardMatrix, 1) * epsitol` of optimal.
* `epsiscale::T`: Scaling rate of `epsi` at each interation `epsinew = epsi * epsiscale`.

* `dfltTwo::T`: default second largest value passed to `reverse_bid` if the rewardMatrix is
sparse.  Ignored if the reward matrix is not sparse.

See also: [`forward_iteration!`](@ref), [`auction_assignment_asfr1`](@ref), [`auction_assignment_padasfr1`](@ref), [`AssignmentState`](@ref)
"""
function auction_assignment_asfr1(rewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}};
                                  astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = false),
                                  frewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}} = forward_rewardmatrix(rewardMatrix),
                                  lambda0::T = zero(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                  dfltTwo::T = -T(Inf)) where {T <: AbstractFloat}
    
    check_epsilons(epsi0, epsitol, epsiscale)
    epsi = copy(epsi0)
    lambda = copy(lambda0)

    abovetol = epsi >= epsitol
    while abovetol
        
         astate, lambda = scaling_asfr1!(astate, lambda, epsi, rewardMatrix, frewardMatrix, dfltTwo)
        
        if epsi < epsitol
            abovetol = false
        else
            #newepsi = epsi * epsiscale
            #deltaepsi = epsi - newepsi

            #rowPrices are increased so that complimentary slackness is maintained with smaller epsilon
            #astate.rowPrices .+= deltaepsi
            #astate = clear_assignment!(astate)
            #epsi = newepsi
            astate, epsi = scale_assignment!(astate, epsi, epsiscale)
        end
    end
    return astate, lambda
end

"""
    auction_assignment_asfr1(rewardMatrix::A; astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = false), frewardMatrix::A = forward_rewardmatrix(rewardMatrix), lambda0::T = zero(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1), dfltTwo::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}} -> astate, lambda

Find approximately optimal (maximal) assignment for `rewardMatrix` and `epsitol` employing epsilon scaling.

Implementation of the 'ASFR1' auction algorithm for asymmetric assignment problems (Bertsekas 1992).
Asymmetry assumes that `size(rewardMatrix, 1) <= size(rewardMatrix, 2)`.  For sparse problems
a feasible solution is also assumed to exist.  If this is not certain it is recommended that
`auction_assignment_padasfr1` be used as this will guarantee a feasible solution exists.

# Arguments

* `rewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: reward matrix.
* `astate::AssignmentState{G, T}` State of assignment solution.
* `frewardMatrix::Union{SparseMatrixCSC{T, G}, Array{T, 2}}`: forward reward matrix.  If the
reward matrix is space then this is the transpose of the reward matrix.  Otherwise it is just
the reward matrix.
* `lambda0::T`: Initial value of the profitability threshold parameter for asymmetric auction algorithm.
* `epsi0::T`: Starting tolerance associated with auction algorithm iteration.
* `epsitol::T`: Final tolerance, result with be within `size(rewardMatrix, 1) * epsitol` of optimal.
* `epsiscale::T`: Scaling rate of `epsi` at each interation `epsinew = epsi * epsiscale`.

* `dfltTwo::T`: default second largest value passed to `reverse_bid` if the rewardMatrix is
sparse.  Ignored if the reward matrix is not sparse.
* `dfltReward::T`: Default reward value used for implicity added entries if a padded reward
matrix is being used, passed to `reverse_bid`.

See also: [`forward_iteration!`](@ref), [`auction_assignment_asfr1`](@ref), [`auction_assignment_padasfr1`](@ref), [`AssignmentState`](@ref)
"""
function auction_assignment_padasfr1(rewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}};
                                     astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = true),
                                     frewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}} = forward_rewardmatrix(rewardMatrix),
                                     lambda0::T = zero(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                     dfltReward::T = zero(T), dfltTwo::T = -T(Inf)) where {T <: AbstractFloat}

    check_epsilons(epsi0, epsitol, epsiscale)
    epsi = copy(epsi0)
    lambda = copy(lambda0)

    abovetol = epsi >= epsitol
    while abovetol
        
         astate, lambda = scaling_asfr1!(astate, lambda, epsi, rewardMatrix, frewardMatrix, dfltReward, dfltTwo)
        
        if epsi < epsitol
            abovetol = false
        else
            #newepsi = epsi * epsiscale
            #deltaepsi = epsi - newepsi

            #rowPrices are increased so that complimentary slackness is maintained with smaller epsilon
            #astate.rowPrices .+= deltaepsi
            #astate = clear_assignment!(astate)
            #epsi = newepsi
            astate, epsi = scale_assignment!(astate, epsi, epsiscale)
        end
    end
    return astate, lambda
end

#=
function auction_assignment_asfr1(rewardMatrix::A,
                                  frewardMatrix::A = forward_rewardmatrix(rewardMatrix);
                                  lambda0::T = -one(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                  dfltTwo::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}
    rowPrices = vec(maximum(rewardMatrix, dims = 2))
    rowPrices, colPrices = dimmaximums(rewardMatrix)
    rowPrices .-= minimum(colPrices)
    astate = AssignmentState(rowPrices, colPrices)
    
    if (lambda0 < zero(T)) && (astate.nexcesscols > zero(G))
        lambda0 = sort(colPrices)[astate.ncol - astate.nrow]
    end
    return auction_assignment_asfr1(astate, rewardMatrix, lambda0 = lambda0, epsi0 = epsi0, epsitol = epsitol, epsiscale = epsiscale, dfltTwo = dfltTwo)
end
=#

#=
function auction_assignment_asfr1(rewardMatrix::SparseMatrixCSC{T, G},
                                  trewardMatrix::SparseMatrixCSC{T, G} = permutedims(rewardMatrix, [2, 1]);
                                  lambda0::T = -one(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                  dfltTwo::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    rowPrices, colPrices = dimmaximums(rewardMatrix)
    rowPrices .-= minimum(colPrices)
    astate = AssignmentState(rowPrices, colPrices)
    
    if (lambda0 < zero(T)) && (astate.ncol - astate.nrow) > zero(G)
        lambda0 = sort(colPrices)[astate.ncol - astate.nrow]
    end
    return auction_assignment_asfr1(astate, rewardMatrix, trewardMatrix, lambda0 = lambda0, epsi0 = epsi0, epsitol = epsitol, epsiscale = epsiscale, dfltTwo = dfltTwo)
end
=#
#Forward Reverse Algorithm, fixed lambda
#Purely Reverse Algorithm
#    1) Forward Algorithm until assignment is feasible (everything assigned)
#    2) Set lambda to min assigned object prices
#    3) Run reverse algorithm until termination
# *Track unassigned rows and columns

function scaling_asfr2!(astate::AssignmentState{G, T},
                       lambda::T, epsi::T,
                       rewardMatrix::A, frewardMatrix::A,
                       dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}

    ##Count initially assign rows and add others to Queue
    openRows = get_openrows(astate)
    openColsAbove = get_opencolsabove(astate, lambda)
    
    while astate.nassigned < astate.nrow || length(openColsAbove) > 0

        ##Only execute forward iteration if not all rows are assigned
        if astate.nassigned < astate.nrow
            #println("Forward Iteration: $nassigned assigned")
            addassign = false
            while !addassign
                addassign, openRows, astate = forward_iteration!(openRows, astate, lambda, epsi, frewardMatrix, dfltTwo)
            end
        end

        ##Only execute reverse iteration if open columns above lambda
        addassign = false
        while !addassign && length(openColsAbove) > 0
            #println("Forward Iteration: $nassigned assigned")
            addassign, openColsAbove, astate = reverse_iteration!(openColsAbove, astate, lambda, epsi, rewardMatrix, dfltTwo)
        end
    end

    ##lambda returned for consistency with other implimentations
    return astate, lambda
end

function scaling_asfr2!(astate::AssignmentState{G, T},
                       lambda::T, epsi::T,
                       rewardMatrix::A, frewardMatrix::A,
                       dfltReward::T, dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}

    ##Count initially assign rows and add others to Queue
    openRows = get_openrows(astate)
    openColsAbove = get_opencolsabove(astate, lambda)
    
    while astate.nassigned < astate.nrow || length(openColsAbove) > 0

        ##Only execute forward iteration if not all rows are assigned
        if astate.nassigned < astate.nrow
            #println("Forward Iteration: $nassigned assigned")
            addassign = false
            while !addassign
                addassign, openRows, astate = forward_iteration!(openRows, astate, lambda, epsi, frewardMatrix, dfltReward, dfltTwo)
            end
        end

        ##Only execute reverse iteration if open columns above lambda
        addassign = false
        while !addassign && length(openColsAbove) > 0
            #println("Forward Iteration: $nassigned assigned")
            addassign, openColsAbove, astate = reverse_iteration!(openColsAbove, astate, lambda, epsi, rewardMatrix, dfltReward, dfltTwo)
        end
    end

    ##lambda returned for consistency with other implimentations
    return astate, lambda
end

function auction_assignment_asfr2(rewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}};
                                  astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = false),
                                  frewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}} = forward_rewardmatrix(rewardMatrix),
                                  lambda0::T = zero(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                  dfltTwo::T = -T(Inf)) where {T <: AbstractFloat}

    check_epsilons(epsi0, epsitol, epsiscale)
    epsi = copy(epsi0)
    lambda = copy(lambda0)

    abovetol = epsi >= epsitol
    while abovetol
        
         astate, lambda = scaling_asfr2!(astate, lambda, epsi, rewardMatrix, frewardMatrix, dfltTwo)
        
        if epsi < epsitol
            abovetol = false
        else
            astate, epsi = scale_assignment!(astate, epsi, epsiscale)
        end
    end
    return astate, lambda
end

function auction_assignment_padasfr2(rewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}};
                                     astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = true),
                                     frewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}} = forward_rewardmatrix(rewardMatrix),
                                     lambda0::T = zero(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                     dfltReward::T = zero(T), dfltTwo::T = -T(Inf)) where {T <: AbstractFloat}
    
    check_epsilons(epsi0, epsitol, epsiscale)
    epsi = copy(epsi0)
    lambda = copy(lambda0)

    abovetol = epsi >= epsitol
    while abovetol
        
         astate, lambda = scaling_asfr2!(astate, lambda, epsi, rewardMatrix, frewardMatrix, dfltReward, dfltTwo)
        
        if epsi < epsitol
            abovetol = false
        else
            astate, epsi = scale_assignment!(astate, epsi, epsiscale)
        end
    end
    return astate, lambda
end

#=
function auction_assignment_asfr2(rewardMatrix::A,
                                  frewardMatrix::A = forward_rewardmatrix(rewardMatrix);
                                  lambda0::T = zero(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                  dfltTwo::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}
    astate = AssignmentState(rewardMatrix)
    return auction_assignment_asfr2(astate, rewardMatrix, frewardMatrix, lambda0 = lambda0, epsi0 = epsi0, epsitol = epsitol, epsiscale = epsiscale, dfltTwo = dfltTwo)
end
=#

function scaling_syfr!(astate::AssignmentState{G, T},
                       epsi::T,
                       rewardMatrix::A, frewardMatrix::A,
                       dfltTwo::T) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}

    ##Count initially assign rows and add others to Queue
    openRows = get_openrows(astate)
    openCols = get_opencols(astate)

    while astate.nassigned < astate.nrow

        ##Only execute forward iteration if not all rows are assigned
        if astate.nassigned < astate.nrow
            addassign = false
            while !addassign
                #println("Forward Iteration: $(astate.nassigned) assigned")
                addassign, openRows, astate = forward_iteration!(openRows, astate, epsi, frewardMatrix, dfltTwo)
            end
        end
        
        ##Only execute reverse iteration if not all columns are assigned
        if astate.nassigned < astate.nrow
            addassign = false
            while !addassign
                #println("Reverse Iteration: $(astate.nassigned) assigned")
                addassign, openCols, astate = reverse_iteration!(openCols, astate, epsi, rewardMatrix, dfltTwo)                
            end
        end

    end

    return astate
end

function auction_assignment_syfr(rewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}};
                                 astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = false),
                                 frewardMatrix::Union{SparseMatrixCSC{T}, Array{T, 2}} = forward_rewardmatrix(rewardMatrix),
                                 epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                 dfltTwo::T = -T(Inf)) where {T <: AbstractFloat}

    check_epsilons(epsi0, epsitol, epsiscale)
    epsi = copy(epsi0)

    abovetol = epsi >= epsitol
    while abovetol
        
         astate = scaling_syfr!(astate, epsi, rewardMatrix, frewardMatrix, dfltTwo)
        
        if epsi < epsitol
            abovetol = false
        else
            astate, epsi = scale_assignment!(astate, epsi, epsiscale)
        end
    end
    return astate, zero(T)
end

#=
function auction_assignment_syfr(rewardMatrix::A,
                                 frewardMatrix::A = forward_rewardmatrix(rewardMatrix);
                                 epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                  dfltTwo::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat, A <: Union{SparseMatrixCSC{T, G}, Array{T, 2}}}
    astate = AssignmentState(rewardMatrix, maximize = true, assign = true, pad = false)
    return auction_assignment_syfr(astate, rewardMatrix, frewardMatrix, epsi0 = epsi0, epsitol = epsitol, epsiscale = epsiscale, dfltTwo = dfltTwo)
end
=#

function auction_assignment(rewardMatrix::Array{T, 2}; algorithm = "") where T <: AbstractFloat
    if size(rewardMatrix, 1) > size(rewardMatrix, 2)
        @warn "more columns than rows, some rows will be unassigned"
        astate, lambda = auction_assignment(permutedims(rewardMatrix, [2, 1]), algorithm = algorithm)
        return flip(astate), lambda
    else
        if algorithm == ""
            if size(rewardMatrix, 1) == size(rewardMatrix, 2)
                @warn "No algorithm specified for symmetric rewardMatrix, setting algorithm = 'syfr'"
                astate, lambda = auction_assignment_syfr(rewardMatrix)
            else
                @warn "No algorithm specified for asymmetric rewardMatrix, setting algorithm = 'asfr2'"
                astate, lambda = auction_assignment_asfr2(rewardMatrix)
            end
        elseif algorithm == "as"
            if size(rewardMatrix, 1) == size(rewardMatrix, 2)
                @warn "Symmetric rewardMatrix, consider setting algorithm = 'syfr'"
            end
            astate, lambda = auction_assignment_as(rewardMatrix)
        elseif algorithm == "asfr1"
            if size(rewardMatrix, 1) == size(rewardMatrix, 2)
                @warn "Symmetric rewardMatrix, consider setting algorithm = 'syfr'"
            end
            astate, lambda = auction_assignment_asfr1(rewardMatrix)
        elseif algorithm == "asfr2"
            if size(rewardMatrix, 1) == size(rewardMatrix, 2)
                @warn "Symmetric rewardMatrix, consider setting algorithm = 'syfr'"
            end
            astate, lambda = auction_assignment_asfr2(rewardMatrix)
        elseif algorithm == "syfr"
            if size(rewardMatrix, 1) != size(rewardMatrix, 2)
                @warn "Symmetric algorithm selected for asymmetric problem, setting algorithm to 'asfr2'"
                astate, lambda = auction_assignment_asfr2(rewardMatrix)
            else
                astate, lambda = auction_assignment_syfr(rewardMatrix)
            end
        else
            error("algorithm = $algorithm not supported. Set algorithm to one of 'as', 'asfr1', 'asfr2' or 'syfr'")
        end
        return astate, lambda
    end
end

function auction_assignment(rewardMatrix::SparseMatrixCSC{T, G}; algorithm = "") where {G <: Integer, T <: AbstractFloat}
    if size(rewardMatrix, 1) > size(rewardMatrix, 2)
        @warn "more columns than rows, some rows will be unassigned"
        if algorithm == ""
            @warn "no algorithm specificed using 'asfr2'"
            astate, lambda = auction_assignment_asfr2(permutedims(rewardMatrix, [2, 1]), rewardMatrix)
        elseif algorithm == "as"
            astate, lambda = auction_assignment_as(permutedims(rewardMatrix, [2, 1]), rewardMatrix)
        elseif algorithm == "asfr1"
            astate, lambda = auction_assignment_asfr1(permutedims(rewardMatrix, [2, 1]), rewardMatrix)
        elseif algorithm == "asfr2"
            astate, lambda = auction_assignment_asfr2(permutedims(rewardMatrix, [2, 1]), rewardMatrix)
        elseif algorithm == "syfr"
            @warn "Symmetric algorithm selected for asymmetric problem, setting algorithm to 'asfr2'"
            astate, lambda = auction_assignment_asfr2(permutedims(rewardMatrix, [2, 1]), rewardMatrix)
        else
            error("algorithm = '$algorithm' not supported set algorithm to one of 'as', 'asfr1', or 'asfr2'")
        end
        return flip(astate), lambda
    else
        if algorithm == ""
            if size(rewardMatrix, 1) == size(rewardMatrix, 2)
                astate, lambda = auction_assignment_syfr(rewardMatrix)
            else
                astate, lambda = auction_assignment_asfr2(rewardMatrix)
            end
        elseif algorithm == "as"
            if size(rewardMatrix, 1) == size(rewardMatrix, 2)
                @warn "Symmetric rewardMatrix, consider setting algorithm = 'syfr'"
            end
            astate, lambda = auction_assignment_as(rewardMatrix)
        elseif algorithm == "asfr1"
            if size(rewardMatrix, 1) == size(rewardMatrix, 2)
                @warn "Symmetric rewardMatrix, consider setting algorithm = 'syfr'"
            end
            astate, lambda = auction_assignment_asfr1(rewardMatrix)
        elseif algorithm == "asfr2"
            if size(rewardMatrix, 1) == size(rewardMatrix, 2)
                @warn "Symmetric rewardMatrix, consider setting algorithm = 'syfr'"
            end
            astate, lambda = auction_assignment_asfr2(rewardMatrix)
        elseif algorithm == "syfr"
            if size(rewardMatrix, 1) != size(rewardMatrix, 2)
                @warn "Symmetric algorithm selected for asymmetric problem, setting algorithm to 'asfr2'"
                astate, lambda = auction_assignment_asfr2(rewardMatrix)
            else
                astate, lambda = auction_assignment_syfr(rewardMatrix)
            end
        else
            error("algorithm = $algorithm not supported set algorithm to one of 'as', 'asfr1', 'asfr2' or 'syfr'")
        end
        return astate, lambda
    end
end
