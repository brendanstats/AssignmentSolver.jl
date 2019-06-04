"""
    forward_iteration!(openRows::Queue{G},
                       r2c::Array{G, 1}, c2r::Array{G, 1},
                       rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                       lambda::T, epsi::T,
                       trewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}

### Arguments

* `openRows` : Queue containing all currently unassigned rows, algorithm is
    robust to containing already assigned rows
* `r2c` : Array where `r2c[ii]` contains the column row `ii` is assigned
    to zero if row `ii` is currently unasssigned
* `c2r` : Inverse of `r2c`
* `rowPrices` : Prices associated with each row in auction assignment algorithm
* `colPrices` : Prices associated with each column in auction assignment algorithm
* `lambda` : Lambda parameter in auction assignment algorithm separating prices of
    column which are or can be assigned from those which cannot (currently)
* `epsi` : Epsilon parameter for epsilon-complimentary slackness condition mantained
* `trewardMatrix` : Transpose of reward matrix which assignment is attempting
     to maximize
* `dfltTwo` : Default value for the second largest value to be returned if only
              one allowed assignment exists

### Details

Performs a forward iteration of the auction assigment algorithm where the
algorithm attempts to assign an additional row to the best available column, the
column can be previously either assigned or unassigned.  In the case where the
column is already assigned the previous assignment is removed.

### Value

    `(addassign::Bool, openRows::Queue{G}, r2c::Array{G, 1}, c2r::Array{G, 1},
    rowPrices::Array{T, 1}, colPrices::Array{T, 1})`

* `addassign` : Indicator for whether an additional (net) assignment has been added
* `openRows` : Updated queue of unassigned rows
* `r2c` : Updated assignments of all rows
* `c2r` : Updated assignments of all columns
* `rowPrices` : Updated prices associated with each row
* `colPrices` : Updated prices associated with each column

### Examples

```julia

```
"""
function forward_iteration!(openRows::Queue{G},
                            r2c::Array{G, 1}, c2r::Array{G, 1},
                            rowPrices::Array{T, 1}, colPrices::Array{T, 1}, epsi::T,
                            trewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}

    ##Choose unassigned row to attempt to match
    row = dequeue!(openRows)

    ##Row may have been assigned in a reverse iteration, assumes another free row since otherwise all rows are already assigned...
    if !iszero(r2c[row])
        return forward_iteration!(openRows, r2c, c2r, rowPrices, colPrices, epsi, trewardMatrix, dfltTwo)
    end
    
    ##Find maximum reward - object price
    maxidx, maxval, maxtwo = maxtwo_column(row, colPrices, trewardMatrix, dfltTwo)

    if maxidx == 0
        println(row)
        println(rowPrices)
        println(colPrices)
        println(trewardMatrix[:, row])
        println(count(iszero.(r2c)))
        println(findall(x -> iszero(x), r2c))
        println(epsi)
    end
    
    ##Compute useful values
    maxcol = trewardMatrix.rowval[maxidx]
    twoMinusEpsilon = maxtwo - epsi
    bid = trewardMatrix.nzval[maxidx] - twoMinusEpsilon

    ##Assign new row and column costs
    rowPrices[row] = twoMinusEpsilon
    colPrices[maxcol] = bid

    ##Check if column is already assigned, unassign if necessary
    if iszero(c2r[maxcol])
        addassign = true
    else
        prevrow = c2r[maxcol]
        r2c[prevrow] = zero(G)
        
        ##add row to queue of unassigned rows
        enqueue!(openRows, prevrow)
        addassign = false
    end

    ##Assign row to maxcol
    r2c[row] = maxcol
    c2r[maxcol] = row
    
    return addassign, openRows, r2c, c2r, rowPrices, colPrices
end

"""
    reverse_iteration!(openCols::Queue{G},
                       r2c::Array{G, 1}, c2r::Array{G, 1},
                       rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                       lambda::T, epsi::T,
                       trewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}

### Arguments

* `openCols` : Queue containing all currently unassigned columns with columns
    prices greater than lambda.  Algorithm is robust to containing already
    assigned columns or columns which prices not greater than lambda
* `r2c` : Array where `r2c[ii]` contains the column row `ii` is assigned
    to zero if row `ii` is currently unasssigned
* `c2r` : Inverse of `r2c`
* `rowPrices` : Prices associated with each row in auction assignment algorithm
* `colPrices` : Prices associated with each column in auction assignment algorithm
* `lambda` : Lambda parameter in auction assignment algorithm separating prices of
    column which are or can be assigned from those which cannot (currently)
* `epsi` : Epsilon parameter for epsilon-complimentary slackness condition mantained
* `rewardMatrix` : Reward matrix which assignment is attempting to maximize
* `dfltTwo` : Default value for the second largest value to be returned if only
              one allowed assignment exists

where `G <: Integer` `T <: AbstractFloat`

### Details

Performs a forward iteration of the auction assigment algorithm where the
algorithm attempts to assign an additional row to the best available column, the
column can be previously either assigned or unassigned.  In the case where the
column is already assigned the previous assignment is removed.

### Value

    `(addassign::Bool, openCols::Queue{G}, r2c::Array{G, 1}, c2r::Array{G, 1},
     rowPrices::Array{T, 1}, colPrices::Array{T, 1})`

* `addassign` : Indicator for whether an additional (net) assignment has been added
* `openCols` : Updated queue of unassigned columns with prices greater than
    lambda
* `r2c` : Updated assignments of all rows
* `c2r` : Updated assignments of all columns
* `rowPrices` : Updated prices associated with each row
* `colPrices` : Updated prices associated with each column

### Examples

```julia

```
"""
function reverse_iteration!(openCols::Queue{G},
                            r2c::Array{G, 1}, c2r::Array{G, 1},
                            rowPrices::Array{T, 1}, colPrices::Array{T, 1}, epsi::T,
                            rewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}

    ##Choose unassigned col to attempt to match, needs to have price > lambda
    col = dequeue!(openCols)

    ##col price may not be > lambda since queue is not updated in forward iterations
    if !iszero(c2r[col])
        return reverse_iteration!(openCols, r2c, c2r, rowPrices, colPrices, epsi, rewardMatrix, dfltTwo)
    end
    
    ##Find maximum reward - object price
    maxidx, maxval, maxtwo = maxtwo_column(col, rowPrices, rewardMatrix, dfltTwo)
    
    if maxidx == 0
        println(col)
        println(rowPrices)
        println(colPrices)
        println(rewardMatrix[:, col])
        println(count(iszero.(r2c)))
        println(findall(x -> iszero(x), r2c))
        println(epsi)
    end
    
    ##Compute useful values
    maxrow = rewardMatrix.rowval[maxidx]
    twoMinusEpsilon = maxtwo - epsi
    bid = rewardMatrix.nzval[maxidx] - twoMinusEpsilon
    
    ##Set new row and column costs
    rowPrices[maxrow] = bid
    colPrices[col] = twoMinusEpsilon
        
    ##If maxrow already assigned, unassign
    if iszero(r2c[maxrow])
        addassign = true
    else
        prevcol = r2c[maxrow]
        c2r[prevcol] = zero(G)

        ##add column to queue of unassigned columns
        enqueue!(openCols, prevcol)
        addassign = false
    end
    
    ##Add assignment
    r2c[maxrow] = col
    c2r[col] = maxrow
    
    return addassign, openCols, r2c, c2r, rowPrices, colPrices
end

function scaling_syfr(r2c::Array{G, 1}, c2r::Array{G, 1},
                      rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                      epsi::T, rewardMatrix::SparseMatrixCSC{T, G},
                      trewardMatrix::SparseMatrixCSC{T, G}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}

    ##Count initially assign rows and add others to Queue
    nassigned, openRows = get_openrows(r2c)
    openCols = get_openrows(c2r)[2]
    
    ##Use forward iterations to find feasible assignment
    while nassigned < rewardMatrix.m

        #println("Forward Iteration: $nassigned assigned")
        addassign = false
        while !addassign
            addassign, openRows, r2c, c2r, rowPrices, colPrices = forward_iteration!(openRows, r2c, c2r, rowPrices, colPrices, epsi, trewardMatrix, dfltTwo)
        end
        if addassign
            nassigned += one(G)
        end

        ##Only execute reverse iteration if open columns
        if nassigned < rewardMatrix.m
            addassign = false
        end
        
        while !addassign
            #println("Reverse Iteration: $nassigned assigned")
            addassign, openCols, r2c, c2r, rowPrices, colPrices = reverse_iteration!(openCols, r2c, c2r, rowPrices, colPrices, epsi, rewardMatrix, dfltTwo)
        end
        if addassign
            nassigned += one(G)
        end
    end
    
    return r2c, c2r, rowPrices, colPrices
end



function auction_assignment_syfr(r2c::Array{G, 1}, c2r::Array{G, 1},
                                 rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                                 rewardMatrix::SparseMatrixCSC{T, G},
                                 epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                  dfltTwo::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    trewardMatrix = permutedims(rewardMatrix, [2, 1])
    epsi = epsi0

    abovetol = epsi >= epsitol
    while abovetol
        
         r2c, c2r, rowPrices, colPrices = scaling_syfr(r2c, c2r, rowPrices, colPrices, epsi, rewardMatrix, trewardMatrix, dfltTwo)
        
        if epsi < epsitol
            abovetol = false
        else
            newepsi = epsi * epsiscale
            deltaepsi = epsi - newepsi
            rowPrices .+= deltaepsi #rowPrices are increased so that complimentary slackness is maintained with smaller epsilon
            r2c .= zero(G)
            for jj in 1:length(c2r)
                if !iszero(c2r[jj])
                    c2r[jj] = zero(G)
                end
            end
            epsi = newepsi
        end
    end
    return r2c, c2r, rowPrices, colPrices
end

function auction_assignment_syfr(rewardMatrix::SparseMatrixCSC{T, G},
                                 epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1),
                                  dfltTwo::T = -T(Inf)) where {G <: Integer, T <: AbstractFloat}
    r2c = zeros(G, size(rewardMatrix, 1))
    c2r = zeros(G, size(rewardMatrix, 2))
    rowPrices, colPrices = dimmaximums(rewardMatrix)
    rowPrices .-= minimum(colPrices)
    return auction_assignment_syfr(r2c, c2r, rowPrices, colPrices, rewardMatrix, epsi0, epsitol, epsiscale, dfltTwo)
end
