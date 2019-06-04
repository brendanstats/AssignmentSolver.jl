"""
    forward_iteration!(openRows::Queue{G},
                       r2c::Array{G, 1}, c2r::Array{G, 1},
                       rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                       lambda::T, epsi::T,
                       trewardMatrix::Array{T, 2}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}

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
                            rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                            lambda::T, epsi::T,
                            rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}

    ##Choose unassigned row to attempt to match
    row = dequeue!(openRows)

    ##Row may have been assigned in a reverse iteration, assumes another free row since otherwise all rows are already assigned...
    if !iszero(r2c[row])
        return forward_iteration!(openRows, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
    end
    
    ##Find maximum reward - object price
    maxcol, maxval, maxtwo = maxtwo_row(row, colPrices, rewardMatrix)

    ##Compute useful values
    twoMinusEpsilon = maxtwo - epsi
    bid = rewardMatrix[row, maxcol] - twoMinusEpsilon

    ##If bid >= lambda add assignment
    if bid >= lambda
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
    else
        colPrices[maxcol] = lambda
        enqueue!(openRows, row)
        addassign = false
    end

    ##Set column cost
    rowPrices[row] = twoMinusEpsilon
    
    return addassign, openRows, r2c, c2r, rowPrices, colPrices
end

"""
    reverse_iteration!(openColsAbove::Queue{G},
                       r2c::Array{G, 1}, c2r::Array{G, 1},
                       rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                       lambda::T, epsi::T,
                       trewardMatrix::Array{T, 2}, dfltTwo::T) where {G <: Integer, T <: AbstractFloat}

### Arguments

* `openColsAbove` : Queue containing all currently unassigned columns with columns
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

    `(addassign::Bool, openColsAbove::Queue{G}, r2c::Array{G, 1}, c2r::Array{G, 1},
     rowPrices::Array{T, 1}, colPrices::Array{T, 1})`

* `addassign` : Indicator for whether an additional (net) assignment has been added
* `openColsAbove` : Updated queue of unassigned columns with prices greater than
    lambda
* `r2c` : Updated assignments of all rows
* `c2r` : Updated assignments of all columns
* `rowPrices` : Updated prices associated with each row
* `colPrices` : Updated prices associated with each column

### Examples

```julia

```
"""
function reverse_iteration!(openColsAbove::Queue{G},
                            r2c::Array{G, 1}, c2r::Array{G, 1},
                            rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                            lambda::T, epsi::T,
                            rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}

    ##Choose unassigned col to attempt to match, needs to have price > lambda
    col = dequeue!(openColsAbove)

    ##col price may not be > lambda since queue is not updated in forward iterations
    if colPrices[col] <= lambda || !iszero(c2r[col])

        ##Recursively call if additional columns above lambda are available
        if length(openColsAbove) > 0
            return reverse_iteration!(openColsAbove, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
        else
            addassign = false
            return addassign, openColsAbove, r2c, c2r, rowPrices, colPrices
        end
    end
    
    ##Find maximum reward - object price
    maxrow, maxval, maxtwo = maxtwo_column(col, rowPrices, rewardMatrix)

    ##Compute useful values
    if maxval >= (lambda + epsi)

        ##Set new row and column costs
        twoMinusEpsilon = maxtwo - epsi
        if twoMinusEpsilon > lambda
            colPrices[col] = twoMinusEpsilon
            rowPrices[maxrow] = rewardMatrix[maxrow, col] - twoMinusEpsilon
        else
            colPrices[col] = lambda
            rowPrices[maxrow] = rewardMatrix[maxrow, col] - lambda
        end

        ##If maxrow already assigned, unassign
        if iszero(r2c[maxrow])
            addassign = true
        else
            prevcol = r2c[maxrow]
            c2r[prevcol] = zero(G)

            ##add back to queue of open columns if price is sufficiently high
            if colPrices[prevcol] > lambda
                enqueue!(openColsAbove, prevcol)
            end
            addassign = false
        end
        
        ##Add assignment
        c2r[col] = maxrow
        r2c[maxrow] = col
    else
        colPrices[col] = maxval - epsi #will be <= lambda so not added to queue
        addassign = false
        #col = enqueue!(openColsLeq)
    end
    #return openColsAbove, openColsLeq, nassigned, r2c, c2r, rowPrices, colPrices
    return addassign, openColsAbove, r2c, c2r, rowPrices, colPrices
end

function forward_iteration_counting!(nbelow::G, openRows::Queue{G},
                                     r2c::Array{G, 1}, c2r::Array{G, 1},
                                     rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                                     lambda::T, epsi::T,
                                     rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}

    ##Choose unassigned row to attempt to match
    row = dequeue!(openRows)

    ##Row may have been assigned in a reverse iteration, assumes another free row since otherwise all rows are already assigned...
    if !iszero(r2c[row])
        return forward_iteration_counting!(nbelow, openRows, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
    end
    
    ##Find maximum reward - object price
    maxcol, maxval, maxtwo = maxtwo_row(row, colPrices, rewardMatrix)

    ##Compute useful values
    twoMinusEpsilon = maxtwo - epsi
    bid = rewardMatrix[row, maxcol] - twoMinusEpsilon

    ##If bid >= lambda add assignment
    if bid >= lambda
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
    else
        #If assigned column price was previously below lambda
        if colPrices[maxcol] < lambda
            nbelow -= one(G)
        end
        colPrices[maxcol] = lambda
        enqueue!(openRows, row)
        addassign = false
    end

    ##Set column cost
    rowPrices[row] = twoMinusEpsilon
    
    return addassign, nbelow, openRows, r2c, c2r, rowPrices, colPrices
end

function reverse_iteration_counting!(nbelow::G, openColsAbove::Queue{G},
                                     r2c::Array{G, 1}, c2r::Array{G, 1},
                                     rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                                     lambda::T, epsi::T,
                                     rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}

    ##Choose unassigned col to attempt to match, needs to have price > lambda
    col = dequeue!(openColsAbove)

    ##col price may not be > lambda since queue is not updated in forward iterations
    if colPrices[col] <= lambda || !iszero(c2r[col])

        ##Recursively call if additional columns above lambda are available
        if length(openColsAbove) > 0
            return reverse_iteration_counting!(nbelow, openColsAbove, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
        else
            addassign = false
            return addassign, nbelow, openColsAbove, r2c, c2r, rowPrices, colPrices
        end
    end
    
    ##Find maximum reward - object price
    maxrow, maxval, maxtwo = maxtwo_column(col, rowPrices, rewardMatrix)

    if maxval >= (lambda + epsi)

        ##Set new row and column costs
        twoMinusEpsilon = maxtwo - epsi
        if twoMinusEpsilon > lambda
            colPrices[col] = twoMinusEpsilon
            rowPrices[maxrow] = rewardMatrix[maxrow, col] - twoMinusEpsilon
        else
            colPrices[col] = lambda
            rowPrices[maxrow] = rewardMatrix[maxrow, col] - lambda
        end

        ##If maxrow already assigned, unassign
        if iszero(r2c[maxrow])
            addassign = true
        else
            prevcol = r2c[maxrow]
            c2r[prevcol] = zero(G)

            ##add back to queue of open columns if price is sufficiently high
            if colPrices[prevcol] > lambda
                enqueue!(openColsAbove, prevcol)
            end
            addassign = false
        end
        
        ##Add assignment
        c2r[col] = maxrow
        r2c[maxrow] = col
    else
        colPrices[col] = maxval - epsi #will be <= lambda so not added to queue
        if colPrices[col] < lambda
            nbelow += one(G)
        end

        addassign = false
        #col = enqueue!(openColsLeq)
    end
    #return openColsAbove, openColsLeq, nassigned, r2c, c2r, rowPrices, colPrices
    return addassign, nbelow, openColsAbove, r2c, c2r, rowPrices, colPrices
end

function scaling_as(r2c::Array{G, 1}, c2r::Array{G, 1},
                    rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                    lambda::T, epsi::T,
                    rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}

    ##Count initially assign rows and add others to Queue
    nassigned, openRows = get_openrows(r2c)
    
    ##Use forward iterations to find feasible assignment
    while nassigned < size(rewardMatrix, 1)
        addassign, openRows, r2c, c2r, rowPrices, colPrices = forward_iteration!(openRows, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
        if addassign
            nassigned += one(G)
        end
    end

    ##Set lambda to minimum assigned prices
    lambda = colPrices[r2c[1]]
    if size(rewardMatrix, 1) > 1
        for ii in 2:size(rewardMatrix, 1)
            if colPrices[r2c[ii]] < lambda
                lambda = colPrices[r2c[ii]]
            end
        end
    end

    openColsAbove = get_opencolsabove(c2r, colPrices, lambda)
    
    ##Execute reverse iterations until termination
    while length(openColsAbove) > 0
        addassign, openColsAbove, r2c, c2r, rowPrices, colPrices = reverse_iteration!(openColsAbove, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
    end
    
    return r2c, c2r, rowPrices, colPrices, lambda
end

function auction_assignment_as(r2c::Array{G, 1}, c2r::Array{G, 1},
                               rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                               rewardMatrix::Array{T, 2},
                               epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.2)) where {G <: Integer, T <: AbstractFloat}

    epsi = epsi0
    lambda = zero(T)

    abovetol = epsi >= epsitol
    while abovetol
        
        ##Count initially assign rows and add others to Queue
        nassigned, openRows = get_openrows(r2c)
        
        ##Use forward iterations to find feasible assignment
        while nassigned < size(rewardMatrix, 1)
            addassign, openRows, r2c, c2r, rowPrices, colPrices = forward_iteration!(openRows, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
            if addassign
                nassigned += one(G)
            end
        end

        if epsi < epsitol
            abovetol = false
        else
            newepsi = epsi * epsiscale
            deltaepsi = epsi - newepsi
            rowPrices .+= deltaepsi #rowPrices are increased so that complimentary slackness is maintained with smaller epsilon
            r2c .= zero(G)
            #c2r .= zero(G)
            for jj in 1:length(c2r)
                if !iszero(c2r[jj])
                    c2r[jj] = zero(G)
                end
            end
            epsi = newepsi
        end
    end
    
    ##Set lambda to minimum assigned prices
    lambda = colPrices[r2c[1]]
    if size(rewardMatrix, 1) > 1
        for ii in 2:size(rewardMatrix, 1)
            if colPrices[r2c[ii]] < lambda
                lambda = colPrices[r2c[ii]]
            end
        end
    end

    openColsAbove = get_opencolsabove(c2r, colPrices, lambda)
    
    ##Execute reverse iterations until termination
    while length(openColsAbove) > 0
        addassign, openColsAbove, r2c, c2r, rowPrices, colPrices = reverse_iteration!(openColsAbove, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
    end
    
    return r2c, c2r, rowPrices, colPrices, lambda
end

function auction_assignment_as(rewardMatrix::Array{T, 2},
                               epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.2)) where T <: AbstractFloat
    r2c = zeros(Int, size(rewardMatrix, 1))
    c2r = zeros(Int, size(rewardMatrix, 2))
    rowPrices = vec(maximum(rewardMatrix, dims = 2))
    colPrices = zeros(T, size(rewardMatrix, 2))
    return auction_assignment_as(r2c, c2r, rowPrices, colPrices, rewardMatrix, epsi0, epsitol, epsiscale)
end

#Forward Reverse Algorithm
#    1) Forward iterations until additional person assigned
#    2) Reverse iterations until additional person assigned or until p <= lambda for all unassigned objects
#    3) Reverse iterations until p <= lambda for all unassigned objects
# Track unassigned rows and columns
# *adjust lambda throughout

function scaling_asfr1(r2c::Array{G, 1}, c2r::Array{G, 1},
                       rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                       lambda::T, epsi::T,
                       rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}
    nexcesscols = size(rewardMatrix, 2) - size(rewardMatrix, 1)


    ##Count initially assign rows and add others to Queue
    nassigned, openRows = get_openrows(r2c)
    nbelow, openColsAbove = get_nbelow_opencolsabove(c2r, colPrices, lambda)
    
    while nassigned < size(rewardMatrix, 1) || length(openColsAbove) > 0

        ##Only execute forward iteration if not all rows are assigned
        if nassigned < size(rewardMatrix, 1)
            #println("Forward Iteration: $nassigned assigned")
            addassign = false
            while !addassign
                addassign, nbelow, openRows, r2c, c2r, rowPrices, colPrices = forward_iteration_counting!(nbelow, openRows, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
            end
            nassigned += one(G)
        end

        ##Only execute reverse iteration if open columns above lambda
        addassign = false
        while !addassign && length(openColsAbove) > 0
            #println("Reverse Iteration: $nassigned assigned")
            addassign, nbelow, openColsAbove, r2c, c2r, rowPrices, colPrices = reverse_iteration_counting!(nbelow, openColsAbove, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
        end
        if addassign
            nassigned += one(G)
        end

        ## Adjust lambda after reverse iterations if necessary
        if nbelow > nexcesscols
            colperm = sortperm(colPrices)
            newlambda = colPrices[colperm[nexcesscols]]

            #find column prices above new lambda value but <= old lambda value and add to set of columns that can be assigned
            jj = nexcesscols + 1
            while (jj <= size(rewardMatrix, 2)) && (colPrices[colperm[jj]] <= lambda)
                if colPrices[colperm[jj]] > newlambda
                    enqueue!(openColsAbove, colperm[jj])
                end
                jj += 1
            end
            lambda = newlambda
        end
    end

    return r2c, c2r, rowPrices, colPrices, lambda
end

function auction_assignment_asfr1(r2c::Array{G, 1}, c2r::Array{G, 1},
                                  rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                                  rewardMatrix::Array{T, 2},
                                  lambda0::T, epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1)) where {G <: Integer, T <: AbstractFloat}

    epsi = epsi0
    lambda = lambda0

    abovetol = epsi >= epsitol
    while abovetol
        
         r2c, c2r, rowPrices, colPrices, lambda = scaling_asfr1(r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
        
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
            #c2r .= zero(G)
            epsi = newepsi
        end
    end
    return r2c, c2r, rowPrices, colPrices, lambda
end

function auction_assignment_asfr1(rewardMatrix::Array{T, 2},
                                  lambda0::T = -one(T), epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1)) where T <: AbstractFloat
    r2c = zeros(Int, size(rewardMatrix, 1))
    c2r = zeros(Int, size(rewardMatrix, 2))
    rowPrices, colPrices = dimmaximums(rewardMatrix)
    rowPrices .-= minimum(colPrices)
    if lambda0 < zero(T) && (size(rewardMatrix, 2) - size(rewardMatrix, 1)) > 0
        lambda0 = sort(colPrices)[size(rewardMatrix, 2) - size(rewardMatrix, 1)]
    end
    return auction_assignment_asfr1(r2c, c2r, rowPrices, colPrices, rewardMatrix, lambda0, epsi0, epsitol, epsiscale)
end

#Forward Reverse Algorithm, fixed lambda
#Purely Reverse Algorithm
#    1) Forward Algorithm until assignment is feasible (everything assigned)
#    2) Set lambda to min assigned object prices
#    3) Run reverse algorithm until termination
# *Track unassigned rows and columns

function scaling_asfr2(r2c::Array{G, 1}, c2r::Array{G, 1},
                       rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                       lambda::T, epsi::T,
                       rewardMatrix::Array{T, 2}) where {G <: Integer, T <: AbstractFloat}

    ##Count initially assign rows and add others to Queue
    nassigned, openRows = get_openrows(r2c)
    openColsAbove = get_opencolsabove(c2r, colPrices, lambda)
    
    while nassigned < size(rewardMatrix, 1) || length(openColsAbove) > 0

        ##Only execute forward iteration if not all rows are assigned
        if nassigned < size(rewardMatrix, 1)
            #println("Forward Iteration: $nassigned assigned")
            addassign = false
            while !addassign
                addassign, openRows, r2c, c2r, rowPrices, colPrices = forward_iteration!(openRows, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
            end
            nassigned += one(G)
        end

        ##Only execute reverse iteration if open columns above lambda
        addassign = false
        while !addassign && length(openColsAbove) > 0
            #println("Forward Iteration: $nassigned assigned")
            addassign, openColsAbove, r2c, c2r, rowPrices, colPrices = reverse_iteration!(openColsAbove, r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
        end
        if addassign
            nassigned += one(G)
        end
    end

    ##lambda returned for consistency with other implimentations
    return r2c, c2r, rowPrices, colPrices, lambda
end

function auction_assignment_asfr2(r2c::Array{G, 1}, c2r::Array{G, 1},
                                  rowPrices::Array{T, 1}, colPrices::Array{T, 1},
                                  rewardMatrix::Array{T, 2},
                                  epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1)) where {G <: Integer, T <: AbstractFloat}

    epsi = epsi0
    lambda = zero(T)

    abovetol = epsi >= epsitol
    while abovetol
        
         r2c, c2r, rowPrices, colPrices, lambda = scaling_asfr2(r2c, c2r, rowPrices, colPrices, lambda, epsi, rewardMatrix)
        
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
            #c2r .= zero(G)
            epsi = newepsi
        end
    end
    return r2c, c2r, rowPrices, colPrices, lambda
end

function auction_assignment_asfr2(rewardMatrix::Array{T, 2},
                                  epsi0::T = one(T), epsitol::T = T(one(T) / size(rewardMatrix, 1)), epsiscale::T = T(0.1)) where T <: AbstractFloat
    r2c = zeros(Int, size(rewardMatrix, 1))
    c2r = zeros(Int, size(rewardMatrix, 2))
    rowPrices = vec(maximum(rewardMatrix, dims = 2))
    colPrices = zeros(T, size(rewardMatrix, 2))
    return auction_assignment_asfr2(r2c, c2r, rowPrices, colPrices, rewardMatrix, epsi0, epsitol, epsiscale)
end
