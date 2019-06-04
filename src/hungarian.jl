#using DataStructures

"""
    lsap_solver_tracking(costMatrix::Array{G<:Real, 2}; verbose::Bool = false)

### Arguments

* `costMatrix` : `Float` or `Integer` valued cost matrix for which a maximal assignment is found
* `verbose` : Optional `Bool` parameter controling if step and iteration should be reported

### Details

Based on Hungarian algorithm described by Bob Pilgrim at 
http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html Returns maximal
assignment for supplied cost Matrix.

### Value

    (starredRow2Col, rowOffsets, colOffsets)

Where `starredRow2Col` contrains the column assignment for each row, note that if
the number of rows in `costMatrix` is greater than the number of columns then some
rows will not be assigned and `starredRow2Col` will contain zeros for unassigned
rows.  `rowOffsets` and `colOffsets` are the dual variables corresponding to the
solution found.  This version tracks the zeros in the adjusted cost matrix instead
of re-computting them.  This reduces issue with numerical precision when a
non-integer cost matrix is supplied.

### Examples

```julia

```
"""
function lsap_solver_tracking(costMatrix::Array{T, 2}; verbose::Bool = false) where T <: Real
    
    ##Flip if more rows than columns
    if size(costMatrix, 1) > size(costMatrix, 2)
        
        ##Find optimal assignment for transpose
        astate = lsap_solver_tracking(costMatrix')
        
        return flip(astate)
    end
    
    ##Define algorithm variables
    #astate = AssignmentState(costMatrix)
    astate = AssignmentState(costMatrix, maximize = false, assign = true, pad = false)
    rowCover = falses(astate.nrow) #row covered true/false
    colCover = falses(astate.ncol) #column covered true/false
    primedRow2Col = zeros(Int, astate.nrow) #either zero or column index of primed zero
    minval = typemax(T) #tracking minimum value
    zeroCol2Row = Dict{Int, Array{Int, 1}}()
    for jj in 1:astate.ncol
        zeroCol2Row[jj] = Int[]
    end
    minPoints = Array{Tuple{Int, Int}, 1}(undef, 0)
    colsUncovered = Array{Int}(undef, 0)
    rowsUncovered = Array{Int}(undef, 0)
    #for ii in 1:astate.nrow
    #    astate.rowPrices[ii] = costMatrix[ii, one(Int)]
    #    astate.r2c[ii] = one(Int)
    #end
    
    ##Find row minimums
    #for jj in 1:astate.ncol, ii in 1:astate.nrow
    #    if costMatrix[ii, jj] < astate.rowPrices[ii]
    #        astate.rowPrices[ii] = costMatrix[ii, jj]
    #        astate.r2c[ii] = jj #track row minimums for the moment
    #    end
    #end

    ##Find all zeros
    for jj in 1:astate.ncol, ii in 1:astate.nrow
        if costMatrix[ii, jj] <= astate.rowPrices[ii]
            push!(zeroCol2Row[jj], ii)
        end
    end

    ##Cover rows and columns
    for ii in 1:astate.nrow
        if !iszero(astate.r2c[ii])
    #    if !colCover[astate.r2c[ii]]
    #        astate.c2r[astate.r2c[ii]] = ii
            rowCover[ii] = true
            colCover[astate.r2c[ii]] = true
    #    else
    #        astate.r2c[ii] = zero(Int)
            #    end
        end
    end

    ##Check remaining zeros
    for jj in 1:astate.ncol
        for ii in zeroCol2Row[jj]
            if !rowCover[ii] && !colCover[jj]
                rowCover[ii] = true
                colCover[jj] = true
                astate.r2c[ii] = jj
                astate.c2r[jj] = ii
            end
        end
        
        ##Add still uncovered columns to column ids
        if !colCover[jj]
            push!(colsUncovered, jj)
        end
    end

    ##Check if initial assignment is optimal
    if sum(colCover) == astate.nrow
        nstep = 7
    else
        nstep = 4
    end

    ##Uncover all rows
    for ii in 1:astate.nrow
        push!(rowsUncovered, ii)
        if rowCover[ii]
            rowCover[ii] = false
        end
    end

    iter = 0
    while true
        iter += 1
        if verbose
            println("Iteration: $iter")
            println("step = ", nstep)
        end
        if nstep == 3
            nstep = step3_tracking!(colCover, astate, colsUncovered)
        elseif nstep == 4
            nstep, minval, minPoints = step4_tracking!(costMatrix,
                                                       astate,
                                                       rowCover, colCover,
                                                       primedRow2Col,
                                                       zeroCol2Row,
                                                       minval, minPoints,
                                                       colsUncovered, rowsUncovered)
        elseif nstep == 5
            nstep = step5_tracking!(rowCover, colCover,
                           astate,
                           primedRow2Col,
                           minPoints, rowsUncovered)
        elseif nstep == 6
            nstep = step6_tracking!(astate, rowCover, colCover,
                           zeroCol2Row, minval, minPoints)
        elseif nstep == 7
            break
        else
            error("Non-listed step introduced")
        end
        if verbose
            ctrow = count(rowCover)
            ctcol = count(colCover)
            ct = astate.nrow - count(iszero.(astate.r2c))
            if length(minPoints) > 1
                println("Value: $minval, Assigned: $ct, Covered Rows: $ctrow, Covered Cols: $ctcol")
            else
                println("Value: $minval, Assigned: $ct, Covered Rows: $ctrow, Covered Cols: $ctcol, Point: $minPoints")
            end
            println("")
        end
    end

    return astate
end

"""
    lsap_solver_tracking!(costMatrix::Array{G, 2},
                          astate.rowPrices::Array{G, 1},
                          astate.colPrices::Array{G, 1},
                          rowInitial::Array{Int, 1} = zeros(Int, size(costMatrix, 1));
                          check::Bool = true,
                          verbose::Bool = false) where G <: Real
"""
function lsap_solver_tracking!(costMatrix::Array{T, 2},
                               astate::AssignmentState{G, T};
                               check::Bool = true,
                               verbose::Bool = false) where {G <: Integer, T <: Real}
    
    ##Flip if more rows than columns
    if size(costMatrix, 1) > size(costMatrix, 2)

        ##Switch initial assignment from rows map to column

        ##Find optimal assignment for transpose
        astate = lsap_solver_tracking(costMatrix', flip(astate), check = check, verbose = verbose)
        
        return flip(astate)
    end
    
    ##Define algorithm variables
    #astate.nrow, astate.ncol = size(costMatrix) #set matrix dimensions
    rowCover = falses(astate.nrow) #row covered true/false
    colCover = falses(astate.ncol) #column covered true/false
    #astate.r2c = zeros(Int, astate.nrow) #either zero or column index of starred zero
    #astate.c2r = zeros(Int, astate.ncol) #either zero or row index of starred zero
    primedRow2Col = zeros(Int, astate.nrow) #either zero or column index of primed zero
    minval = typemax(T)::T #tracking minimum value
    zeroCol2Row = Dict{Int, Array{Int, 1}}()
    for jj in 1:astate.ncol
        zeroCol2Row[jj] = Int[]
    end
    minPoints = Array{Tuple{Int, Int}, 1}(undef, 0)
    colsUncovered = Array{Int}(undef, 0)
    rowsUncovered = Array{Int}(undef, 0)
    
    ##Find row minimums
    for jj in 1:astate.ncol, ii in 1:astate.nrow
        if costMatrix[ii, jj] < astate.rowPrices[ii]
            astate.rowPrices[ii] = costMatrix[ii, jj]
        end
    end
    
    ##Check that the row and column offsets are not too high at any point
    if check
        for jj in 1:astate.ncol, ii in 1:astate.nrow
            if costMatrix[ii, jj] < (astate.rowPrices[ii] + astate.colPrices[jj])
                astate.colPrices[jj] = costMatrix[ii, jj] - astate.rowPrices[ii]
            end
        end
    end
    
    ##Find all zeros
    for jj in 1:astate.ncol, ii in 1:astate.nrow
        if zero_cost(ii, jj, costMatrix, astate.rowPrices, astate.colPrices)
            push!(zeroCol2Row[jj], ii)
        end
    end

    ##Initialize with supplied assignments, checking that they still meet assignment criteria
    for (ii, jj) in enumerate(IndexLinear(), astate.r2c)
        if !iszero(jj) && !colCover[jj]
            if zero_cost(ii, jj, costMatrix, astate)
                rowCover[ii] = true
                colCover[jj] = true
                astate.r2c[ii] = jj
                astate.c2r[jj] = ii
            end
        end
    end
    
    ##Check remaining zeros
    for jj in 1:astate.ncol
        for ii in zeroCol2Row[jj]
            if !rowCover[ii] && !colCover[jj]
                rowCover[ii] = true
                colCover[jj] = true
                astate.r2c[ii] = jj
                astate.c2r[jj] = ii
            end
        end
        
        ##Add still uncovered columns to column ids
        if !colCover[jj]
            push!(colsUncovered, jj)
        end
    end

    ##Check if initial assignment is optimal
    astate.nassigned = count(colCover)
    if astate.nassigned == astate.nrow
        nstep = 7
    else
        nstep = 4
    end

    ##Uncover all rows
    for ii in 1:astate.nrow
        push!(rowsUncovered, ii)
        if rowCover[ii]
            rowCover[ii] = false
        end
    end

    iter = 0
    while true
        iter += 1
        if verbose
            println("Iteration: $iter")
            println("step = ", nstep)
        end
        if nstep == 3
            nstep = step3_tracking!(colCover, astate, colsUncovered)
        elseif nstep == 4
            nstep, minval, minPoints = step4_tracking!(costMatrix,
                                                       astate,
                                                       rowCover, colCover,
                                                       primedRow2Col,
                                                       zeroCol2Row,
                                                       minval, minPoints,
                                                       colsUncovered, rowsUncovered)
        elseif nstep == 5
            nstep = step5_tracking!(rowCover, colCover,
                           astate,
                           primedRow2Col,
                           minPoints, rowsUncovered)
        elseif nstep == 6
            nstep = step6_tracking!(astate,
                           rowCover, colCover,
                           zeroCol2Row, minval, minPoints)
        elseif nstep == 7
            break
        else
            error("Non-listed step introduced")
        end
        if verbose
            ctrow = count(rowCover)
            ctcol = count(colCover)
            ct = astate.nrow - count(iszero.(astate.r2c))
            if length(minPoints) > 1
                println("Value: $minval, Assigned: $ct, Covered Rows: $ctrow, Covered Cols: $ctcol")
            else
                println("Value: $minval, Assigned: $ct, Covered Rows: $ctrow, Covered Cols: $ctcol, Point: $minPoints")
            end
            println("")
        end
    end

    return astate
end

function lsap_solver_tracking(costMatrix::Array{T, 2},
                              astate::AssignmentState{G, T};
                              check::Bool = true,
                              verbose::Bool = false) where {G <: Integer, T <: Real}
    
    return lsap_solver_tracking!(costMatrix, deepcopy(astate), check = check, verbose = verbose)
end



"""
    step3!(colCover, astate.r2c, astate.nrow, m) -> step 3

Internal function for assignment solver, returns number of the next step in the
algorithm.  Returns 7 (terminates algorithm) if the number of covered columns is
equal to the number of rows, otherwise returns 4
"""
function step3_tracking!(colCover::BitArray{1},
                astate::AssignmentState{G, T},
                colsUncovered::Array{Int, 1}) where {G <: Integer, T <: Real}
    
    ##Non-zero values for astate.r2c are columns with starred zeros
    #switch to updating colsUncovered
    jj = 1
    while jj <= length(colsUncovered)
        if iszero(astate.c2r[colsUncovered[jj]]) #!= zero(T)
            jj += 1
        else
            colCover[colsUncovered[jj]] = true
            deleteat!(colsUncovered, jj)
        end
    end

    ##Check if done, otherwise go to step 4
    #switch to length(colsUncovered)
    astate.nassigned = count(colCover)
    if astate.nassigned == astate.nrow
        return 7
    else
        return 4
    end
end

"""
    step4!(costMatrix, astate.rowPrices, astate.colPrices, rowCover, colCover,
           astate.r2c, astate.c2r, primedRow2Col,
           minval, minrow, mincol, astate.nrow, m) -> step, minval, minrow, mincol

Find 0's where cost[ii, jj] == astate.rowPrices[ii] + astate.colPrices[jj].  If
cover row ii and column jj are false set them to true and star 0.
When staring add to astate.r2c and astate.c2r
"""
function step4_tracking!(costMatrix::Array{T, 2},
                         astate::AssignmentState{G, T},
                         rowCover::BitArray{1},
                         colCover::BitArray{1},
                         primedRow2Col::Array{Int, 1},
                         zeroCol2Row::Dict{Int, Array{Int, 1}},
                         minval::T,
                         minPoints::Array{Tuple{Int, Int}, 1},
                         colsUncovered::Array{Int, 1}, rowsUncovered::Array{Int, 1}) where {G <: Integer, T <: Real}

    ##value defaults
    empty!(minPoints)
    minval = typemax(T)::T
    
    ##loop popping items off queue so that uncovered columns can be added
    col = one(G)
    while col <= length(colsUncovered)
        jj = colsUncovered[col]

        for row in zeroCol2Row[jj]
            if rowCover[row]
                continue
            else
                primedRow2Col[row] = jj
                if iszero(astate.r2c[row])
                    push!(minPoints, (row, jj))
                    return 5, minval, minPoints
                else
                    rowCover[row] = true
                    colCover[astate.r2c[row]] = false
                    push!(colsUncovered, astate.r2c[row])
                end
            end
        end #end for
        col += 1
    end #end while

    ii = 1
    while ii <= length(rowsUncovered)
        if rowCover[rowsUncovered[ii]]
            deleteat!(rowsUncovered, ii)
        else
            ii += 1
        end
    end

    ##Find minimum value
    for jj in colsUncovered
        for ii in rowsUncovered
            val = adjusted_cost(ii, jj, costMatrix, astate)
            if  val < minval
                if iszero(val)
                    push!(zeroCol2Row[jj], ii)
                    return 4, minval, minPoints
                end
                empty!(minPoints)
                minval = val
                push!(minPoints, (ii, jj))
            elseif val == minval
                push!(minPoints, (ii, jj))
            end
        end
    end
    
    ##If min value is negative warn
    if iszero(minval)
        warn("minvalue is zero")
    end
    return 6, minval, minPoints
end

"""
    step5!(rowCover, colCover, astate.r2c, astate.c2r,
           primedRow2Col, minrow, mincol, astate.nrow, m) -> 3

Find 0's where cost[ii, jj] == astate.rowPrices[ii] + astate.colPrices[jj].  If
cover row ii and column jj are false set them to true and star 0.
When staring add to astate.r2c and astate.c2r
"""
function step5_tracking!(rowCover::BitArray{1}, colCover::BitArray{1},
                         astate::AssignmentState{G, T},
                         primedRow2Col::Array{Int, 1},
                         minPoints::Array{Tuple{Int, Int}, 1},
                         rowsUncovered::Array{Int, 1}) where {G <: Integer, T <: Real} 

    ##initialize array for tracking sequence, alternating primed and starred
    primedRows = Int[minPoints[1][1]]
    primedCols = Int[minPoints[1][2]]
    starredRows = Int[]
    starredCols = Int[]

    ##If no starred zero in row terminate, otherwise add starred zero and the primed
    ##zero in the column of the starred zero.  Continue until terminations...
    while true
        row = astate.c2r[primedCols[end]]
        if !iszero(row) # != zero(T)
            push!(starredRows, row)
            push!(starredCols, primedCols[end])
            push!(primedRows, row)
            push!(primedCols, primedRow2Col[row])
        else
            break
        end
    end
    
    ##Unstar each starred of the series, star each primed zero of the series,
    ##every starred has a primed in the same row, move starred to column corresponding
    ##primed column
    for (row, col) in zip(starredRows, starredCols)
        astate.r2c[row] = primedRow2Col[row]
        astate.c2r[primedRow2Col[row]] = row
    end

    ##Initial primed zero has no corresponding starred zero in column so star it
    astate.r2c[minPoints[1][1]] = minPoints[1][2]::Int
    astate.c2r[minPoints[1][2]] = minPoints[1][1]::Int
    
    ##Erase all primes
    primedRow2Col[:] .= zero(G)
    
    ##uncover every line in the matrix
    for ii in 1:astate.nrow
        if rowCover[ii]
            rowCover[ii] = false
            push!(rowsUncovered, ii)
        end
    end

    return 3
end

"""
    step6!(astate.rowPrices, astate.colPrices, rowCover, colCover, minval) -> 4


"""
function step6_tracking!(astate::AssignmentState{G, T},
                         rowCover::BitArray{1},
                         colCover::BitArray{1},
                         zeroCol2Row::Dict{Int, Array{Int, 1}},
                         minval::T,
                         minPoints::Array{Tuple{Int, Int}, 1}) where {G <: Integer, T <: Real}
    ##Add min to (subtract from offset) all elements in covered rows
    for ii in 1:astate.nrow
        if rowCover[ii]
            astate.rowPrices[ii] -= minval
        end
    end
    
    ##Subtract min from (add to offset) all elements in uncovered columns
    for jj in 1:astate.ncol
        if !colCover[jj]
            astate.colPrices[jj] += minval
        end
    end

    #Remove zeros that where rowCover and colCover are both covered
    for jj in 1:astate.ncol
        if colCover[jj]
            rows = zeroCol2Row[jj]
            if length(rows) > 0
                ridx = 1
                while ridx <= length(rows)
                    if rowCover[rows[ridx]]
                        deleteat!(rows, ridx)
                    else
                        ridx += 1
                    end
                end
            end
        end
    end

    #Set minPoints to zeros
    for (ii, jj) in minPoints
        push!(zeroCol2Row[jj], ii)
    end
    
    return 4
end
