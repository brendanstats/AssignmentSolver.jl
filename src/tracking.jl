#using DataStructures
#http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html

function lsap_solver_tracking{G <: Real}(costMatrix::Array{G, 2};
                                         verbose::Bool = false)
    
    ##Flip if more rows than columns
    if size(costMatrix, 1) > size(costMatrix, 2)
        
        ##Find optimal assignment for transpose
        colAssignments, colOffsets, rowOffsets = lsap_solver_tracking(costMatrix')
        
        ##Switch from returned row assignment of transpose to row assigment of input
        rowAssignments = zeros(Int, size(costMatrix, 1))
        for (jj, row) in enumerate(IndexLinear(), colAssignments)
            rowAssignments[row] = jj
        end
        return rowAssignments, rowOffsets, colOffsets
    end
    
    ##Define algorithm variables
    n, m = size(costMatrix) #set matrix dimensions
    rowOffsets = zeros(G, n) #row cost adjustments
    colOffsets = zeros(G, m) #column cost adjustments
    rowCover = falses(n) #row covered true/false
    colCover = falses(m) #column covered true/false
    starredRow2Col = zeros(Int, n) #either zero or column index of starred zero
    starredCol2Row = zeros(Int, m) #either zero or row index of starred zero
    primedRow2Col = zeros(Int, n) #either zero or column index of primed zero
    minval = typemax(G)::G #tracking minimum value
    zeroCol2Row = Dict{Int, Array{Int, 1}}()
    for jj in 1:m
        zeroCol2Row[jj] = Int[]
    end
    minPoints = Array{Tuple{Int, Int}, 1}(0)
    colsUncovered = Array{Int}(0)
    rowsUncovered = Array{Int}(0)
    for ii in 1:n
        rowOffsets[ii] = costMatrix[ii, 1]
        starredRow2Col[ii] = 1
    end
    
    ##Find row minimums
    for jj in 1:m, ii in 1:n
        if costMatrix[ii, jj] < rowOffsets[ii]
            rowOffsets[ii] = costMatrix[ii, jj]
            starredRow2Col[ii] = jj #track row minimums for the moment
        end
    end

    ##Find all zeros
    for jj in 1:m, ii in 1:n
        if costMatrix[ii, jj] <= rowOffsets[ii]
            push!(zeroCol2Row[jj], ii)
        end
    end

    ##Cover rows and columns
    for ii in 1:n
        if !colCover[starredRow2Col[ii]]
            starredCol2Row[starredRow2Col[ii]] = ii
            rowCover[ii] = true
            colCover[starredRow2Col[ii]] = true
        else
            starredRow2Col[ii] = zero(Int)
        end
    end

    ##Check remaining zeros
    for jj in 1:m
        for ii in zeroCol2Row[jj]
            if !rowCover[ii] && !colCover[jj]
                rowCover[ii] = true
                colCover[jj] = true
                starredRow2Col[ii] = jj
                starredCol2Row[jj] = ii
            end
        end
        
        ##Add still uncovered columns to column ids
        if !colCover[jj]
            push!(colsUncovered, jj)
        end
    end

    ##Check if initial assignment is optimal
    if sum(colCover) == n
        nstep = 7
    else
        nstep = 4
    end

    ##Uncover all rows
    for ii in 1:n
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
            nstep = step3_tracking!(colCover, starredCol2Row, n, m, colsUncovered)
        elseif nstep == 4
            nstep, minval, minPoints = step4_tracking!(costMatrix, rowOffsets, colOffsets,
                                              rowCover, colCover,
                                              starredRow2Col, starredCol2Row,
                                              primedRow2Col,
                                              zeroCol2Row,
                                              minval, minPoints, n, m, colsUncovered, rowsUncovered)
        elseif nstep == 5
            nstep = step5_tracking!(rowCover, colCover,
                           starredRow2Col, starredCol2Row,
                           primedRow2Col,
                           minPoints, n, m, rowsUncovered)
        elseif nstep == 6
            nstep = step6_tracking!(rowOffsets, colOffsets,
                           rowCover, colCover,
                           zeroCol2Row, minval, minPoints, n, m)
        elseif nstep == 7
            break
        else
            error("Non-listed step introduced")
        end
        if verbose
            ctrow = count(rowCover)
            ctcol = count(colCover)
            ct = n - count(iszero.(starredRow2Col))
            if length(minPoints) > 1
                println("Value: $minval, Assigned: $ct, Covered Rows: $ctrow, Covered Cols: $ctcol")
            else
                println("Value: $minval, Assigned: $ct, Covered Rows: $ctrow, Covered Cols: $ctcol, Point: $minPoints")
            end
            println("")
        end
    end

    return starredRow2Col, rowOffsets, colOffsets
end

function lsap_solver_tracking!{G <: Real}(costMatrix::Array{G, 2},
                                          rowOffsets::Array{G, 1},
                                          colOffsets::Array{G, 1},
                                          rowInitial::Array{Int, 1} = zeros(Int, size(costMatrix, 1));
                                          check::Bool = true,
                                          verbose::Bool = false)
    
    ##Flip if more rows than columns
    if size(costMatrix, 1) > size(costMatrix, 2)

        ##Switch initial assignment from rows map to column
        colInitial = zeros(eltype(rowInitial), size(costMatrix, 2))
        for (ii, jj) in enumerate(IndexLinear(), rowInitial)
            if jj != 0::Int
                colInitial[jj] = ii
            end
        end

        ##Find optimal assignment for transpose
        colAssignments, colOffsets, rowOffsets = lsap_solver_tracking(costMatrix')
        
        ##Switch from returned row assignment of transpose to row assigment of input
        rowAssignments = zeros(Int, size(costMatrix, 1))
        for (jj, row) in enumerate(IndexLinear(), colAssignments)
            rowAssignments[row] = jj
        end
        return rowAssignments, rowOffsets, colOffsets
    end
    
    ##Define algorithm variables
    n, m = size(costMatrix) #set matrix dimensions
    rowCover = falses(n) #row covered true/false
    colCover = falses(m) #column covered true/false
    starredRow2Col = zeros(Int, n) #either zero or column index of starred zero
    starredCol2Row = zeros(Int, m) #either zero or row index of starred zero
    primedRow2Col = zeros(Int, n) #either zero or column index of primed zero
    minval = typemax(G)::G #tracking minimum value
    zeroCol2Row = Dict{Int, Array{Int, 1}}()
    for jj in 1:m
        zeroCol2Row[jj] = Int[]
    end
    minPoints = Array{Tuple{Int, Int}, 1}(0)
    colsUncovered = Array{Int}(0)
    rowsUncovered = Array{Int}(0)
    
    ##Find row minimums
    for jj in 1:m, ii in 1:n
        if costMatrix[ii, jj] < rowOffsets[ii]
            rowOffsets[ii] = costMatrix[ii, jj]
        end
    end
    
    ##Check that the row and column offsets are not too high at any point
    if check
        for jj in 1:m, ii in 1:n
            if costMatrix[ii, jj] < (rowOffsets[ii] + colOffsets[jj])
                colOffsets[jj] = costMatrix[ii, jj] - rowOffsets[ii]
            end
        end
    end
    
    ##Find all zeros
    for jj in 1:m, ii in 1:n
        if zero_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
            push!(zeroCol2Row[jj], ii)
        end
    end

    ##Initialize with supplied assignments, checking that they still meet assignment criteria
    for (ii, jj) in enumerate(IndexLinear(), rowInitial)
        if !iszero(jj) && !colCover[jj]
            if zero_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
                rowCover[ii] = true
                colCover[jj] = true
                starredRow2Col[ii] = jj
                starredCol2Row[jj] = ii
            end
        end
    end
    
    ##Check remaining zeros
    for jj in 1:m
        for ii in zeroCol2Row[jj]
            if !rowCover[ii] && !colCover[jj]
                rowCover[ii] = true
                colCover[jj] = true
                starredRow2Col[ii] = jj
                starredCol2Row[jj] = ii
            end
        end
        
        ##Add still uncovered columns to column ids
        if !colCover[jj]
            push!(colsUncovered, jj)
        end
    end

    ##Check if initial assignment is optimal
    if sum(colCover) == n
        nstep = 7
    else
        nstep = 4
    end

    ##Uncover all rows
    for ii in 1:n
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
            nstep = step3_tracking!(colCover, starredCol2Row, n, m, colsUncovered)
        elseif nstep == 4
            nstep, minval, minPoints = step4_tracking!(costMatrix, rowOffsets, colOffsets,
                                              rowCover, colCover,
                                              starredRow2Col, starredCol2Row,
                                              primedRow2Col,
                                              zeroCol2Row,
                                              minval, minPoints, n, m, colsUncovered, rowsUncovered)
        elseif nstep == 5
            nstep = step5_tracking!(rowCover, colCover,
                           starredRow2Col, starredCol2Row,
                           primedRow2Col,
                           minPoints, n, m, rowsUncovered)
        elseif nstep == 6
            nstep = step6_tracking!(rowOffsets, colOffsets,
                           rowCover, colCover,
                           zeroCol2Row, minval, minPoints, n, m)
        elseif nstep == 7
            break
        else
            error("Non-listed step introduced")
        end
        if verbose
            ctrow = count(rowCover)
            ctcol = count(colCover)
            ct = n - count(iszero.(starredRow2Col))
            if length(minPoints) > 1
                println("Value: $minval, Assigned: $ct, Covered Rows: $ctrow, Covered Cols: $ctcol")
            else
                println("Value: $minval, Assigned: $ct, Covered Rows: $ctrow, Covered Cols: $ctcol, Point: $minPoints")
            end
            println("")
        end
    end

    return starredRow2Col, rowOffsets, colOffsets
end

function lsap_solver_tracking{G <: Real}(costMatrix::Array{G, 2},
                                         rowOffsets::Array{G, 1},
                                         colOffsets::Array{G, 1},
                                         rowInitial::Array{Int, 1} = zeros(Int, size(costMatrix, 1));
                                         check::Bool = true,
                                         verbose::Bool = false)
    return lsap_solver_tracking!(costMatrix, copy(rowOffsets), copy(colOffsets), rowInitial,
                                 check = check, verbose = verbose)
end



"""
    step3!(colCover, starredRow2Col, n, m) -> step 3


"""
function step3_tracking!(colCover::BitArray{1},
                starredCol2Row::Array{Int, 1},
                n::Int,
                m::Int,
                colsUncovered::Array{Int, 1})
    
    ##Non-zero values for starredRow2Col are columns with starred zeros
    #switch to updating colsUncovered
    jj = 1
    while jj <= length(colsUncovered)
        if iszero(starredCol2Row[colsUncovered[jj]]) #!= zero(T)
            jj += 1
        else
            colCover[colsUncovered[jj]] = true
            deleteat!(colsUncovered, jj)
        end
    end

    ##Check if done, otherwise go to step 4
    #switch to length(colsUncovered)
    if count(colCover) == n
        return 7
    else
        return 4
    end
end

"""
    step4!(costMatrix, rowOffsets, colOffsets, rowCover, colCover,
           starredRow2Col, starredCol2Row, primedRow2Col,
           minval, minrow, mincol, n, m) -> step, minval, minrow, mincol

Find 0's where cost[ii, jj] == rowOffsets[ii] + colOffsets[jj].  If
cover row ii and column jj are false set them to true and star 0.
When staring add to starredRow2Col and starredCol2Row
"""
function step4_tracking!{G <: Real}(costMatrix::Array{G, 2},
                           rowOffsets::Array{G, 1},
                           colOffsets::Array{G, 1},
                           rowCover::BitArray{1},
                           colCover::BitArray{1},
                           starredRow2Col::Array{Int, 1},
                           starredCol2Row::Array{Int, 1},
                           primedRow2Col::Array{Int, 1},
                           zeroCol2Row::Dict{Int, Array{Int, 1}},
                           minval::G,
                           minPoints::Array{Tuple{Int, Int}, 1},
                           n::Int, m::Int, colsUncovered::Array{Int, 1}, rowsUncovered::Array{Int, 1})

    ##value defaults
    empty!(minPoints)
    minval = typemax(G)::G
    
    ##loop popping items off queue so that uncovered columns can be added
    col = 1
    while col <= length(colsUncovered)
        jj = colsUncovered[col]

        for row in zeroCol2Row[jj]
            if rowCover[row]
                continue
            else
                primedRow2Col[row] = jj
                if iszero(starredRow2Col[row])
                    push!(minPoints, (row, jj))
                    return 5, minval, minPoints
                else
                    rowCover[row] = true
                    colCover[starredRow2Col[row]] = false
                    push!(colsUncovered, starredRow2Col[row])
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
            val = adjusted_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
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
    step5!(rowCover, colCover, starredRow2Col, starredCol2Row,
           primedRow2Col, minrow, mincol, n, m) -> 3

Find 0's where cost[ii, jj] == rowOffsets[ii] + colOffsets[jj].  If
cover row ii and column jj are false set them to true and star 0.
When staring add to starredRow2Col and starredCol2Row
"""
function step5_tracking!(rowCover::BitArray{1}, colCover::BitArray{1},
                starredRow2Col::Array{Int, 1}, starredCol2Row::Array{Int, 1},
                primedRow2Col::Array{Int, 1}, minPoints::Array{Tuple{Int, Int}, 1},
                n::Int, m::Int, rowsUncovered::Array{Int, 1})

    ##initialize array for tracking sequence, alternating primed and starred
    primedRows = Int[minPoints[1][1]]
    primedCols = Int[minPoints[1][2]]
    starredRows = Int[]
    starredCols = Int[]

    ##If no starred zero in row terminate, otherwise add starred zero and the primed
    ##zero in the column of the starred zero.  Continue until terminations...
    while true
        row = starredCol2Row[primedCols[end]]
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
        starredRow2Col[row] = primedRow2Col[row]
        starredCol2Row[primedRow2Col[row]] = row
    end

    ##Initial primed zero has no corresponding starred zero in column so star it
    starredRow2Col[minPoints[1][1]] = minPoints[1][2]::Int
    starredCol2Row[minPoints[1][2]] = minPoints[1][1]::Int
    
    ##Erase all primes
    primedRow2Col[:] = 0::Int
    
    ##uncover every line in the matrix
    for ii in 1:n
        if rowCover[ii]
            rowCover[ii] = false
            push!(rowsUncovered, ii)
        end
    end

    return 3
end

"""
    step6!(rowOffsets, colOffsets, rowCover, colCover, minval) -> 4


"""
function step6_tracking!{G <: Real}(rowOffsets::Array{G, 1},
                           colOffsets::Array{G, 1},
                           rowCover::BitArray{1},
                           colCover::BitArray{1},
                           zeroCol2Row::Dict{Int, Array{Int, 1}},
                           minval::G,
                           minPoints::Array{Tuple{Int, Int}, 1}, n::Int, m::Int)
    ##Add min to (subtract from offset) all elements in covered rows
    for ii in 1:n
        if rowCover[ii]
            rowOffsets[ii] -= minval
        end
    end
    
    ##Subtract min from (add to offset) all elements in uncovered columns
    for jj in 1:m
        if !colCover[jj]
            colOffsets[jj] += minval
        end
    end

    #Remove zeros that where rowCover and colCover are both covered
    for jj in 1:m
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
