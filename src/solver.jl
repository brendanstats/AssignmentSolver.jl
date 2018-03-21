#using DataStructures
#http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html

"""
    step1!(costMatrix, rowOffsets) -> nextstep

Initialization, set rowOffsets to row minimum, returns step2
"""
function step1!{G <: Real}(costMatrix::Array{G, 2}, rowOffsets::Array{G, 1})
    rowOffsets[:] = vec(minimum(costMatrix, 2))::Array{G, 1}
    return 2
end

"""
    step2!(costMatrix, rowOffsets, colOffsets, rowCover, colCover,
           starredRow2Col, starredCol2Row, n, m) -> step 3

Find 0's where cost[ii, jj] == rowOffsets[ii] + colOffsets[jj].  If
cover row ii and column jj are false set them to true and star 0.
When staring add to starredRow2Col and starredCol2Row
"""
function step2!{G <: Real}(costMatrix::Array{G, 2},
                           rowOffsets::Array{G, 1},
                           colOffsets::Array{G, 1},
                           rowCover::Array{Bool, 1},
                           colCover::Array{Bool, 1},
                           starredRow2Col::Array{Int, 1},
                           starredCol2Row::Array{Int, 1},
                           n::Int, m::Int)
    for jj in 1:m
        for ii in 1:n
            ##skip row if covered
            if rowCover[ii]::Bool
                continue
            end

            ##check if adjusted cost is zero
            if zero_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
            #if costMatrix[ii, jj] == rowOffsets[ii] + colOffsets[jj]
            #if zero(G) == costMatrix[ii, jj] - rowOffsets[ii] - colOffsets[jj]
                rowCover[ii] = true
                colCover[jj] = true
                starredRow2Col[ii] = jj::Int
                starredCol2Row[jj] = ii::Int
                
                ##row and column are now covered, go to next column
                break
            end
        end
    end

    ##terminate if n columns are covered, otherwise uncover rows and go to 4 as columns are already covered so step 3 is unnecessary
    if count(colCover) == n::Int
        return 7
    else
        rowCover[:] = false
        return 4
    end
end

function step2_col!{G <: Real}(costMatrix::Array{G, 2},
                               rowOffsets::Array{G, 1},
                               colOffsets::Array{G, 1},
                               rowCover::Array{Bool, 1},
                               colCover::Array{Bool, 1},
                               starredRow2Col::Array{Int, 1},
                               starredCol2Row::Array{Int, 1},
                               n::Int, m::Int)
    for jj in 1:m
        ##skip column if covered
        if colCover[jj]
            continue
        end
        
        for ii in 1:n
            ##skip row if covered
            if rowCover[ii]
                continue
            end

            ##check if adjusted cost is zero
            if zero_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
            #if costMatrix[ii, jj] == rowOffsets[ii] + colOffsets[jj]::G
            #if zero(G) == costMatrix[ii, jj] - rowOffsets[ii] - colOffsets[jj]
                rowCover[ii] = true
                colCover[jj] = true
                starredRow2Col[ii] = jj::Int
                starredCol2Row[jj] = ii::Int
                
                ##row and column are now covered, go to next column
                break
            end
        end
    end

    ##terminate if n columns are covered, otherwise uncover rows and go to 4 as columns are already covered so step 3 is unnecessary
    if count(colCover) == n
        return 7
    else
        rowCover[:] = false
        return 4
    end
end

"""
    step3!(colCover, starredRow2Col, n, m) -> step 3


"""
function step3!(colCover::Array{Bool, 1},
                starredRow2Col::Array{Int, 1},
                n::Int, m::Int)
    
    ##Non-zero values for starredRow2Col are columns with starred zeros
    for colId in starredRow2Col
        if !iszero(colId) #!= zero(T)
            colCover[colId] = true
        end
    end

    ##Check if done, otherwise go to step 4
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
function step4!{G <: Real}(costMatrix::Array{G, 2},
                           rowOffsets::Array{G, 1},
                           colOffsets::Array{G, 1},
                           rowCover::Array{Bool, 1},
                           colCover::Array{Bool, 1},
                           starredRow2Col::Array{Int, 1},
                           starredCol2Row::Array{Int, 1},
                           primedRow2Col::Array{Int, 1},
                           minval::G,
                           minrow::Int,
                           mincol::Int,
                           n::Int, m::Int)
    ##add uncovered columns to Queue to make it easier to add additional columns later
    loopCols = Queue(Int)
    for jj in 1:m
        if !colCover[jj]
            enqueue!(loopCols, jj)
        end
    end

    ##value defaults
    minval = typemax(G)::G
    minrow = -1
    mincol = -1
    
    ##loop popping items off queue so that uncovered columns can be added
    while length(loopCols) > 0::Int
        jj = dequeue!(loopCols)

        ##loop checking uncovered rows
        for ii in 1:n
            if rowCover[ii]
                continue
            end

            ##compute adjusted cost and prime zeros
            if zero_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
                primedRow2Col[ii] = jj
                if iszero(starredRow2Col[ii])
                    minrow = ii
                    mincol = jj
                    return 5, minval, minrow, mincol
                else
                    rowCover[ii] = true
                    colCover[starredRow2Col[ii]] = false
                    enqueue!(loopCols, starredRow2Col[ii])
                end
            end
        end #end for
    end #end while

    #If all rows are covered then go to step 5, not sure why this happens
    if all(rowCover)
        warn("All rows coverd by step 5 skipped find bug...")
        println("Rows covered: ", count(rowCover))
        println("Columns covered: ", count(colCover))
        println("Rows starred: ", count(starredRow2Col .!= 0))
        println("Columns starred: ", count(starredCol2Row .!= 0))
        println("Rows primed: ", count(primedRow2Col .!= 0))
        return 7, minval, minrow, mincol
    end
        
    for jj in find(.!colCover), ii in find(.!rowCover)
        if adjusted_cost(ii, jj, costMatrix, rowOffsets, colOffsets) < minval
            minval = adjusted_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
            minrow = ii::Int
            mincol = jj::Int
        end
    end
    
    ##If min value is negative (due to numerical issues) unprime and uncover and re-start at step 3
    if iszero(minval) # == zero(G)
        warn("minvalue is zero")
        primedRow2Col[minrow] = mincol::Int
        if iszero(starredRow2Col[minrow]) # == zero(T)
            return 5, minval, minrow, mincol
        else
            rowCover[minrow] = true
            colCover[starredRow2Col[minrow]] = false
            return 4, minval, minrow, mincol
        end

    elseif minval::G < zero(G)
        warn("minimum is less than 0: $minval")
        colOffsets[mincol] = costMatrix[minrow, mincol]::G - rowOffsets[minrow]::G
        primedRow2Col[:] = 0::Int
        colCover[:] = false
        rowCover[:] = false
        println("new value: $(costMatrix[minrow, mincol] - (rowOffsets[minrow] + colOffsets[mincol]))")
        return 3, minval, minrow, mincol
    end
    return 6, minval, minrow, mincol
end

"""
    step5!(rowCover, colCover, starredRow2Col, starredCol2Row,
           primedRow2Col, minrow, mincol, n, m) -> 3

Find 0's where cost[ii, jj] == rowOffsets[ii] + colOffsets[jj].  If
cover row ii and column jj are false set them to true and star 0.
When staring add to starredRow2Col and starredCol2Row
"""
function step5!(rowCover::Array{Bool, 1},
                              colCover::Array{Bool, 1},
                              starredRow2Col::Array{Int, 1},
                              starredCol2Row::Array{Int, 1},
                              primedRow2Col::Array{Int, 1},
                              minrow::Int,
                              mincol::Int,
                              n::Int, m::Int)

    ##initialize array for tracking sequence, alternating primed and starred
    primedRows = Int[minrow]
    primedCols = Int[mincol]
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
    starredRow2Col[minrow] = mincol::Int
    starredCol2Row[mincol] = minrow::Int
    
    ##Erase all primes
    primedRow2Col[:] = 0::Int
    
    ##uncover every line in the matrix
    rowCover[:] = false

    return 3
end

"""
    step6!(rowOffsets, colOffsets, rowCover, colCover, minval) -> 4


"""
function step6!{G <: Real}(rowOffsets::Array{G, 1},
                           colOffsets::Array{G, 1},
                           rowCover::Array{Bool, 1},
                           colCover::Array{Bool, 1},
                           minval::G)
    ##Add min to (subtract from offset) all elements in covered rows
    rowOffsets[rowCover] .-= minval::G
    #adjust = 0.5 * minval
    #rowOffsets[rowCover] += adjust
    #rowOffsets[.!rowCover] -= adjust
    
    ##Subtract min from (add to offset) all elements in uncovered columns
    colOffsets[.!colCover] .+= minval::G
    #colOffsets[.!colCover] += adjust
    #colOffsets[colCover] -= adjust
    return 4
end

function zero_cost{G <: Real}(ii::Integer, jj::Integer, costMatrix::Array{G, 2},
                              rowOffsets::Array{G, 1},
                              colOffsets::Array{G, 1})
    return iszero(adjusted_cost(ii, jj, costMatrix, rowOffsets, colOffsets))
end

function adjusted_cost{G <: Real}(ii::Integer, jj::Integer,
                                  costMatrix::Array{G, 2},
                                  rowOffsets::Array{G, 1},
                                  colOffsets::Array{G, 1})
    return costMatrix[ii, jj] - (rowOffsets[ii] + colOffsets[jj])
end

"""
    compute_adjusted_cost(costMatrix, rowOffsets, colOffsets) -> adjustedCostMatrix

Compute the adjustedCostMatrix[ii, jj] = costMatrix[ii, jj] - rowOffsets[ii] - colOffsets[jj]
"""
function adjusted_cost{G <: Real}(costMatrix::Array{G, 2},
                                  rowOffsets::Array{G, 1},
                                  colOffsets::Array{G, 1})
    out = Array{G}(size(costMatrix))
    for jj in 1:length(colOffsets), ii in 1:length(rowOffsets)
        out[ii, jj] = costMatrix[ii, jj] - (rowOffsets[ii] + colOffsets[jj])
    end
    return out
end

"""
    lsap_solver(costMatrix) -> rowAssignments, rowOffsets, colOffsets
"""
function lsap_solver{G <: Real}(costMatrix::Array{G, 2}; verbose::Bool = false)
    if size(costMatrix, 1) > size(costMatrix, 2)
        
        #warn("more rows than columns, 0 assigments correspond to empty rows")

        ##Find optimal assignment for transpose
        colAssignments, colOffsets, rowOffsets = lsap_solver(costMatrix')

        ##Switch from returned row assignment of transpose to row assigment of input
        rowAssignments = zeros(Int64, size(costMatrix, 1))
        for (jj, row) in enumerate(IndexLinear(), colAssignments)
            rowAssignments[row] = jj
        end
        return rowAssignments, rowOffsets, colOffsets
    end

    ##Define algorithm variables
    n = size(costMatrix, 1) #set matrix dimensions
    m = size(costMatrix, 2)
    rowOffsets = zeros(G, n) #row cost adjustments
    colOffsets = zeros(G, m) #column cost adjustments
    rowCover = zeros(Bool, n) #row covered true/false
    colCover = zeros(Bool, m) #column covered true/false
    starredRow2Col = zeros(Int, n) #either zero or column index of starred zero
    starredCol2Row = zeros(Int, m) #either zero or row index of starred zero
    primedRow2Col = zeros(Int, n) #either zero or column index of primed zero
    minval = typemax(G)::G #tracking minimum value
    minrow = -1 #track row of minimum value (or of last primed zero)
    mincol = -1 #track column of minimum value (or of last primed zero)

    ##Initialize
    nstep = step1!(costMatrix, rowOffsets)
    nstep = step2!(costMatrix, rowOffsets, colOffsets,
                   rowCover, colCover,
                   starredRow2Col, starredCol2Row, n, m)
    while true
        if verbose
            println("step = ", nstep)
        end
        if nstep == 3
            nstep = step3!(colCover, starredRow2Col, n, m)
        elseif nstep == 4
            nstep, minval, minrow, mincol = step4!(costMatrix, rowOffsets, colOffsets,
                                                   rowCover, colCover,
                                                   starredRow2Col, starredCol2Row,
                                                   primedRow2Col,
                                                   minval, minrow, mincol, n, m)
        elseif nstep == 5
            nstep = step5!(rowCover, colCover,
                           starredRow2Col, starredCol2Row,
                           primedRow2Col,
                           minrow, mincol, n, m)
        elseif nstep == 6
            nstep = step6!(rowOffsets, colOffsets,
                           rowCover, colCover, minval)
        elseif nstep == 7
            break
        else
            error("Non-listed step introduced")
        end
        if verbose
            if any(rowOffsets .< 0.0)
                rowmin = minimum(rowOffsets)
                println("Row minimum: $rowmin")
            end
        end
    end

    return starredRow2Col, rowOffsets, colOffsets
end

"""
    lsap_solver_initialized!(costMatrix, rowOffsets, colOffsets) -> rowAssignments, rowOffsets, colOffsets
"""
function lsap_solver_initialized!{G <: Real}(costMatrix::Array{G, 2},
                                             rowInitial::Array{Int, 1},
                                             rowOffsets::Array{G, 1},
                                             colOffsets::Array{G, 1};
                                             check::Bool = true,
                                             adjustRowOffsets::Bool = true,
                                             verbose::Bool = false)
    if size(costMatrix, 1) > size(costMatrix, 2)
        
        warn("more rows than columns, 0 assigments correspond to empty rows")

        ##Switch initial assignment from rows map to column
        colInitial = zeros(eltype(rowInitial), size(costMatrix, 2))
        for (ii, jj) in enumerate(IndexLinear(), rowInitial)
            if jj != 0::Int
                colInitial[jj] = ii
            end
        end
        
        ##Find optimal assignment for transpose
        colAssignments, colOffsets, rowOffsets = lsap_solver_initialized!(costMatrix', colInitial, colOffsets, rowOffsets, check = check, adjustRowOffsets = adjustRowOffsets)

        ##Switch from returned row assignment of transpose to row assigment of input
        rowAssignments = zeros(Int, size(costMatrix, 1))
        for (jj, row) in enumerate(IndexLinear(), colAssignments)
            rowAssignments[row] = jj
        end
        return rowAssignments, rowOffsets, colOffsets
    end

    ##Define algorithm variables
    n, m = size(costMatrix) #set matrix dimensions
    rowCover = zeros(Bool, n) #row covered true/false
    colCover = zeros(Bool, m) #column covered true/false
    starredRow2Col = zeros(Int, n) #either zero or column index of starred zero
    starredCol2Row = zeros(Int, m) #either zero or row index of starred zero
    primedRow2Col = zeros(Int, n) #either zero or column index of primed zero
    minval = typemax(G) #tracking minimum value
    minrow = -1 #track row of minimum value (or of last primed zero)
    mincol = -1 #track column of minimum value (or of last primed zero)

    ##Set costs of initial assignment to zero
    if adjustRowOffsets

        rowMins = vec(minimum(costMatrix, 2))
        for ii in 1:n
            if rowOffsets[ii] > rowMins[ii]
                rowOffsets[ii] = rowMins[ii]
            end
        end
        #for (ii, jj) in enumerate(IndexLinear(), rowInitial)
        #    if costMatrix[ii, jj] > (rowOffsets[ii] + colOffsets[jj])
        #        rowOffsets[ii] = costMatrix[ii, jj] - colOffsets[jj]
        #    end
        #end
    end

    ##Check that the row and column offsets are not too high at any point
    if check
        for jj in 1:m, ii in 1:n
            if costMatrix[ii, jj] < (rowOffsets[ii] + colOffsets[jj])
                #warn("Initial offsets total more than cost matrix, reducing column offset")
                colOffsets[jj] = costMatrix[ii, jj] - rowOffsets[ii]
            end
        end
    end

    ##Initialize with supplied assignments, checking that they still meet assignment criteria
    for (ii, jj) in enumerate(IndexLinear(), rowInitial)
        if jj != 0
            #if costMatrix[ii, jj] == (rowOffsets[ii] + colOffsets[jj])
            if zero_cost(ii, jj, costMatrix, rowOffsets, colOffsets)
                rowCover[ii] = true
                colCover[jj] = true
                starredRow2Col[ii] = jj
                starredCol2Row[jj] = ii
            end
        end
    end
    
    nstep = step2_col!(costMatrix, rowOffsets, colOffsets,
                       rowCover, colCover,
                       starredRow2Col, starredCol2Row, n, m)
    while true
        if verbose
            println("step = ", nstep)
        end
        if nstep == 3
            nstep = step3!(colCover, starredRow2Col, n, m)
        elseif nstep == 4
            nstep, minval, minrow, mincol = step4!(costMatrix, rowOffsets, colOffsets,
                                                   rowCover, colCover,
                                                   starredRow2Col, starredCol2Row,
                                                   primedRow2Col,
                                                   minval, minrow, mincol, n, m)
        elseif nstep == 5
            nstep = step5!(rowCover, colCover,
                           starredRow2Col, starredCol2Row,
                           primedRow2Col,
                           minrow, mincol, n, m)
        elseif nstep == 6
            nstep = step6!(rowOffsets, colOffsets,
                           rowCover, colCover, minval)
        elseif nstep == 7
            break
        else
            error("Non-listed step introduced")
        end
    end

    return starredRow2Col, rowOffsets, colOffsets
end
