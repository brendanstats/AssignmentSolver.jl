#β - Bertsekas 1988 pdf 20
#implement third best prices

#number of column costs below λ
#number of unassigned columnCosts above λ

"""
Find the index of the maximum adjusted reward in the row and the amount this exceeds the second largest value
"""
function maxtwoRow(row::Integer, rewardMatrix::Array{T, 2}, colCosts::Array{T, 1}, ncol::Integer) where T <: AbstractFloat
    maxcol = 1
    maxval = rewardMatrix[row, 1] - colCosts[1]
    maxtwo = typemin(T)
    if ncol == 1
        return maxcol, maxval, maxtwo
    end
    for jj in 2:ncol
        val = (rewardMatrix[row, jj] - colCosts[jj])
        if  val > maxval
            maxtwo = maxval
            maxval = val
            maxcol = jj
        elseif val > maxtwo
            maxtwo = val
        end
    end
    return maxcol, maxval, maxtwo
end

#not complete
function maxthreeRow(row::Integer, rewardMatrix::Array{T, 2}, colCosts::Array{T, 1}, ncol::Integer) where T <: AbstractFloat
    maxcol = 1
    maxval = rewardMatrix[row, 1] - colCosts[1]
    maxtwo = typemin(T)
    maxthree = typemin(T)
    if ncol == 1
        return maxcol, maxval, maxtwo
    end
    for jj in 2:ncol
        val = (rewardMatrix[row, jj] - colCosts[jj])
        if  val > maxval
            maxtwo = maxval
            maxval = val
            maxcol = jj
        elseif val > maxtwo
            maxtwo = val
        end
    end
    return maxcol, maxval, maxtwo
end

function maxtwoCol(col::G, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, nrow::G) where {G <: Integer, T <: AbstractFloat}
    maxrow = 1
    maxval = rewardMatrix[1, col] - rowCosts[1]
    maxtwo = typemin(T)
    if nrow == 1
        return maxrow, maxval, maxtwo
    end
    for ii in 2:nrow
        val = rewardMatrix[ii, col] - rowCosts[ii]
        if val > maxval
            maxtwo = maxval
            maxval = val
            maxrow = ii
        elseif val > maxtwo
            maxtwo = val
        end
    end
    return maxrow, maxval, maxtwo
end

function maxtwoCol(col::G, rewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}) where {G <: Integer, T <: AbstractFloat}
    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)
    rng = nzrange(rewardMatrix, col)
    if length(rng) == 0
        error("problem not feasible")
    elseif length(rng) == 1
        return rows[rng[1]], rewards[rng[1]], -1000.0
    end
    maxrow = 0
    maxval = -1000.0
    maxtwo = -1000.0
    for ii in rng
        val = rewards[ii] - rowCosts[rows[ii]]
        if val > maxval
            maxtwo = maxval
            maxval = val
            maxrow = rows[ii]
        elseif val > maxtwo
            maxtwo = val
        end
    end
    return maxrow, maxval, maxtwo
end

function maxtwoCol_shadow(col::G, rewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}) where {G <: Integer, T <: AbstractFloat}
    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)
    rng = nzrange(rewardMatrix, col)
    if length(rng) == 0
        return zero(G), zero(T), typemin(T)
    elseif length(rng) == 1
        return rows[rng[1]], rewards[rng[1]] - rowCosts[rows[rng[1]]], typemin(T)
    end
    maxrow = 0
    maxval = typemin(T)
    maxtwo = typemin(T)
    for ii in rng
        val = rewards[ii] - rowCosts[rows[ii]]
        if val > maxval
            maxtwo = maxval
            maxval = val
            maxrow = rows[ii]
        elseif val > maxtwo
            maxtwo = val
        end
    end
    return maxrow, maxval, maxtwo
end


"""
forward iteration for a single un-assigned row in a symmetric auction assignment algorithm
"""
function forward_iteration(row::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ncol::G, ε::T) where {G <: Integer, T <: AbstractFloat}
    maxcol, maxval, maxtwo = maxtwoRow(row, rewardMatrix, colCosts, ncol)
    margin = maxval - maxtwo + ε
    colCosts[maxcol] += margin
    rowCosts[row] = rewardMatrix[row, maxcol] - colCosts[maxcol]
    if iszero(c2r[maxcol]) #only generates an additional assignment if maxcol is unassigned
        unassignedrow = 0
    else
        unassignedrow = c2r[maxcol]
        r2c[c2r[maxcol]] = zero(G)
    end
    r2c[row] = maxcol
    c2r[maxcol] = row
    
    return r2c, c2r, rowCosts, colCosts, unassignedrow
end

"""
forward iteration for a single un-assigned row in an asymmetric auction assignment algorithm
"""
function forward_iteration(row::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ncol::G, ncolbelowλ::G, ε::T, λ::T) where {G <: Integer, T <: AbstractFloat}
    maxcol, maxval, maxtwo = maxtwoRow(row, rewardMatrix, colCosts, ncol)
    margin = maxtwo - ε
    adj = rewardMatrix[row, maxcol] - margin

    #This will be increased to at least λ so if below then the number below λ will fall
    if colCosts[maxcol] < λ
        ncolbelowλ -= 1
    end
    
    if λ <= adj
        colCosts[maxcol] = adj
        if iszero(c2r[maxcol]) #only generates an additional assignment if maxcol is unassigned)
            unassignedrow = 0
        else
            unassignedrow = c2r[maxcol]
            r2c[c2r[maxcol]] = zero(G)
        end
        r2c[row] = maxcol
        c2r[maxcol] = row
    else
        colCosts[maxcol] = λ
        unassignedrow = row
    end
    rowCosts[row] = margin
    return r2c, c2r, rowCosts, colCosts, unassignedrow, ncolbelowλ
end

function forward_iteration(row::G, r2c::Array{G, 1}, c2r::Array{G, 1}, trewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ncol::G, ncolbelowλ::G, ε::T, λ::T) where {G <: Integer, T <: AbstractFloat}
    maxcol, maxval, maxtwo = maxtwoCol(row, trewardMatrix, colCosts)
    margin = maxtwo - ε
    adj = trewardMatrix[maxcol, row] - margin #indicies switched intentionally as matrix is transposed

    #This will be increased to at least λ so if below then the number below λ will fall
    if colCosts[maxcol] < λ
        ncolbelowλ -= 1
    end
    
    if λ <= adj
        colCosts[maxcol] = adj
        if iszero(c2r[maxcol]) #only generates an additional assignment if maxcol is unassigned)
            unassignedrow = 0
        else
            unassignedrow = c2r[maxcol]
            r2c[c2r[maxcol]] = zero(G)
        end
        r2c[row] = maxcol
        c2r[maxcol] = row
    else
        colCosts[maxcol] = λ
        unassignedrow = row
    end
    rowCosts[row] = margin
    return r2c, c2r, rowCosts, colCosts, unassignedrow, ncolbelowλ
end

function forward_iteration_shadow(row::G, r2c::Array{G, 1}, c2r::Array{G, 1}, trewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, shadowCosts::Array{T, 1}, ncol::G, ncolbelowλ::G, nshadow::G, ε::T, λ::T) where {G <: Integer, T <: AbstractFloat}
    maxcol, maxval, maxtwo = maxtwoCol_shadow(row, trewardMatrix, colCosts)
    if iszero(maxcol) #|| (maxval < zero(T)) #try - λ instead of maxval < zero(T) adj < zero
        println("row: $row, margin: $margin, adj: $adj, maxval: $maxval, maxtwo: $maxtwo")
        rowCosts[row] = -λ
        r2c[row] = row + ncol
        unassignedrow = 0
        nshadow += 1
    elseif maxval < -shadowCosts[row] #case where shadow entry is the max
        margin = maxval - ε #maxval has become maxtwo
        adj = zero(T) - margin
        if λ <= adj
            shadowCosts[row] = adj
            r2c[row] = row + ncol
            unassignedrow = 0
            nshadow += 1
        else
            shadowCosts[row] = λ
            unassignedrow = row
        end
    else #case where things are normal
        maxtwo = max(-shadowCosts[row], maxtwo) # #can always assign to other column with zero cost
        margin = maxtwo - ε
        adj = trewardMatrix[maxcol, row] - margin #indicies switched intentionally as matrix is transposed
        
        #This will be increased to at least λ so if below then the number below λ will fall
        if colCosts[maxcol] < λ
            ncolbelowλ -= 1
        end
        #Determine if assignment is made
        if λ <= adj
            colCosts[maxcol] = adj
            if iszero(c2r[maxcol]) #only generates an additional assignment if maxcol is unassigned)
                unassignedrow = 0
            else
                unassignedrow = c2r[maxcol]
                r2c[c2r[maxcol]] = zero(G)
            end
            r2c[row] = maxcol
            c2r[maxcol] = row
        else
            colCosts[maxcol] = λ
            unassignedrow = row
        end
    end
    rowCosts[row] = margin
    return r2c, c2r, rowCosts, colCosts, shadowCosts, unassignedrow, ncolbelowλ, nshadow
end

function backward_iteration(col::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, nrow::G, ε::T) where {G <: Integer, T <: AbstractFloat}
    maxrow, maxval, maxtwo = maxtwoCol(col, rewardMatrix, rowCosts, nrow)
    margin = maxval - maxtwo + ε
    rowCosts[maxrow] += margin
    colCosts[col] = rewardMatrix[maxrow, col] - rowCosts[maxrow]
    if iszero(r2c[maxrow]) #only generates an additional assignment if maxrow is unassigned
        unassignedcol = 0
    else
        unassignedcol = r2c[maxrow]
        c2r[r2c[maxrow]] = zero(G)
    end    
    
    r2c[maxrow] = col
    c2r[col] = maxrow
    
    return r2c, c2r, rowCosts, colCosts, unassignedcol
end

function backward_iteration(col::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, nrow::G, ncolbelowλ::G, ε::T, λ::T) where {G <: Integer, T <: AbstractFloat}
    maxrow, maxval, maxtwo = maxtwoCol(col, rewardMatrix, rowCosts, nrow)
    margin = maxtwo - ε
    if maxval >= (λ + ε)
        
        if colCosts[col] < λ
            ncolbelowλ -= 1
        end        
        colCosts[col] = max(λ, margin)
        rowCosts[maxrow] = rewardMatrix[maxrow, col] - max(λ, margin)
        
        if iszero(r2c[maxrow])
            unassignedcol = 0
        else
            unassignedcol = r2c[maxrow]
            c2r[r2c[maxrow]] = zero(G)
        end
        r2c[maxrow] = col
        c2r[col] = maxrow
        #belowλ = false
    else
        unassignedcol = col
        if colCosts[col] >= λ && (maxval - ε) < λ
            #belowλ = true
            ncolbelowλ += 1
        else
            #belowλ = false
        end
        colCosts[col] = maxval - ε
    end
    return r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ
end

function backward_iteration(col::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, nrow::G, ncolbelowλ::G, ε::T, λ::T) where {G <: Integer, T <: AbstractFloat}
    maxrow, maxval, maxtwo = maxtwoCol(col, rewardMatrix, rowCosts)
    margin = maxtwo - ε
    if maxval >= (λ + ε)
        
        if colCosts[col] < λ
            ncolbelowλ -= 1
        end        
        colCosts[col] = max(λ, margin)
        rowCosts[maxrow] = rewardMatrix[maxrow, col] - max(λ, margin)
        
        if iszero(r2c[maxrow])
            unassignedcol = 0
        else
            unassignedcol = r2c[maxrow]
            c2r[r2c[maxrow]] = zero(G)
        end
        r2c[maxrow] = col
        c2r[col] = maxrow
        #belowλ = false
    else
        unassignedcol = col
        if colCosts[col] >= λ && (maxval - ε) < λ
            #belowλ = true
            ncolbelowλ += 1
        else
            #belowλ = false
        end
        colCosts[col] = maxval - ε
    end
    return r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ
end

function backward_iteration_shadow(col::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, nrow::G, ncolbelowλ::G, ε::T, λ::T) where {G <: Integer, T <: AbstractFloat}
    maxrow, maxval, maxtwo = maxtwoCol_shadow(col, rewardMatrix, rowCosts)
    if iszero(maxrow)
        unassignedcol = col
        if colCost[col] > λ
            ncolbelowλ += 1
            colCost[col] = zero(T)
        end
        return r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ
    end
    margin = maxtwo - ε
    if maxval >= (λ + ε)
        
        if colCosts[col] < λ
            ncolbelowλ -= 1
        end        
        colCosts[col] = max(λ, margin)
        rowCosts[maxrow] = rewardMatrix[maxrow, col] - max(λ, margin)
        
        if iszero(r2c[maxrow])
            unassignedcol = 0
        else
            unassignedcol = r2c[maxrow]
            if r2c[maxrow] <= length(c2r)
                c2r[r2c[maxrow]] = zero(G)
            end
        end
        r2c[maxrow] = col
        c2r[col] = maxrow
        #belowλ = false
    else
        unassignedcol = col
        if colCosts[col] >= λ && (maxval - ε) < λ
            #belowλ = true
            ncolbelowλ += 1
        else
            #belowλ = false
        end
        colCosts[col] = maxval - ε
    end
    return r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ
end

function check_costs!(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, nrow::Integer, ncol::Integer) where {G <: Integer, T <: AbstractFloat}
    for jj in 1:ncol, ii in 1:nrow
        if (rewardMatrix[ii, jj] - ε) > (rowCosts[ii] + colCosts[jj])
            colCosts[jj] = rewardMatrix[ii, jj] - ε - rowCosts[ii]

            #remove assignment if increase column costs since equality will no longer hold
            if !iszero(c2r[jj])
                r2c[c2r[jj]]  = zero(G)
                c2r[jj] = zero(G)
            end
        end
    end
    return nothing
end

function check_costs!(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, nrow::Integer, ncol::Integer) where {G <: Integer, T <: AbstractFloat}
    rows = rowvals(rewardMatrix)
    rewards = nonzeros(rewardMatrix)
    for jj in 1:ncol
        for ii in nzrange(rewardMatrix, jj)
            row = rows[ii]
            if (rewardMatrix[row, jj] - ε) > (rowCosts[row] + colCosts[jj])
                colCosts[jj] = rewardMatrix[row, jj] - ε - rowCosts[row]
                
                #remove assignment if increase column costs since equality will no longer hold
                if !iszero(c2r[jj])
                    r2c[c2r[jj]]  = zero(G)
                    c2r[jj] = zero(G)
                end
            end
        end
    end
    return nothing
end

function check_assignments!(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, nrow::Integer, ncol::Integer) where {G <: Integer, T <: AbstractFloat}
    nassigned = 0
    for ii in 1:nrow
        if !iszero(r2c[ii])
            if rewardMatrix[ii, r2c[ii]] == (rowCosts[ii] + colCosts[r2c[ii]])
                nassigned += 1
            else
                c2r[r2c[ii]] = zero(G)
                r2c[ii] = zero(G)
            end
        end
    end
    return nassigned
end

function check_assignments!(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, nrow::Integer, ncol::Integer) where {G <: Integer, T <: AbstractFloat}
    nassigned = 0
    for ii in 1:nrow
        if !iszero(r2c[ii])
            if rewardMatrix[ii, r2c[ii]] == (rowCosts[ii] + colCosts[r2c[ii]])
                nassigned += 1
            else
                c2r[r2c[ii]] = zero(G)
                r2c[ii] = zero(G)
            end
        end
    end
    return nassigned
end

function check_assignments!(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, shadowCosts::Array{T, 1}, ε::T, nrow::Integer, ncol::Integer) where {G <: Integer, T <: AbstractFloat}
    nassigned = 0
    for ii in 1:nrow
        if r2c[ii] == (ii + ncol) #remove cases assigned to shadow nodes as their status may have changed
            if (rowCosts[ii] + shadowCosts[ii]) == zero(T)
                nassigned += 1
            else
                r2c[ii] = zero(G)
            end
        elseif !iszero(r2c[ii])
            if rewardMatrix[ii, r2c[ii]] == (rowCosts[ii] + colCosts[r2c[ii]])
                nassigned += 1
            else
                c2r[r2c[ii]] = zero(G)
                r2c[ii] = zero(G)
            end
        end
    end
    return nassigned
end



function symmetric_forward_backward(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T; verbose::Bool = false, check::Bool = true) where {G <: Integer, T <: AbstractFloat}
    if size(rewardMatrix, 1) != size(rewardMatrix, 2)
        error("reward matrix must have the same number of rows and columns")
    end

    nrow, ncol = size(rewardMatrix)
    
    if check
        #first enforce greater than condition
        check_costs!(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, nrow, ncol)
        
        #second remove assignments that are not valid
        nassigned = check_assignments!(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, nrow, ncol)
        
    else
        nassigned = count(!iszero, r2c)
    end

    #find open rows
    urows = G[]
    for ii in 1:nrow
        if iszero(r2c[ii])
            push!(urows, ii)
        end
    end

    #here was only care about unassigned columns with prices > λ as others cannot be assigned
    ucols = G[]
    for jj in 1:ncol
        if iszero(c2r[jj])
            push!(ucols, jj)
        end
    end

    #Terminate if everything is assigned
    if nrow == nassigned
        return r2c, c2r, rowCosts, colCosts
    end
        
    #Initialize, looping down columns is more efficent in this phase
    for jj in 1:(nrow - nassigned)
        jj = shift!(ucols)
        r2c, c2r, rowCosts, colCosts, unassignedcol = backward_iteration(jj, r2c, c2r, rewardMatrix, rowCosts, colCosts, nrow, ε)
        if iszero(unassignedcol)
            nassigned += 1
        end
    end
    
    #Body...
    nstep = 1
    iter = 1
    converged = false
    while !converged
        if verbose
            println("nstep: $nstep, Assigned: $nassigned")
        end
        if nstep == 1
            #loop until additional assignment is made
            addition = false
            while !addition && length(urows) > 0
                ii = shift!(urows)
                if iszero(r2c[ii]) #need to check as when rows get assigned in reverse iteration there is no efficient way to delete them
                    r2c, c2r, rowCosts, colCosts, unassignedrow = forward_iteration(ii, r2c, c2r, rewardMatrix, rowCosts, colCosts, ncol, ε)
                    if iszero(unassignedrow)
                        addition = true
                    else
                        push!(urows, unassignedrow)
                    end
                end
            end
            if addition
                nassigned += 1
            end
            if nassigned == nrow
                converged = true
            else
                nstep = 2
            end
        elseif nstep == 2
            #loop until additional assignment is made
            addition = false
            while !addition && length(ucols) > 0
                jj = shift!(ucols)
                if iszero(c2r[jj])
                    r2c, c2r, rowCosts, colCosts, unassignedcol = backward_iteration(jj, r2c, c2r, rewardMatrix, rowCosts, colCosts, nrow, ε)
                    if iszero(unassignedcol)
                        addition = true
                    else
                        push!(ucols, unassignedcol)
                    end
                end
            end
            if addition
                nassigned += 1
            end
            if nassigned == nrow
                converged = true
            else
                nstep = 1
            end
        else
            error("unallowed step value")
        end
        iter += 1
    end
    return r2c, c2r, rowCosts, colCosts
end

function asymmetric_forward_backward(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, λ::T; verbose::Bool = false, check::Bool = true) where {G <: Integer, T <: AbstractFloat}
    nrow, ncol = size(rewardMatrix)
    nextracols = ncol - nrow
    nassigned = 0
    
    if check
        #first enforce greater than condition
        check_costs!(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, nrow, ncol)
        
        #second remove assignments that are not valid
        nassigned = check_assignments!(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, nrow, ncol)
    else
        nassigned = count(!iszero, r2c)
    end

    #find empty rows
    urows = G[]
    for ii in 1:nrow
        if iszero(r2c[ii])
            push!(urows, ii)
        end
    end

    #here we only care about unassigned columns with prices > λ as others cannot be assigned
    ucolsAbove = G[]
    for jj in 1:ncol
        if iszero(c2r[jj]) && colCosts[jj] > λ
            push!(ucolsAbove, jj)
        end
    end

    ##Set initial step
    if nrow == nassigned
        nstep = 3
    else
        nstep = 1
    end

    ncolbelowλ = count(x -> x < λ, colCosts)
    
    iter = 1
    converged = false
    while !converged
        if verbose
            println("nstep: $nstep, Assigned: $nassigned, Below Lambda: $ncolbelowλ, Lambda: $λ")
        end
        if nstep == 1
            #loop until additional assignment is made
            addition = false
            while !addition && length(urows) > 0
                ii = shift!(urows)
                if iszero(r2c[ii])
                    r2c, c2r, rowCosts, colCosts, unassignedrow, ncolbelowλ = forward_iteration(ii, r2c, c2r, rewardMatrix, rowCosts, colCosts, ncol, ncolbelowλ, ε, λ)
                    if iszero(unassignedrow)
                        addition = true
                    else
                        push!(urows, unassignedrow)
                    end
                end
            end
            if addition
                nassigned += 1
            end
            if nassigned == nrow
                nstep = 3
            else
                nstep = 2
            end
        elseif nstep == 2
            #loop until additional assignment is made
            addition = false
            while !addition && length(ucolsAbove) > 0
                jj = shift!(ucolsAbove)
                if iszero(c2r[jj]) && colCosts[jj] > λ
                    r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ = backward_iteration(jj, r2c, c2r, rewardMatrix, rowCosts, colCosts, nrow, ncolbelowλ, ε, λ)
                    
                    #update λ if enough colCosts are below it - should be faster than counting each time
                    if ncolbelowλ > nextracols
                        if verbose
                            println("updating λ")
                        end
                        scolCosts = sort(colCosts)
                        λ = scolCosts[nextracols] #change this?
                        ncolbelowλ = findprev(x -> x < λ, scolCosts, nextracols) #expect this to be nextracols - 1 unless there are ties, in which case it will be smaller
                    end
                    if iszero(unassignedcol)
                        addition = true
                    elseif colCosts[unassignedcol] > λ
                        push!(ucolsAbove, unassignedcol)
                    end
                end
            end
            
            #assignment found
            if addition
                nassigned += 1
            end
            if nassigned == nrow
                nstep = 3
            else
                nstep = 1
            end
            
        elseif nstep == 3
            while count(x -> x <= λ, colCosts) < nextracols
                if isempty(ucolsAbove)
                    ucolsAbove = find(1:ncol) do jj
                        colCosts[jj] > λ && iszero(c2r[jj])
                    end
                end 
                jj = shift!(ucolsAbove)
                if iszero(c2r[jj]) && colCosts[jj] > λ
                    r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ = backward_iteration(jj, r2c, c2r, rewardMatrix, rowCosts, colCosts, nrow, ncolbelowλ, ε, λ)
                    if !iszero(unassignedcol) && (colCosts[unassignedcol] > λ)
                        push!(ucolsAbove, unassignedcol)
                    end
                end
            end
            converged = true
        else
            error("unallowed step value")
        end
        if ncolbelowλ < nextracols && isempty(ucolsAbove)
            ncolbelowλ = count(x -> x < λ, colCosts)
            ucols = find(1:ncol) do jj
                iszero(c2r[jj]) && colCosts[jj] >= λ
            end
        end
        iter += 1
    end
    
    return r2c, c2r, rowCosts, colCosts, λ
end

function asymmetric_forward_backward(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::SparseMatrixCSC{T, G}, trewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, λ::T; verbose::Bool = false, check::Bool = true) where {G <: Integer, T <: AbstractFloat}
    nrow, ncol = size(rewardMatrix)
    nextracols = ncol - nrow
    nassigned = 0
    
    if check
        #first enforce greater than condition
        check_costs!(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, nrow, ncol)
        
        #second remove assignments that are not valid
        nassigned = check_assignments!(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, nrow, ncol)
    else
        nassigned = count(!iszero, r2c)
    end

    #for ii in 1:nrow
    #    if iszero(r2c[ii]) && length(nzrange(trewardMatrix, ii)) == 1
    #        r2c[ii] = rows[nzrange(trewardMatrix, ii)[1]]
    #        c2r[nzrange(trewardMatrix, ii)[1]] = ii #error here?????
    #        rowCosts[ii] = -λ
    #        colCosts[nzrange(trewardMatrix, ii)[1]] = λ
    #        nassigned += 1
    #    else
    #        r2c, c2r, rowCosts, colCosts, unassignedrow, ncolbelowλ = forward_iteration(ii, r2c, c2r, trewardMatrix, rowCosts, colCosts, ncol, ncolbelowλ, ε, λ)
    #    end
    #end
    
    #set rows with a single entry
    rows = rowvals(trewardMatrix)
    for ii in 1:nrow
        if iszero(r2c[ii]) && length(nzrange(trewardMatrix, ii)) == 1
            r2c[ii] = rows[nzrange(trewardMatrix, ii)[1]]
            c2r[nzrange(trewardMatrix, ii)[1]] = ii #error here?????
            rowCosts[ii] = -λ
            colCosts[nzrange(trewardMatrix, ii)[1]] = λ
            nassigned += 1
        end
    end
    
    #set empty columns so they are ignored
    for jj in 1:ncol
        if iszero(length(nzrange(rewardMatrix, jj)))
            colCosts[jj] = zero(T)
        end
    end

    #here we only care about unassigned columns with prices > λ as others cannot be assigned
    ucolsAbove = G[]
    for jj in 1:ncol
        if iszero(c2r[jj]) && colCosts[jj] > λ
            push!(ucolsAbove, jj)
        end
    end

    #ncolbelowλ = count(x -> x < λ, colCosts)
    ncolbelowλ = 0
    for jj in 1:ncol
        if colCosts[jj] < λ
            ncolbelowλ += 1
        end
    end

    #initialized each row
    for ii in 1:nrow
        if iszero(r2c[ii])
            r2c, c2r, rowCosts, colCosts, unassignedrow, ncolbelowλ = forward_iteration(ii, r2c, c2r, trewardMatrix, rowCosts, colCosts, ncol, ncolbelowλ, ε, λ)
            if iszero(unassignedrow)
                nassigned += 1
            end
        end
    end
    
    #find empty rows
    urows = G[]
    for ii in 1:nrow
        if iszero(r2c[ii])
            push!(urows, ii)
        end
    end

    ##Set initial step
    if nrow == nassigned
        nstep = 3
    else
        nstep = 1
    end
    
    iter = 1
    converged = false
    while !converged
        if verbose
            println("nstep: $nstep, Assigned: $nassigned, Below Lambda: $ncolbelowλ, Lambda: $λ")
        end
        if nstep == 1
            #loop until additional assignment is made
            addition = false
            while !addition && length(urows) > 0
                ii = shift!(urows)
                if iszero(r2c[ii])
                    r2c, c2r, rowCosts, colCosts, unassignedrow, ncolbelowλ = forward_iteration(ii, r2c, c2r, trewardMatrix, rowCosts, colCosts, ncol, ncolbelowλ, ε, λ)
                    if iszero(unassignedrow)
                        addition = true
                    else
                        push!(urows, unassignedrow)
                    end
                end
            end
            if addition
                nassigned += 1
            end
            if nassigned == nrow
                nstep = 3
            else
                nstep = 2
            end
        elseif nstep == 2
            #loop until additional assignment is made
            addition = false
            while !addition && length(ucolsAbove) > 0
                jj = shift!(ucolsAbove)
                if iszero(c2r[jj]) && colCosts[jj] > λ
                    r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ = backward_iteration(jj, r2c, c2r, rewardMatrix, rowCosts, colCosts, nrow, ncolbelowλ, ε, λ)
                    
                    #update λ if enough colCosts are below it - should be faster than counting each time
                    if ncolbelowλ > nextracols
                        if verbose
                            println("updating λ")
                        end
                        scolCosts = sort(colCosts)
                        λ = scolCosts[nextracols]
                        ncolbelowλ = findprev(x -> x < λ, scolCosts, nextracols) #expect this to be nextracols - 1 unless there are ties, in which case it will be smaller
                    end
                    if iszero(unassignedcol)
                        addition = true
                    elseif colCosts[unassignedcol] > λ
                        push!(ucolsAbove, unassignedcol)
                    end
                end
            end
            
            #assignment found
            if addition
                nassigned += 1
            end
            if nassigned == nrow
                nstep = 3
            else
                nstep = 1
            end
            
        elseif nstep == 3
            while count(x -> x <= λ, colCosts) < nextracols
                if isempty(ucolsAbove)
                    ucolsAbove = find(1:ncol) do jj
                        colCosts[jj] > λ && iszero(c2r[jj])
                    end
                end 
                jj = shift!(ucolsAbove)
                if iszero(c2r[jj]) && colCosts[jj] > λ
                    r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ = backward_iteration(jj, r2c, c2r, rewardMatrix, rowCosts, colCosts, nrow, ncolbelowλ, ε, λ)
                    if !iszero(unassignedcol) && (colCosts[unassignedcol] > λ)
                        push!(ucolsAbove, unassignedcol)
                    end
                end
            end
            converged = true
        else
            error("unallowed step value")
        end
        if ncolbelowλ < nextracols && isempty(ucolsAbove)
            ncolbelowλ = count(x -> x < λ, colCosts)
            ucols = find(1:ncol) do jj
                iszero(c2r[jj]) && colCosts[jj] >= λ
            end
        end
        iter += 1
    end
    
    return r2c, c2r, rowCosts, colCosts, λ
end

function asymmetric_forward_backward_shadow(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::SparseMatrixCSC{T, G}, trewardMatrix::SparseMatrixCSC{T, G}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, shadowCosts::Array{T, 1}, ε::T, λ::T; verbose::Bool = false, check::Bool = true) where {G <: Integer, T <: AbstractFloat}
    nrow, ncol = size(rewardMatrix)
    nextracols = ncol - nrow
    nassigned = zero(G)
    
    if check
        #first enforce greater than condition
        check_costs!(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, nrow, ncol)
        
        #second remove assignments that are not valid
        nassigned = check_assignments!(r2c, c2r, rewardMatrix, rowCosts, colCosts, shadowCosts, ε, nrow, ncol)
    else
        nassigned = count(!iszero, r2c)
    end

    #set empty rows to shadow zeros
    for ii in 1:nrow
        if iszero(r2c[ii]) && iszero(length(nzrange(trewardMatrix, ii)))
            shadowCosts[ii] = λ
            rowCosts[ii] = -λ
            r2c[ii] = ii + ncol
            nassigned += one(G)
        end
    end
    
    #set empty columns so they are ignored
    for jj in 1:ncol
        if colCosts[jj] > zero(T) &&iszero(length(nzrange(rewardMatrix, jj)))
            colCosts[jj] = zero(T)
        end
    end

    #assigned to shadow entries
    nshadow = 0
    for ii in 1:nrow
        if r2c[ii] > ncol
            nshadow += 1
        end
    end
    
    #find empty rows
    urows = G[]
    for ii in 1:nrow
        if iszero(r2c[ii])
            push!(urows, ii)
        end
    end

    #here we only care about unassigned columns with prices > λ as others cannot be assigned
    ucolsAbove = G[]
    ncolbelowλ = zero(G)
    for jj in 1:ncol
        if colCosts[jj] < λ
            ncolbelowλ += one(G)
        elseif iszero(c2r[jj]) && colCosts[jj] > λ
            push!(ucolsAbove, jj)
        end
    end

    ##Set initial step
    if nrow == nassigned
        nstep = 3
    else
        nstep = 1
    end
    
    iter = 1
    converged = false
    while !converged
        if verbose
            println("nstep: $nstep, Assigned: $nassigned, Below Lambda: $ncolbelowλ, Lambda: $λ")
        end
        if nstep == 1
            #loop until additional assignment is made
            addition = false
            while !addition && length(urows) > 0
                ii = shift!(urows)
                if iszero(r2c[ii])
                    r2c, c2r, rowCosts, colCosts, shadowCosts, unassignedrow, ncolbelowλ, nshadow = forward_iteration_shadow(ii, r2c, c2r, trewardMatrix, rowCosts, colCosts, shadowCosts, ncol, ncolbelowλ, nshadow, ε, λ)
                    if iszero(unassignedrow)
                        addition = true
                    else
                        push!(urows, unassignedrow)
                    end
                end
            end
            if addition
                nassigned += 1
            end
            if nassigned == nrow
                nstep = 3
            else
                nstep = 2
            end
        elseif nstep == 2
            #loop until additional assignment is made
            addition = false
            while !addition && length(ucolsAbove) > 0
                jj = shift!(ucolsAbove)
                if iszero(c2r[jj]) && colCosts[jj] > λ
                    r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ = backward_iteration_shadow(jj, r2c, c2r, rewardMatrix, rowCosts, colCosts, nrow, ncolbelowλ, ε, λ)
                    
                    #update λ if enough colCosts are below it - should be faster than counting each time
                    if ncolbelowλ > (nextracols + nshadow) #shadow assigments count as they implicitly add columns, should not enter this step if nextracols + nshadow == 0
                        if verbose
                            println("updating λ")
                        end
                        scolCosts = sort(colCosts)
                        λ = scolCosts[nextracols  + nshadow]
                        ncolbelowλ = findprev(x -> x < λ, scolCosts, nextracols  + nshadow) #expect this to be nextracols - 1 unless there are ties, in which case it will be smaller
                    end
                    if iszero(unassignedcol)
                        addition = true
                    elseif unassignedcol <= ncol && colCosts[unassignedcol] > λ
                        push!(ucolsAbove, unassignedcol)
                    end
                end
            end
            
            #assignment found
            if addition
                nassigned += 1
            end
            if nassigned == nrow
                nstep = 3
            else
                nstep = 1
            end
            
        elseif nstep == 3
            while count(x -> x <= λ, colCosts) < (nextracols + nshadow)
                if isempty(ucolsAbove)
                    ucolsAbove = find(1:ncol) do jj
                        colCosts[jj] > λ && iszero(c2r[jj])
                    end
                end 
                jj = shift!(ucolsAbove)
                if iszero(c2r[jj]) && colCosts[jj] > λ
                    r2c, c2r, rowCosts, colCosts, unassignedcol, ncolbelowλ = backward_iteration_shadow(jj, r2c, c2r, rewardMatrix, rowCosts, colCosts, nrow, ncolbelowλ, ε, λ)
                    if !iszero(unassignedcol) && (colCosts[unassignedcol] > λ)
                        push!(ucolsAbove, unassignedcol)
                    end
                end
            end
            converged = true
        else
            error("unallowed step value")
        end
        if ncolbelowλ < nextracols && isempty(ucolsAbove)
            ncolbelowλ = count(x -> x < λ, colCosts)
            ucols = find(1:ncol) do jj
                iszero(c2r[jj]) && colCosts[jj] >= λ
            end
        end
        iter += 1
    end
    
    return r2c, c2r, rowCosts, colCosts, shadowCosts, λ
end

function symmetric_scaling_forward_backward(rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2; verbose::Bool = false, check::Bool = true) where T <: AbstractFloat
    r2c = zeros(Int, size(rewardMatrix, 1))
    c2r = zeros(Int, size(rewardMatrix, 2))
    
    ε = ε0
    
    #Initial
    if verbose
        println("Running with ε = $ε")
    end
    r2c, c2r, rowCosts, colCosts = symmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, verbose = verbose, check = false)

    while ε >= εfinal
        if verbose
            println("Scaling ε")
        end
        εnew = ε * εscale
        decr = ε - εnew
        for ii in 1:size(rewardMatrix, 1)
            rowCosts[ii] = rowCosts[ii] + decr
        end
        ε = εnew
        if verbose
            println("Running with ε = $ε")
        end
        if check
            r2c, c2r, rowCosts, colCosts = symmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, verbose = verbose, check = true)
        else
            r2c = zeros(Int, size(rewardMatrix, 1))
            c2r = zeros(Int, size(rewardMatrix, 2))
            r2c, c2r, rowCosts, colCosts = symmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, verbose = verbose, check = false)
        end
    end
    return r2c, c2r, rowCosts, colCosts
end

function symmetric_scaling_forward_backward(rewardMatrix::Array{T, 2}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2; verbose::Bool = false, check::Bool = false) where T <: AbstractFloat
    #rowCosts = vec(minimum(rewardMatrix, 2))
    rowCosts = vec(maximum(rewardMatrix, 2))
    colCosts = zeros(T, size(rewardMatrix, 2))
    return symmetric_scaling_forward_backward(rewardMatrix, rowCosts, colCosts, ε0, εfinal, εscale, verbose = verbose, check = check)
end

function asymmetric_scaling_forward_backward(rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = true) where T <: AbstractFloat
    if size(rewardMatrix, 1) >= size(rewardMatrix, 2)
        error("number of rows must be less than number of columns")
    end
    r2c = zeros(Int, size(rewardMatrix, 1))
    c2r = zeros(Int, size(rewardMatrix, 2))
    
    ε = ε0
    
    #Initial
    if verbose
        println("Running with ε = $ε")
    end
    r2c, c2r, rowCosts, colCosts, λ = asymmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, λ, verbose = verbose, check = false)

    while ε >= εfinal
        if verbose
            println("Scaling ε")
        end
        εnew = ε * εscale
        decr = ε - εnew
        for ii in 1:size(rewardMatrix, 1)
            rowCosts[ii] = rowCosts[ii] + decr
        end
        ε = εnew
        λ = minimum(colCosts[.!iszero.(c2r)])

        if verbose
            println("Running with ε = $ε")
        end
        
        if check
            r2c, c2r, rowCosts, colCosts, λ = asymmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, λ, verbose = verbose, check = true)
        else
            r2c = zeros(Int, size(rewardMatrix, 1))
            c2r = zeros(Int, size(rewardMatrix, 2))
            r2c, c2r, rowCosts, colCosts = asymmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, λ, verbose = verbose, check = false)
        end
    end
    return r2c, c2r, rowCosts, colCosts, λ
end

function asymmetric_scaling_forward_backward(rewardMatrix::Array{T, 2}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false) where T <: AbstractFloat
    #rowCosts = vec(minimum(rewardMatrix, 2))
    rowCosts = vec(maximum(rewardMatrix, 2))
    colCosts = zeros(T, size(rewardMatrix, 2))
    return asymmetric_scaling_forward_backward(rewardMatrix, rowCosts, colCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
end

function asymmetric_scaling_forward_backward(rewardMatrix::SparseMatrixCSC{T}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = true) where T <: AbstractFloat
    if size(rewardMatrix, 1) >= size(rewardMatrix, 2)
        error("number of rows must be less than number of columns")
    end
    r2c = zeros(Int, size(rewardMatrix, 1))
    c2r = zeros(Int, size(rewardMatrix, 2))
    
    ε = ε0
    trewardMatrix = rewardMatrix'
    
    #Initial
    if verbose
        println("Running with ε = $ε")
    end
    r2c, c2r, rowCosts, colCosts, λ = asymmetric_forward_backward(r2c, c2r, rewardMatrix, trewardMatrix, rowCosts, colCosts, ε, λ, verbose = verbose, check = false)

    while ε >= εfinal
        if verbose
            println("Scaling ε")
        end
        εnew = ε * εscale
        decr = ε - εnew
        for ii in 1:size(rewardMatrix, 1)
            rowCosts[ii] = rowCosts[ii] + decr
        end
        ε = εnew
        λ = minimum(colCosts[.!iszero.(c2r)])

        if verbose
            println("Running with ε = $ε")
        end
        
        if check
            r2c, c2r, rowCosts, colCosts, λ = asymmetric_forward_backward(r2c, c2r, rewardMatrix, trewardMatrix, rowCosts, colCosts, ε, λ, verbose = verbose, check = true)
        else
            r2c = zeros(Int, size(rewardMatrix, 1))
            c2r = zeros(Int, size(rewardMatrix, 2))
            r2c, c2r, rowCosts, colCosts = asymmetric_forward_backward(r2c, c2r, rewardMatrix, trewardMatrix, rowCosts, colCosts, ε, λ, verbose = verbose, check = false)
        end
    end
    return r2c, c2r, rowCosts, colCosts, λ
end

function asymmetric_scaling_forward_backward(rewardMatrix::SparseMatrixCSC{T}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false) where T <: AbstractFloat
    #rowCosts = vec(minimum(rewardMatrix, 2))
    rowCosts = vec(maximum(rewardMatrix, 2))
    colCosts = zeros(T, size(rewardMatrix, 2))
    return asymmetric_scaling_forward_backward(rewardMatrix, rowCosts, colCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
end

function asymmetric_scaling_forward_backward_shadow(rewardMatrix::SparseMatrixCSC{T}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, shadowCosts::Array{T, 1}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = true) where T <: AbstractFloat
    #Implicitly the matrix is nrow by (ncol + nrow)
    #if size(rewardMatrix, 1) >= size(rewardMatrix, 2)
    #    error("number of rows must be less than number of columns")
    #end
    r2c = zeros(Int, size(rewardMatrix, 1))
    c2r = zeros(Int, size(rewardMatrix, 2))
    
    ε = ε0
    trewardMatrix = rewardMatrix'
    
    #Initial
    if verbose
        println("Running with ε = $ε")
    end
    r2c, c2r, rowCosts, colCosts, λ = asymmetric_forward_backward_shadow(r2c, c2r, rewardMatrix, trewardMatrix, rowCosts, colCosts, shadowCosts, ε, λ, verbose = verbose, check = false)

    while ε >= εfinal
        if verbose
            println("Scaling ε")
        end
        εnew = ε * εscale
        decr = ε - εnew
        for ii in 1:size(rewardMatrix, 1)
            rowCosts[ii] = rowCosts[ii] + decr
        end
        ε = εnew
        λ = minimum(colCosts[.!iszero.(c2r)])

        if verbose
            println("Running with ε = $ε")
        end
        
        if check
            r2c, c2r, rowCosts, colCosts, λ = asymmetric_forward_backward_shadow(r2c, c2r, rewardMatrix, trewardMatrix, rowCosts, colCosts, shadowCosts, ε, λ, verbose = verbose, check = true)
        else
            r2c = zeros(Int, size(rewardMatrix, 1))
            c2r = zeros(Int, size(rewardMatrix, 2))
            r2c, c2r, rowCosts, colCosts = asymmetric_forward_backward_shadow(r2c, c2r, rewardMatrix, trewardMatrix, rowCosts, colCosts, shadowCosts, ε, λ, verbose = verbose, check = false)
        end
    end
    return r2c, c2r, rowCosts, colCosts, shadowCosts, λ
end

function asymmetric_scaling_forward_backward_shadow(rewardMatrix::SparseMatrixCSC{T}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false) where T <: AbstractFloat
    rowCosts = vec(maximum(rewardMatrix, 2))
    colCosts = zeros(T, size(rewardMatrix, 2))
    shadowCosts = zeros(T, size(rewardMatrix, 1))
    return asymmetric_scaling_forward_backward_shadow(rewardMatrix, rowCosts, colCosts, shadowCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
end

function scaling_forward_backward(rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false) where T <: AbstractFloat
    if size(rewardMatrix, 1) == size(rewardMatrix, 2)
        return symmetric_scaling_forward_backward(rewardMatrix, rowCosts, colCosts, ε0, εfinal, εscale, verbose = verbose, check = check)..., λ
    elseif size(rewardMatrix, 1) < size(rewardMatrix, 2)
        return asymmetric_scaling_forward_backward(rewardMatrix, rowCosts, colCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
    else
        c2r, r2c, colCosts, rowCosts, λ = asymmetric_scaling_forward_backward(rewardMatrix', colCosts, rowCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
        return r2c, c2r, rowCosts, colCosts, λ
    end
end

function scaling_forward_backward(rewardMatrix::Array{T, 2}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false) where T <: AbstractFloat
    if size(rewardMatrix, 1) == size(rewardMatrix, 2)
        return symmetric_scaling_forward_backward(rewardMatrix, ε0, εfinal, εscale, verbose = verbose, check = check)..., λ
    elseif size(rewardMatrix, 1) < size(rewardMatrix, 2)
        return asymmetric_scaling_forward_backward(rewardMatrix, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
    else
        c2r, r2c, colCosts, rowCosts, λ = asymmetric_scaling_forward_backward(rewardMatrix', ε0, εfinal, εscale, λ, verbose = verbose, check = check)
        return r2c, c2r, rowCosts, colCosts, λ
    end
end

function scaling_forward_backward(rewardMatrix::SparseMatrixCSC{T}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false) where T <: AbstractFloat
    if size(rewardMatrix, 1) >= size(rewardMatrix, 2)
        error("assume fewer rows than columns")
    elseif size(rewardMatrix, 1) < size(rewardMatrix, 2)
        return asymmetric_scaling_forward_backward(rewardMatrix, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
    else
        c2r, r2c, colCosts, rowCosts, λ = asymmetric_scaling_forward_backward(rewardMatrix', ε0, εfinal, εscale, λ, verbose = verbose, check = check)
        return r2c, c2r, rowCosts, colCosts, λ
    end
end

function scaling_forward_backward_shadow(rewardMatrix::SparseMatrixCSC{T}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, shadowCosts::Array{T, 1}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false) where T <: AbstractFloat
    if size(rewardMatrix, 1) <= size(rewardMatrix, 2)
        return asymmetric_scaling_forward_backward_shadow(rewardMatrix, rowCosts, colCosts, shadowCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
    else
        c2r, r2c, colCosts, rowCosts, shadowCosts, λ = asymmetric_scaling_forward_backward_shadow(rewardMatrix', colCosts, rowCosts, shadowCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check) #definition of shadowCosts here it a little weird
        return r2c, c2r, rowCosts, colCosts, shadowCosts, λ
    end
end

function scaling_forward_backward_shadow(rewardMatrix::SparseMatrixCSC{T}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false) where T <: AbstractFloat
    if size(rewardMatrix, 1) <= size(rewardMatrix, 2)
        return asymmetric_scaling_forward_backward_shadow(rewardMatrix, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
    else
        c2r, r2c, colCosts, rowCosts, λ = asymmetric_scaling_forward_backward_shadow(rewardMatrix', ε0, εfinal, εscale, λ, verbose = verbose, check = check)
        return r2c, c2r, rowCosts, colCosts, λ
    end
end

function add_dummy_entries(rewardMatrix::SparseMatrixCSC{T}) where T <: AbstractFloat
    nrow, ncol = size(rewardMatrix)
    rows, cols, vals = findnz(rewardMatrix)
    return sparse([rows; 1:nrow], [cols; range(ncol + 1, nrow)], [vals; fill(zero(T), nrow)], nrow, nrow + ncol)
end
