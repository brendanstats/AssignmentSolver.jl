#β - Bertsekas 1988 pdf 20
#implement third best prices

#number of column costs below λ
#number of unassigned columnCosts above λ

"""
Find the index of the maximum adjusted reward in the row and the amount this exceeds the second largest value
"""
function maxtwoRow{T <: AbstractFloat}(row::Integer, rewardMatrix::Array{T, 2}, colCosts::Array{T, 1}, ncol::Integer)
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

function maxtwoCol{G <: Integer, T <: AbstractFloat}(col::G, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, nrow::G)
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

"""
forward iteration for a single un-assigned row in a symmetric auction assignment algorithm
"""
function forward_iteration{G <: Integer, T <: AbstractFloat}(row::G, r2c::Array{G, 1}, c2r::Array{G, 1},
                                                             rewardMatrix::Array{T, 2}, 
                                                             rowCosts::Array{T, 1}, colCosts::Array{T, 1},
                                                             ε::T, ncol::G)
    maxcol, maxval, maxtwo = maxtwoRow(row, rewardMatrix, colCosts, ncol)
    margin = maxval - maxtwo + ε
    colCosts[maxcol] += margin
    rowCosts[row] = rewardMatrix[row, maxcol] - colCosts[maxcol]
    if iszero(c2r[maxcol]) #only generates an additional assignment if maxcol is unassigned
        addition = true
    else
        r2c[c2r[maxcol]] = zero(G)
        addition = false
    end
    r2c[row] = maxcol
    c2r[maxcol] = row
    
    return r2c, c2r, rowCosts, colCosts, addition
end

"""
forward iteration for a single un-assigned row in an asymmetric auction assignment algorithm
"""
function forward_iteration{G <: Integer, T <: AbstractFloat}(row::G, r2c::Array{G, 1}, c2r::Array{G, 1},
                                                             rewardMatrix::Array{T, 2}, 
                                                             rowCosts::Array{T, 1}, colCosts::Array{T, 1},
                                                             ε::T, λ::T, ncol::G)
    maxcol, maxval, maxtwo = maxtwoRow(row, rewardMatrix, colCosts, ncol)
    margin = maxtwo - ε
    adj = rewardMatrix[row, maxcol] - margin
    if λ <= adj
        colCosts[maxcol] = adj
        if iszero(c2r[maxcol]) #only generates an additional assignment if maxcol is unassigned)
            addition = true
        else
            r2c[c2r[maxcol]] = zero(G)
            addition = false
        end
        r2c[row] = maxcol
        c2r[maxcol] = row
    else
        colCosts[maxcol] = λ
        addition = false
    end
    rowCosts[row] = margin
    return r2c, c2r, rowCosts, colCosts, addition
end

function backward_iteration{G <: Integer, T <: AbstractFloat}(col::G, r2c::Array{G, 1}, c2r::Array{G, 1},
                                                              rewardMatrix::Array{T, 2},
                                                              rowCosts::Array{T, 1}, colCosts::Array{T, 1},
                                                              ε::T, nrow::G)
    maxrow, maxval, maxtwo = maxtwoCol(col, rewardMatrix, rowCosts, nrow)
    margin = maxval - maxtwo + ε
    rowCosts[maxrow] += margin
    colCosts[col] = rewardMatrix[maxrow, col] - rowCosts[maxrow]
    if iszero(r2c[maxrow]) #only generates an additional assignment if maxrow is unassigned
        addition = true
    else
        c2r[r2c[maxrow]] = zero(G)
        addition = false
    end    
    
    r2c[maxrow] = col
    c2r[col] = maxrow
    
    return r2c, c2r, rowCosts, colCosts, addition
end

function backward_iteration{G <: Integer, T <: AbstractFloat}(col::G, r2c::Array{G, 1}, c2r::Array{G, 1},
                                                              rewardMatrix::Array{T, 2},
                                                              rowCosts::Array{T, 1}, colCosts::Array{T, 1},
                                                              ε::T, λ::T, nrow::G)
    maxrow, maxval, maxtwo = maxtwoCol(col, rewardMatrix, rowCosts, nrow)
    margin = maxtwo - ε
    if maxval >= (λ + ε)
        colCosts[col] = max(λ, margin)
        rowCosts[maxrow] = rewardMatrix[maxrow, col] - max(λ, margin)
        if iszero(r2c[maxrow])
            addition = true
        else
            c2r[r2c[maxrow]] = zero(G)
            addition = false
        end
        r2c[maxrow] = col
        c2r[col] = maxrow
        belowλ = false
    else
        addition = false
        if colCosts[col] >= λ && (maxval - ε) < λ
            belowλ = true
        else
            belowλ = false
        end
        colCosts[col] = maxval - ε
    end
    if belowλ
        if count(colCosts .< λ) > (size(rewardMatrix, 2) - nrow)
            perm = sortperm(colCosts)
            λ = colCosts[perm[size(rewardMatrix, 2) - nrow]]
        end
    end
    return r2c, c2r, rowCosts, colCosts, λ, addition
end

function forward_backward{T <: AbstractFloat}(rewardMatrix::Array{T, 2}, ε::T, λ::T = 0.0; verbose::Bool = false)
    nrow, ncol = size(rewardMatrix)
    r2c = zeros(Int, nrow)
    c2r = zeros(Int, ncol)
    rowCosts = zeros(T, nrow)
    colCosts = zeros(T, ncol)
    if nrow == ncol
        return symmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, verbose = verbose, check = false)
    elseif nrow < ncol
        return asymmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, λ, verbose = verbose, check = false)
    else
        r2c, c2r, rowCosts, colCosts = asymmetric_forward_backward(c2r, r2c, rewardMatrix', colCosts, rowCosts, ε, λ, verbose = verbose, check = false)
        return c2r, r2c, colCosts, rowCosts
    end
end

function forward_backward{T <: AbstractFloat}(rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, λ::T = 0.0; verbose::Bool = false, check::Bool = false)
    nrow, ncol = size(rewardMatrix)
    r2c = zeros(Int, nrow)
    c2r = zeros(Int, ncol)
    if nrow == ncol
        return symmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, verbose = verbose, check = check)
    elseif nrow < ncol
        return asymmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, λ, verbose = verbose, check = check)
    else
        r2c, c2r, rowCosts, colCosts = asymmetric_forward_backward(c2r, r2c, rewardMatrix', colCosts, rowCosts, ε, λ, verbose = verbose, check = check)
        return c2r, r2c, colCosts, rowCosts
    end
end

function symmetric_forward_backward{G <: Integer, T <: AbstractFloat}(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T; verbose::Bool = false, check::Bool = true)
    if size(rewardMatrix, 1) != size(rewardMatrix, 2)
        error("reward matrix must have the same number of rows and columns")
    end

    nrow, ncol = size(rewardMatrix)
    nassigned = 0
    
    if check
        #first enforce greater than condition
        for jj in 1:ncol, ii in 1:nrow
            if (rewardMatrix[ii, jj] - ε) > (rowCosts[ii] + colCosts[jj])
                colCosts[jj] = rewardMatrix[ii, jj] - ε - rowCosts[ii]
                if !iszero(c2r[jj])
                    c2r[jj] = zero(G)
                end
                
                if !iszero(r2c[ii])
                    r2c[ii] = zero(G)
                end
            end
        end

        #second remove assignments that are not longer valid
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
    end
    
    addition = false
    
    #Initialize, looping down columns is more efficent in this phase
    for jj in 1:ncol
        r2c, c2r, rowCosts, colCosts, addition = backward_iteration(jj, r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, nrow)
        if addition
            nassigned += 1
        end
    end
    
    #Terminate if everything is assigned
    if nrow == nassigned
        return r2c, c2r, rowCosts, colCosts
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
            ii = 1
            while !addition
                if iszero(r2c[ii])
                    r2c, c2r, rowCosts, colCosts, addition = forward_iteration(ii, r2c, c2r, rewardMatrix,
                                                                               rowCosts, colCosts, ε, ncol)
                end
                ii += 1
                if ii > nrow
                    ii = 1
                end
            end
            nassigned += 1
            if nassigned == nrow
                converged = true
            else
                nstep = 2
            end
        elseif nstep == 2
            #loop until additional assignment is made
            addition = false
            jj = 1
            while !addition
                if iszero(c2r[jj])
                    r2c, c2r, rowCosts, colCosts, addition = backward_iteration(jj, r2c, c2r, rewardMatrix,
                                                                                rowCosts, colCosts, ε, nrow)
                end
                jj += 1
                if jj > ncol
                    jj = 1
                end
            end
            nassigned += 1
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

function asymmetric_forward_backward{G <: Integer, T <: AbstractFloat}(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, λ::T; verbose::Bool = false, check::Bool = true)
    nrow, ncol = size(rewardMatrix)
    nassigned = 0
    
    if check
        #first enforce greater than condition
        for jj in 1:ncol, ii in 1:nrow
            if (rewardMatrix[ii, jj] - ε) > (rowCosts[ii] + colCosts[jj])
                colCosts[jj] = rewardMatrix[ii, jj] - ε - rowCosts[ii]
                if !iszero(c2r[jj])
                    c2r[jj] = zero(G)
                end
                
                if !iszero(r2c[ii])
                    r2c[ii] = zero(G)
                end
            end
        end
        
        #second remove assignments that are not longer valid
        for ii in 1:nrow
            if !iszero(r2c[ii])
                #colCosts[r2c[jj]] >= maxunassigned && 
                if rewardMatrix[ii, r2c[ii]] == (rowCosts[ii] + colCosts[r2c[ii]])
                    nassigned += 1
                else
                    c2r[r2c[ii]] = zero(G)
                    r2c[ii] = zero(G)
                end
            end
        end
    end

    
    
    #do something to initialize λ
    uRows = Int[]
    for ii in 1:nrow
        if iszero(r2c[ii])
            push!(uRows, ii)
        end
    end
    uColsAbove = Int[]
    for jj in 1:ncol
        if iszero(c2r[jj]) && colCosts[jj] > λ
            push!(uCols, jj)
        end
    end
    
    addition = false
    
    if nrow == nassigned
        nstep = 3
    else
        nstep = 1
    end
    
    iter = 1
    converged = false
    while !converged
        if verbose
            println("nstep: $nstep, Assigned: $nassigned")
        end
        if nstep == 1
            #loop until additional assignment is made
            addition = false
            ii = 1
            while !addition
                if iszero(r2c[ii])
                    r2c, c2r, rowCosts, colCosts, addition = forward_iteration(ii, r2c, c2r, rewardMatrix,
                                                                               rowCosts, colCosts, ε, λ, ncol)
                end
                ii += 1
                if ii > nrow
                    ii = 1
                end
            end
            nassigned += 1
            if nassigned == nrow
                nstep = 3
            else
                nstep = 2
            end
        elseif nstep == 2
            #loop until additional assignment is made
            addition = false
            jj = 1
            while !addition
                if iszero(c2r[jj]) && colCosts[jj] > λ
                    r2c, c2r, rowCosts, colCosts, λ, addition = backward_iteration(jj, r2c, c2r, rewardMatrix,
                                                                                rowCosts, colCosts, ε, λ, nrow)
                end
                jj += 1
                if jj > ncol
                    jj = 1
                end
            end
            nassigned += 1
            if nassigned == nrow
                nstep = 3
            else
                nstep = 1
            end
        elseif nstep == 3
            converged = true
            for jj in 1:ncol
                if iszero(c2r[jj]) && colCosts[jj] > λ
                    converged = false
                    r2c, c2r, rowCosts, colCosts, λ, addition = backward_iteration(jj, r2c, c2r, rewardMatrix,
                                                                                   rowCosts, colCosts, ε, λ, nrow)
                end
            end
        else
            error("unallowed step value")
        end
        iter += 1
    end
    return r2c, c2r, rowCosts, colCosts, λ
end
