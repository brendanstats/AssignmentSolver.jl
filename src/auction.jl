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

#not complete
function maxthreeRow{T <: AbstractFloat}(row::Integer, rewardMatrix::Array{T, 2}, colCosts::Array{T, 1}, ncol::Integer)
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
function forward_iteration{G <: Integer, T <: AbstractFloat}(row::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ncol::G, ε::T)
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
function forward_iteration{G <: Integer, T <: AbstractFloat}(row::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ncol::G, ncolbelowλ::G, ε::T, λ::T)
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

function backward_iteration{G <: Integer, T <: AbstractFloat}(col::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, nrow::G, ε::T)
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

function backward_iteration{G <: Integer, T <: AbstractFloat}(col::G, r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, nrow::G, ncolbelowλ::G, ε::T, λ::T)
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

function check_costs!{G <: Integer, T <: AbstractFloat}(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, nrow::Integer, ncol::Integer)
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

function check_assignments!{G <: Integer, T <: AbstractFloat}(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, nrow::Integer, ncol::Integer)
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

function symmetric_forward_backward{G <: Integer, T <: AbstractFloat}(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T; verbose::Bool = false, check::Bool = true)
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

function asymmetric_forward_backward{G <: Integer, T <: AbstractFloat}(r2c::Array{G, 1}, c2r::Array{G, 1}, rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε::T, λ::T; verbose::Bool = false, check::Bool = true)
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

function symmetric_scaling_forward_backward{T <: AbstractFloat}(rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2; verbose::Bool = false, check::Bool = true)
    r2c = zeros(Int, size(rewardMatrix, 1))
    c2r = zeros(Int, size(rewardMatrix, 2))
    
    ε = ε0
    
    #Initial
    if verbose
        println("Running with ε = $ε")
    end
    r2c, c2r, rowCosts, colCosts = symmetric_forward_backward(r2c, c2r, rewardMatrix, rowCosts, colCosts, ε, verbose = verbose, check = false)

    while ε > εfinal
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

function symmetric_scaling_forward_backward{T <: AbstractFloat}(rewardMatrix::Array{T, 2}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2; verbose::Bool = false, check::Bool = false)
    rowCosts = vec(minimum(rewardMatrix, 2))
    colCosts = zeros(T, size(rewardMatrix, 2))
    return symmetric_scaling_forward_backward(rewardMatrix, rowCosts, colCosts, ε0, εfinal, εscale, verbose = verbose, check = check)
end

function asymmetric_scaling_forward_backward{T <: AbstractFloat}(rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = true)
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

    while ε > εfinal
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

function asymmetric_scaling_forward_backward{T <: AbstractFloat}(rewardMatrix::Array{T, 2}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false)
    rowCosts = vec(minimum(rewardMatrix, 2))
    colCosts = zeros(T, size(rewardMatrix, 2))
    return asymmetric_scaling_forward_backward(rewardMatrix, rowCosts, colCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
end


function scaling_forward_backward{T <: AbstractFloat}(rewardMatrix::Array{T, 2}, rowCosts::Array{T, 1}, colCosts::Array{T, 1}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false)
    if size(rewardMatrix, 1) == size(rewardMatrix, 2)
        return symmetric_scaling_forward_backward(rewardMatrix, rowCosts, colCosts, ε0, εfinal, εscale, verbose = verbose, check = check)..., λ
    elseif size(rewardMatrix, 1) < size(rewardMatrix, 2)
        return asymmetric_scaling_forward_backward(rewardMatrix, rowCosts, colCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
    else
        c2r, r2c, colCosts, rowCosts, λ = asymmetric_scaling_forward_backward(rewardMatrix', colCosts, rowCosts, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
        return r2c, c2r, rowCosts, colCosts, λ
    end
end

function scaling_forward_backward{T <: AbstractFloat}(rewardMatrix::Array{T, 2}, ε0::T, εfinal::T = 1.0 / size(rewardMatrix, 1), εscale::T = 0.2, λ::T = 0.0; verbose::Bool = false, check::Bool = false)
    if size(rewardMatrix, 1) == size(rewardMatrix, 2)
        return symmetric_scaling_forward_backward(rewardMatrix, ε0, εfinal, εscale, verbose = verbose, check = check)..., λ
    elseif size(rewardMatrix, 1) < size(rewardMatrix, 2)
        return asymmetric_scaling_forward_backward(rewardMatrix, ε0, εfinal, εscale, λ, verbose = verbose, check = check)
    else
        c2r, r2c, colCosts, rowCosts, λ = asymmetric_scaling_forward_backward(rewardMatrix', ε0, εfinal, εscale, λ, verbose = verbose, check = check)
        return r2c, c2r, rowCosts, colCosts, λ
    end
end
