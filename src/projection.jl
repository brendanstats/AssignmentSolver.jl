function project_vecnorm(C::Array{<:Real, 1})
    perm = sortperm(C, rev = true)
    idx = 1
    tot = C[perm[idx]]
    while idx * (C[perm[idx]]) < 1.0
        idx += 1
        tot += C[perm[idx]]
    end
    if tot <= 1.0
        return C
    else
        α = 1.0 / tot
        for ii in 1:idx
            C[perm[ii]] *= α
        end
        for ii in (idx + 1):length(C)
            C[perm[ii]] = 0.0
        end
        return C
    end
end

function projection{G <: Real}(x::Array{G, 1}, z::G = one(G))
    u = sort(x, rev = true)
    ρ = length(u)
    tot = sum(u) - z
    while u[ρ] - (1.0 / ρ) * tot <= 0
        tot -= u[ρ]
        ρ -= 1
    end
    θ = (1.0 / ρ) * tot
    w = x .- θ
    w[x .< θ] = 0.0
    return w
end

map(ii -> projection(x[ii, :]), 1:size(x, 1))

function projection_row{G <: Real}(x::Array{G, 2}, z::G = one(G))
    return mapreduce(ii -> projection(x[ii, :])', vcat, 1:size(x, 1))
end

function projection_col{G <: Real}(x::Array{G, 2}, z::G = one(G))
    return mapreduce(jj -> projection(x[:, jj]), hcat, 1:size(x, 2))
end


delta = [1 2; 2 4] ./ maximum([1 2; 2 4])

for ii in 1:1000
    x -= delta
    x = projection_row(x)
    x = projection_col(x)
    
    x -= delta
    x = projection_col(x)
    x = projection_row(x)
end

x = zeros(n,n)
@time for ii in 1:1000
    x -= (1.0 / ii) .* delta
    x = 0.5 .* (projection_row(x) + projection_col(x)) 
end

x = zeros(n,n)
xnew = -1.0 .* delta
xnew = 0.5 .* (projection_row(xnew) + projection_col(xnew))
ii = 0
@time while sum(abs, x - xnew) > .001
    x = xnew
    ii += 1
    xnew = x - (1.0 / ii) .* delta
    xnew = 0.5 .* (projection_row(xnew) + projection_col(xnew))
end
