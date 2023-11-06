function flatten(A::Array{T,2}) where {T <: Number}
    reduce(vcat,A)
end

function flatten(A::Array{T,3}) where {T <: Number}
    reduce(vcat,A)
end

function unflatten(A::Array{T,1},n::Int64) where {T <: Number}
    reshape(A,(n,n))
end

function unflatten(A::Array{T,1},n::Int64,m::Int64) where {T <: Number}
    reshape(A,(n,n,m))
end

function kernmat(kernfun::Function,l::Float64,n::Int64)
    km = zeros(n^2,n^2)
    
    dA = (l/n)^2
    pts = [0:n-1;]*l/n
    xs = flatten(pts' .* ones(n))
    ys = flatten(ones(n)' .* pts)
    
    dxs = abs.(xs .- xs')
    dxs[dxs .>  l/2] .= l .- dxs[dxs .>  l/2]
    dys = abs.(ys .- ys')
    dys[dys .>  l/2] .= l .- dys[dys .>  l/2]
    dls = sqrt.(dxs.^2 + dys.^2)
    
    km .= kernfun.(dls)
    km .*= dA
    return km
end

function kernvec(kernfun::Function,l::Float64,n::Int64,pt::Array{Int64,1}=[n÷2,n÷2])
    km = zeros(n^2)
    
    pts = [0:n-1;]*l/n
    xs = flatten(pts' .* ones(n))
    ys = flatten(ones(n)' .* pts)
    
    dxs = abs.(xs .- pt[1]*l/n)
    dxs[dxs .>  l/2] .= l .- dxs[dxs .>  l/2]
    dys = abs.(ys .- pt[2]*l/n)
    dys[dys .>  l/2] .= l .- dys[dys .>  l/2]
    dls = sqrt.(dxs.^2 + dys.^2)
    
    km .= kernfun.(dls)
    return km
end

function wavemat(k::Array{Float64,1},l::Float64,n::Int64)
    pts = [0:n-1;]*l/n
    xs = flatten(pts' .* ones(n))
    ys = flatten(ones(n)' .* pts)
    
    dxs = xs .- xs'
    dys = ys .- ys'
    
    return exp.(1im*(k[1]*dxs+k[2]*dys))
end

function wavevec(k::Array{Float64,1},l::Float64,n::Int64,pt::Array{Int64,1}=[n÷2,n÷2])
    pts = [0:n-1;]*l/n
    xs = flatten((pts .- pt[1]*l/n)' .* ones(n))
    ys = flatten(ones(n)' .* (pts .- pt[1]*l/n))
    xs[xs .>  l/2] .-= l
    ys[ys .>  l/2] .-= l
    
    return exp.(1im*(k[1]*xs+k[2]*ys))
end