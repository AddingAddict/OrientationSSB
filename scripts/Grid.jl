function flatten(A::Array{T,2}) where {T <: Real}
    reduce(vcat,A)
end

function flatten(A::Array{T,3}) where {T <: Real}
    reduce(vcat,A)
end

function unflatten(A::Array{T,1},n::Int64) where {T <: Real}
    reshape(A,(n,n))
end

function unflatten(A::Array{T,1},n::Int64,m::Int64) where {T <: Real}
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