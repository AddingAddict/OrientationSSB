using HDF5
using HypergeometricFunctions
using Interpolations
using QuadGK
using SpecialFunctions

sr2 = √(2)
sr2π = √(2π)

abstract type RateParams end

struct RicciardiParams <: RateParams
    θ::Float64
    Vr::Float64
    σn::Float64
    τs::Float64
    τr::Float64
end

struct SSNParams <: RateParams
    k::Float64
    n::Float64
    τs::Float64
end

struct PLParams <: RateParams
    k::Float64
    n::Float64
    τs::Float64
end

function μtox(μ::Float64)::Float64
    sign(μ/100-0.2)*abs(μ/100-0.2)^0.5
end

function xtoμ(x::Float64)::Float64
    100(sign(x)*abs(x)^2.0+0.2)
end

function Φint(rp::RicciardiParams,μ::Float64)
    umax = (rp.θ-μ)/rp.σn
    umin = (rp.Vr-μ)/rp.σn
    if umin > 10
        return umax*exp(-umax^2)/rp.τs
    elseif umin > -4
        return 1/(rp.τr+rp.τs*(0.5*π*(erfi(umax)-erfi(umin)) +
                umax^2*pFq([1.0,1.0],[1.5,2.0],umax^2) -
                umin^2*pFq([1.0,1.0],[1.5,2.0],umin^2)))
    else
        return 1/(rp.τr+rp.τs*(log(abs(umin))-log(abs(umax)) +
                (0.25umin^-2-0.1875umin^-4+0.3125umin^-6-
                    0.8203125umin^-8+2.953125umin^-10) -
                (0.25umax^-2-0.1875umax^-4+0.3125umax^-6-
                    0.8203125umax^-8+2.953125umax^-10)))
    end
end

function Φint(rp::SSNParams,μ::Float64)
    return rp.k*(0.5(μ+abs(μ)))^rp.n
end

function Φint(rp::PLParams,μ::Float64)
    return rp.k*μ^rp.n
end

function Φitp(rps::Vector{T}) where {T<:RateParams}
    xs = range(μtox(-1E3), μtox(5E5), length=2*10^5+1)
    Φint(rps[1],-100.0)
    Φint(rps[1],0.0)
    Φint(rps[1],100.0)
    global Φitps = Dict()
    for rp in rps
        Φs = [Φint(rp,xtoμ(x)) for x in xs]
        Φitps[hash(rp)] = CubicSplineInterpolation(xs, Φs, extrapolation_bc = Line())
    end
end

function Φitp(rps::Vector{T},fid::HDF5.File,write::Bool) where {T<:RateParams}
    if write
        xs = range(μtox(-1E3), μtox(5E5), length=2*10^5+1)
        create_group(fid, "PhItp")
        gp = fid["PhItp"]
        gp["xrange"] = [xs[1],xs[end],length(xs)]
    else
        gp = fid["PhItp"]
        xrange = read(gp,"xrange")
        xs = range(xrange[1],xrange[2],length=round(Int,xrange[3]))
    end
    Φint(rps[1],-100.0)
    Φint(rps[1],0.0)
    Φint(rps[1],100.0)
    global Φitps = Dict()
    idx = 0
    for rp in rps
        idx += 1
        if write
            Φs = [Φint(rp,xtoμ(x)) for x in xs]
            create_group(gp,"rp"*string(idx))
            rpgp = gp["rp"*string(idx)]
            rpgp["Phs"] = Φs
        else
            Φs = read(gp["rp"*string(idx)],"Phs")
        end
        Φitps[hash(rp)] = CubicSplineInterpolation(xs, Φs, extrapolation_bc = Line())
    end
end

function Φ(rp::RateParams,μ::Float64)
    return Φitps[hash(rp)](μtox(μ))
end

function dΦ(rp::RateParams,μ::Float64)
    dμ = 1.0
    return (Φitps[hash(rp)](μtox(μ+dμ))-Φitps[hash(rp)](μtox(μ-dμ)))/(2dμ)
end

function ddΦ(rp::RateParams,μ::Float64)
    dμ = 1.0
    return (Φitps[hash(rp)](μtox(μ+dμ))-2Φitps[hash(rp)](μtox(μ))+
        Φitps[hash(rp)](μtox(μ-dμ)))/(dμ^2)
end

function dddΦ(rp::RateParams,μ::Float64)
    dμ = 1.0
    return (Φitps[hash(rp)](μtox(μ+2dμ))-2Φitps[hash(rp)](μtox(μ+dμ))+
        2Φitps[hash(rp)](μtox(μ-dμ))-Φitps[hash(rp)](μtox(μ-2dμ)))/(2dμ^3)
end
