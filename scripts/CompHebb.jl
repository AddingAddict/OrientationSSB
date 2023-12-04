using Base64
using Distributions
using LinearAlgebra
using PyCall
using PyPlot
using Random
@pyimport matplotlib.animation as anim

include("Rate.jl")

struct NetworkParams{T<:RateParams}
    rng::MersenneTwister
    ne::Int64
    ni::Int64
    nr::Int64
    nf::Int64
    rps::Vector{T}
    rpidx::Vector{Int64}
    pmei::Vector{Int64}
    wmax::Matrix{Float64}
    εrate::Matrix{Float64}
    wr::Matrix{Float64}
    wf::Matrix{Float64}
    u::Vector{Float64}
end

function NetworkParams(ne::Int64,ni::Int64,nf::Int64,rps::Vector{T},wmax::Matrix{Float64},
        εrate::Matrix{Float64},seed::Int64) where {T<:RateParams}
    rng = MersenneTwister(seed)
    nr = ne+ni
    rpidx = ones(Int64,nr)
    rpidx[ne+1:end] .= 2
    pmei = ones(Int64,nr)
    pmei[ne+1:end] .= -1
    wr = abs.(randn(rng,nr,nr))
    wf = abs.(randn(rng,nr,nf))
    eidx = 1:ne
    iidx = ne+1:nr
    wf[eidx,:] .*= wmax[1,1]./sum(wf[eidx,:],dims=2)
    wf[iidx,:] .*= wmax[2,1]./sum(wf[iidx,:],dims=2)
    wr[eidx,eidx] .*= wmax[1,2]./sum(wr[eidx,eidx],dims=2)
    wr[eidx,iidx] .*= wmax[1,3]./sum(wr[eidx,iidx],dims=2)
    wr[iidx,eidx] .*= wmax[2,2]./sum(wr[iidx,eidx],dims=2)
    wr[iidx,iidx] .*= wmax[2,3]./sum(wr[iidx,iidx],dims=2)
    u = zeros(nr)
    NetworkParams(rng,ne,ni,nr,nf,rps,rpidx,pmei,wmax,εrate,wr,wf,u)
end

function normw!(net::NetworkParams)
    eidx = 1:net.ne
    iidx = net.ne+1:net.nr
    net.wf[eidx,:] .*= net.wmax[1,1]./sum(net.wf[eidx,:],dims=2)
    net.wf[iidx,:] .*= net.wmax[2,1]./sum(net.wf[iidx,:],dims=2)
    net.wr[eidx,eidx] .*= net.wmax[1,2]./sum(net.wr[eidx,eidx],dims=2)
    net.wr[eidx,iidx] .*= net.wmax[1,3]./sum(net.wr[eidx,iidx],dims=2)
    net.wr[iidx,eidx] .*= net.wmax[2,2]./sum(net.wr[iidx,eidx],dims=2)
    net.wr[iidx,iidx] .*= net.wmax[2,3]./sum(net.wr[iidx,iidx],dims=2)
end

function dori(ori1::Float64,ori2::Float64)
    Δori = abs(ori1-ori2)
    if Δori > 90.0
        Δori -= 180.0
    elseif Δori < -90.0
        Δori += 180.0
    end
    return Δori
end

function evonet!(net::NetworkParams, nrep::Int, Δt::Float64, ttot::Float64,
        tstm::Float64, af::Float64, σf::Float64,
        freezew::Matrix{Bool}=[false false false; false false false])
    eidx = 1:net.ne
    iidx = net.ne+1:net.nr

    ntot = round(Int,ttot/tstm)
    nstm = round(Int,tstm/Δt)

    τs = zeros(Float64,(net.nr))
    τs[eidx] .= net.rps[1].τs::Float64
    τs[iidx] .= net.rps[2].τs::Float64
    τinvs = 1.0./τs

    εs = zeros(Float64,(net.nr))
    εs[eidx] .= net.rps[1].τs::Float64
    εs[iidx] .= net.rps[2].τs::Float64

    rr = zeros(Float64,(net.nr))
    rf = zeros(Float64,(net.nf))

    fori::Float64 = 0.0
    oris = [0:net.nf-1;]*180.0/net.nf
    Δoris = zeros(Float64,(net.nf))

    du = zeros(Float64,(net.nr))
    dwf = zeros(Float64,(net.nr,net.nf))
    dwr = zeros(Float64,(net.nr,net.nr))

    rrs = zeros(Float64,(nrep+1,net.nr))
    rfs = zeros(Float64,(nrep+1,net.nf))
    wrs = zeros(Float64,(nrep+1,net.nr,net.nr))
    wfs = zeros(Float64,(nrep+1,net.nr,net.nf))
    @views rrs[1,:] = [Φ(net.rps[net.rpidx[i]],net.u[i]) for i in 1:net.nr]
    @views wrs[1,:,:] .= net.wr
    @views wfs[1,:,:] .= net.wf

    for repidx in 1:nrep
        for totidx in 1:ntot
            fori = 180.0rand(net.rng)
            Δoris .= [dori(fori,ori) for ori in oris]
            @fastmath rf .= af*exp.(-Δoris.^2/(2*σf^2))
            for stmidx in 1:nstm
                rr .= [Φint(net.rps[net.rpidx[i]],net.u[i]) for i in 1:net.nr]
                
                # update currents
                du .= -net.u
                BLAS.gemv!('N',1.0,net.wf,rf,1.0,du)
                BLAS.gemv!('N',1.0,net.wr,net.pmei.*rr,1.0,du)
                du .*= Δt*τinvs
                net.u .+= du

                # update weights
                dwf .= 0
                BLAS.ger!(1.0,rr,rf,dwf)
                if freezew[1,1]
                    @views dwf[eidx,:] .= 0
                else
                    @views dwf[eidx,:] .*= Δt*net.εrate[1,1]
                end
                if freezew[2,1]
                    @views dwf[iidx,:] .= 0
                else
                    @views dwf[iidx,:] .*= Δt*net.εrate[2,1]
                end
                net.wf .+= dwf
                
                dwr .= 0
                BLAS.ger!(1.0,rr,rr,dwr)
                if freezew[1,2]
                    @views dwr[eidx,eidx] .= 0
                else
                    @views dwr[eidx,eidx] .*= Δt*net.εrate[1,2]
                end
                if freezew[1,3]
                    @views dwr[eidx,iidx] .= 0
                else
                    @views dwr[eidx,iidx] .*= Δt*net.εrate[1,3]
                end
                if freezew[2,2]
                    @views dwr[iidx,eidx] .= 0
                else
                    @views dwr[iidx,eidx] .*= Δt*net.εrate[2,2]
                end
                if freezew[2,3]
                    @views dwr[iidx,iidx] .= 0
                else
                    @views dwr[iidx,iidx] .*= Δt*net.εrate[2,3]
                end
                net.wr .+= dwr

                normw!(net)
            end
        end
        @views rrs[repidx+1,:] .= rr
        @views rfs[repidx+1,:] .= rf
        @views wrs[repidx+1,:,:] .= net.wr
        @views wfs[repidx+1,:,:] .= net.wf
    end
    return (rrs,rfs,wrs,wfs)
end

function symevonet!(net::NetworkParams, nori::Int, nrep::Int, Δt::Float64, ttot::Float64,
        tstm::Float64, af::Float64, σf::Float64,
        freezew::Matrix{Bool}=[false false false; false false false])
    eidx = 1:net.ne
    iidx = net.ne+1:net.nr

    ntot = round(Int,ttot/tstm)
    nstm = round(Int,tstm/Δt)

    τs = zeros(Float64,(net.nr))
    τs[eidx] .= net.rps[1].τs::Float64
    τs[iidx] .= net.rps[2].τs::Float64
    τinvs = 1.0./τs

    εs = zeros(Float64,(net.nr))
    εs[eidx] .= net.rps[1].τs::Float64
    εs[iidx] .= net.rps[2].τs::Float64

    rr = zeros(Float64,(net.nr))
    rf = zeros(Float64,(net.nf))

    fori::Float64 = 0.0
    oris = [0:net.nf-1;]*180.0/net.nf
    Δoris = zeros(Float64,(net.nf))

    du = zeros(Float64,(net.nr))
    dwf = zeros(Float64,(net.nr,net.nf))
    dwr = zeros(Float64,(net.nr,net.nr))

    rrs = zeros(Float64,(nrep+1,net.nr))
    rfs = zeros(Float64,(nrep+1,net.nf))
    wrs = zeros(Float64,(nrep+1,net.nr,net.nr))
    wfs = zeros(Float64,(nrep+1,net.nr,net.nf))
    @views rrs[1,:] = [Φ(net.rps[net.rpidx[i]],net.u[i]) for i in 1:net.nr]
    @views wrs[1,:,:] .= net.wr
    @views wfs[1,:,:] .= net.wf

    for repidx in 1:nrep
        for totidx in 1:ntot
            dwf .= 0
            dwr .= 0
            for oriidx in 1:nori
                fori = 180.0*(oriidx-1)/nori
                Δoris .= [dori(fori,ori) for ori in oris]
                @fastmath rf .= af*exp.(-Δoris.^2/(2*σf^2))
                for stmidx in 1:nstm
                    rr .= [Φint(net.rps[net.rpidx[i]],net.u[i]) for i in 1:net.nr]
                    
                    # update currents
                    du .= -net.u
                    BLAS.gemv!('N',1.0,net.wf,rf,1.0,du)
                    BLAS.gemv!('N',1.0,net.wr,net.pmei.*rr,1.0,du)
                    du .*= Δt*τinvs
                    net.u .+= du
                end

                # update weights
                BLAS.ger!(1.0,rr,rf,dwf)
                BLAS.ger!(1.0,rr,rr,dwr)
            end
            dwf ./= nori
            dwr ./= nori

            if freezew[1,1]
                @views dwf[eidx,:] .= 0
            else
                @views dwf[eidx,:] .*= Δt*nstm*net.εrate[1,1]
            end
            if freezew[2,1]
                @views dwf[iidx,:] .= 0
            else
                @views dwf[iidx,:] .*= Δt*nstm*net.εrate[2,1]
            end
            net.wf .+= dwf
            
            if freezew[1,2]
                @views dwr[eidx,eidx] .= 0
            else
                @views dwr[eidx,eidx] .*= Δt*nstm*net.εrate[1,2]
            end
            if freezew[1,3]
                @views dwr[eidx,iidx] .= 0
            else
                @views dwr[eidx,iidx] .*= Δt*nstm*net.εrate[1,3]
            end
            if freezew[2,2]
                @views dwr[iidx,eidx] .= 0
            else
                @views dwr[iidx,eidx] .*= Δt*nstm*net.εrate[2,2]
            end
            if freezew[2,3]
                @views dwr[iidx,iidx] .= 0
            else
                @views dwr[iidx,iidx] .*= Δt*nstm*net.εrate[2,3]
            end
            net.wr .+= dwr

            normw!(net)
        end
        @views rrs[repidx+1,:] .= rr
        @views rfs[repidx+1,:] .= rf
        @views wrs[repidx+1,:,:] .= net.wr
        @views wfs[repidx+1,:,:] .= net.wf
    end
    return (rrs,rfs,wrs,wfs)
end

function sortcells(rrs::Array{Float64,2},wrs::Array{Float64,3},wfs::Array{Float64,3})::Array{Float64,1}
    θs = [0:net.nf-1;]*2π/net.nf
    poris = angle.(mean(wfs[end,:,:].*exp.(-im*θs)',dims=2))[:]
    poris[poris .< 0.0] .+= 2π
    poris *= 180.0/(2π)
    eidx = 1:net.ne
    iidx = net.ne+1:net.nr
    sorte = sortperm(poris[eidx])
    sorti = sortperm(poris[iidx])
    rrsort = deepcopy(rrs)
    wrsort = deepcopy(wrs)
    wfsort = deepcopy(wfs)
    rrsort[:,eidx] .= rrsort[:,eidx][:,sorte]
    rrsort[:,iidx] .= rrsort[:,iidx][:,sorti]
    wrsort[:,:,eidx] .= wrsort[:,:,eidx][:,:,sorte]
    wrsort[:,:,iidx] .= wrsort[:,:,iidx][:,:,sorti]
    wrsort[:,eidx,:] .= wrsort[:,eidx,:][:,sorte,:]
    wrsort[:,iidx,:] .= wrsort[:,iidx,:][:,sorti,:]
    wfsort[:,eidx,:] .= wfsort[:,eidx,:][:,sorte,:]
    wfsort[:,iidx,:] .= wfsort[:,iidx,:][:,sorti,:]
    rrs .= rrsort
    wrs .= wrsort
    wfs .= wfsort
    poris[eidx] = poris[eidx][sorte]
    poris[iidx] = poris[iidx][sorti]
    return poris
end

function plotanim(net::NetworkParams,rrs::Array{Float64,2},rfs::Array{Float64,2},wrs::Array{Float64,3},
        wfs::Array{Float64,3},poris::Array{Float64,1},name::String)
    fig, axs = subplots(nrows=2,ncols=2,figsize=(10,8),gridspec_kw=Dict("width_ratios"=>(3,2)))
    axs[1,1].set_xlim(0.0,180.0)
    axs[1,1].set_ylim(0.0,1.1*maximum(wfs))
    axs[1,2].set_xlim(-0.5,net.nr-0.5)
    axs[1,2].set_ylim(net.nr-0.5,-0.5)
    axs[2,1].set_xlim(0.0,180.0)
    axs[2,1].set_ylim(0.0,1.1*maximum(rfs))
    axs[2,2].set_axis_off()

    oris = [0:net.nf-1;]*180.0/net.nf
    θs = [0:net.nf-1;]*2π/net.nf
    oris_trans = zeros(net.nf)
    poris_trans = zeros(net.nr)

    wlines = Vector{Any}(undef, net.nr)
    points = Vector{Any}(undef, net.nr)
    for i in 1:net.ne
        wlines[i] = axs[1,1][:plot]([],[],"r-")[1]
        points[i] = axs[2,1][:scatter]([],[],color="r")
    end
    for i in 1:net.ni
        wlines[net.ne+i] = axs[1,1][:plot]([],[],"b-")[1]
        points[net.ne+i] = axs[2,1][:scatter]([],[],color="b")
    end
    rline = axs[2,1][:plot]([],[],"k--")[1]
    wimsh = axs[1,2][:imshow](zeros(net.nr,net.nr),vmin=0,vmax=1.1*maximum(wrs))

    function init()
        for i in 1:net.nr
            wlines[i][:set_data]([],[])
            points[i][:set_offsets]([[-1,-1]])
        end
        rline[:set_data]([],[])
        wimsh[:set_data](zeros(net.nr,net.nr))
        return (wlines...,points...,wimsh,Union{})
    end

    function animate(t)
        foris = angle(mean(rfs[t+1,:].*exp.(im*θs)))
        if foris < 0.0
            foris += 2π
        end
        foris *= 180.0/(2π)
        poris_trans .= poris .- foris .+ 90.0
        poris_trans[poris_trans .> 180.0] .-= 180.0
        poris_trans[poris_trans .< 0.0] .+= 180.0
        oris_trans .= oris .- foris .+ 90.0
        oris_trans[oris_trans .> 180.0] .-= 180.0
        oris_trans[oris_trans .< 0.0] .+= 180.0
        for i in 1:net.nr
            wlines[i][:set_data](oris,wfs[t+1,i,:])
            points[i][:set_offsets]([[poris_trans[i],rrs[t+1,i]]])
        end
        rline[:set_data](oris_trans,rfs[t+1,:])
        wimsh[:set_data](wrs[t+1,:,:])
        return (wlines...,points...,wimsh,Union{})
    end

    myanim = anim.FuncAnimation(fig, animate, init_func=init, frames=size(wfs)[1], interval=20)
    myanim[:save](name*".mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])

    function html_video(filename)
        open(filename) do f
            base64_video = base64encode(f)
            """<video controls src="data:video/x-m4v;base64,$base64_video">"""
        end
    end

    display("text/html", html_video(name*".mp4"))
end

function genrs(net::NetworkParams, nori::Int, Δt::Float64, tstm::Float64, af::Float64, σf::Float64)
    eidx = 1:net.ne
    iidx = net.ne+1:net.nr

    nstm = round(Int,tstm/Δt)

    τs = zeros(Float64,(net.nr))
    τs[eidx] .= net.rps[1].τs::Float64
    τs[iidx] .= net.rps[2].τs::Float64
    τinvs = 1.0./τs

    rr = zeros(Float64,(net.nr))
    rf = zeros(Float64,(net.nf))

    fori::Float64 = 0.0
    oris = [0:net.nf-1;]*180.0/net.nf
    Δoris = zeros(Float64,(net.nf))

    du = zeros(Float64,(net.nr))

    rrs = zeros(Float64,(nori,net.nr))
    rfs = zeros(Float64,(nori,net.nf))

    for oriidx in 1:nori
        fori = 180.0*(oriidx-1)/nori
        Δoris .= [dori(fori,ori) for ori in oris]
        @fastmath rf .= af*exp.(-Δoris.^2/(2*σf^2))
        for stmidx in 1:nstm
            rr .= [Φint(net.rps[net.rpidx[i]],net.u[i]) for i in 1:net.nr]
            
            # update currents
            du .= -net.u
            BLAS.gemv!('N',1.0,net.wf,rf,1.0,du)
            BLAS.gemv!('N',1.0,net.wr,net.pmei.*rr,1.0,du)
            du .*= Δt*τinvs
            net.u .+= du
        end

        @views rrs[oriidx,:] .= rr
        @views rfs[oriidx,:] .= rf
    end

    return (rrs,rfs)
end

function symave(A::Matrix{Float64})
    n1,n2 = size(A)
    A_shift = deepcopy(A)
    A .= 0
    
    if n2 >= n1
        for i in 1:n1
            A .+= A_shift
            A_shift = circshift(A_shift,(1,n2÷n1))
        end
        A ./= n1
    else
        for i in 1:n2
            A .+= A_shift
            A_shift = circshift(A_shift,(n1÷n2,1))
        end
        A ./= n2
    end

    maxidx1 = argmax(A[:,1])
    maxidx2 = argmax(A[1,:])
    A_shift = circshift(circshift(A,(-(maxidx1-1),-(maxidx2-1)))[end:-1:1,end:-1:1],(maxidx1,maxidx2))
    A .= 0.5*(A+A_shift)

    return A
end