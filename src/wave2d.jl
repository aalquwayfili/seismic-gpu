using CUDA, Printf

include(joinpath(@__DIR__, "numerics.jl"))

@inline function stencil(c, u, ix, iz)
    @inbounds begin
        lap = 2 * c[1] * u[ix, iz]
        lap += c[2] * (u[ix-1, iz] + u[ix+1, iz] + u[ix, iz-1] + u[ix, iz+1])
        lap += c[3] * (u[ix-2, iz] + u[ix+2, iz] + u[ix, iz-2] + u[ix, iz+2])
        lap += c[4] * (u[ix-3, iz] + u[ix+3, iz] + u[ix, iz-3] + u[ix, iz+3])
        lap += c[5] * (u[ix-4, iz] + u[ix+4, iz] + u[ix, iz-4] + u[ix, iz+4])
        return lap
    end
end

@inline function update(c, u, up, r2, γ, ix, iz, s)
    lap = stencil(c, u, ix, iz)
    @inbounds return (2 - γ) * u[ix, iz] - (1 - γ) * up[ix, iz] + r2[ix, iz] * lap + s
end

# Consecutive threads walk along z, which is strided in Julia arrays.
function step_cols!(un, u, up, r2, gx, gz, c, nx, nz, isx, isz, s)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= nx * nz
        iz = (i - 1) % nz + 1
        ix = (i - 1) ÷ nz + 1
        if H < ix <= nx - H && H < iz <= nz - H
            @inbounds γ = max(gx[ix], gz[iz])
            src = (ix == isx && iz == isz) ? s : zero(s)
            @inbounds un[ix, iz] = update(c, u, up, r2, γ, ix, iz, src)
        end
    end
    return
end

# Consecutive threads walk along x, the contiguous dimension.
function step_rows!(un, u, up, r2, gx, gz, c, nx, nz, isx, isz, s)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if H < ix <= nx - H && H < iz <= nz - H
        @inbounds γ = max(gx[ix], gz[iz])
        src = (ix == isx && iz == isz) ? s : zero(s)
        @inbounds un[ix, iz] = update(c, u, up, r2, γ, ix, iz, src)
    end
    return
end

const TX, TZ = 32, 8
function step_tile!(un, u, up, r2, gx, gz, c, nx, nz, isx, isz, s)
    T = eltype(u)
    t = CuStaticSharedArray(T, (TX + 2H, TZ + 2H))
    tx, tz = threadIdx().x, threadIdx().y
    ix = (blockIdx().x - 1) * TX + tx
    iz = (blockIdx().y - 1) * TZ + tz
    cx, cz = clamp(ix, 1, nx), clamp(iz, 1, nz)
    @inbounds begin
        t[tx+H, tz+H] = u[cx, cz]
        if tx <= H
            t[tx, tz+H]      = u[clamp(ix - H, 1, nx), cz]
            t[tx+TX+H, tz+H] = u[clamp(ix + TX, 1, nx), cz]
        end
        if tz <= H
            t[tx+H, tz]      = u[cx, clamp(iz - H, 1, nz)]
            t[tx+H, tz+TZ+H] = u[cx, clamp(iz + TZ, 1, nz)]
        end
    end
    # Every thread must reach the barrier, including those outside the live grid.
    sync_threads()
    if H < ix <= nx - H && H < iz <= nz - H
        @inbounds begin
            sx, sz = tx + H, tz + H
            lap = stencil(c, t, sx, sz)
            γ = max(gx[ix], gz[iz])
            src = (ix == isx && iz == isz) ? s : zero(s)
            un[ix, iz] = (2 - γ) * t[sx, sz] - (1 - γ) * up[ix, iz] + r2[ix, iz] * lap + src
        end
    end
    return
end

const NZ = 8
function step_reg!(un, u, up, r2, gx, gz, c, nx, nz, isx, isz, s)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz0 = H + 1 + ((blockIdx().y - 1) * blockDim().y + threadIdx().y - 1) * NZ
    (ix <= H || ix > nx - H || iz0 > nz - H) && return
    @inbounds begin
        γx = gx[ix]
        w1 = u[ix, iz0-4]
        w2 = u[ix, iz0-3]
        w3 = u[ix, iz0-2]
        w4 = u[ix, iz0-1]
        w5 = u[ix, iz0]
        w6 = u[ix, iz0+1]
        w7 = u[ix, iz0+2]
        w8 = u[ix, iz0+3]
        w9 = u[ix, min(iz0 + 4, nz)]
        for j in 0:NZ-1
            iz = iz0 + j
            iz > nz - H && break
            lap = 2 * c[1] * w5
            lap += c[2] * (u[ix-1, iz] + u[ix+1, iz] + w4 + w6)
            lap += c[3] * (u[ix-2, iz] + u[ix+2, iz] + w3 + w7)
            lap += c[4] * (u[ix-3, iz] + u[ix+3, iz] + w2 + w8)
            lap += c[5] * (u[ix-4, iz] + u[ix+4, iz] + w1 + w9)
            γ = max(γx, gz[iz])
            src = (ix == isx && iz == isz) ? s : zero(s)
            un[ix, iz] = (2 - γ) * w5 - (1 - γ) * up[ix, iz] + r2[ix, iz] * lap + src
            w1 = w2
            w2 = w3
            w3 = w4
            w4 = w5
            w5 = w6
            w6 = w7
            w7 = w8
            w8 = w9
            w9 = u[ix, min(iz + 5, nz)]
        end
    end
    return
end

function launch!(kernel, un, u, up, r2, gx, gz, c, nx, nz, isx, isz, s)
    if kernel === :cols
        n = nx * nz
        @cuda threads=256 blocks=cld(n, 256) step_cols!(un, u, up, r2, gx, gz, c, nx, nz, isx, isz, s)
    elseif kernel === :rows
        @cuda threads=(32, 8) blocks=(cld(nx, 32), cld(nz, 8)) step_rows!(un, u, up, r2, gx, gz, c, nx, nz, isx, isz, s)
    elseif kernel === :tile
        @cuda threads=(TX, TZ) blocks=(cld(nx, TX), cld(nz, TZ)) step_tile!(un, u, up, r2, gx, gz, c, nx, nz, isx, isz, s)
    elseif kernel === :reg
        @cuda threads=(32, 4) blocks=(cld(nx, 32), cld(nz - 2H, 4NZ)) step_reg!(un, u, up, r2, gx, gz, c, nx, nz, isx, isz, s)
    else
        error("unknown kernel $kernel")
    end
end

struct Sim{T}
    nx::Int
    nz::Int
    h::T
    dt::T
    f0::T
    nb::Int
    r2::CuMatrix{T}
    gx::CuVector{T}
    gz::CuVector{T}
    c::NTuple{5,T}
    isx::Int
    isz::Int
end

function Sim(T, v::Matrix; h=4.0, f0=15.0, nb=40, γmax=0.08, dt=nothing, isx=size(v, 1) ÷ 2, c=C8, isz=nb + 6)
    nx, nz = size(v)
    vmax = maximum(v)
    dt === nothing && (dt = 0.9 * cfl_coeff(2) * h / vmax)
    r2 = CuArray(T.((v .* dt ./ h) .^ 2))
    Sim{T}(nx, nz, T(h), T(dt), T(f0), nb, r2,
           CuArray(T.(damping(nx, nb, γmax))), CuArray(T.(damping(nz, nb, γmax))), T.(c),
           isx, isz)
end

mutable struct Fields{T}
    u::CuMatrix{T}
    up::CuMatrix{T}
    un::CuMatrix{T}
end

Fields(S::Sim{T}) where T = Fields(ntuple(_ -> CUDA.zeros(T, S.nx, S.nz), 3)...)

function step!(S::Sim, F::Fields, kernel, t; src=true)
    s = src ? S.dt^2 * ricker(t, S.f0) * 1f4 : 0.0
    launch!(kernel, F.un, F.u, F.up, S.r2, S.gx, S.gz, S.c, S.nx, S.nz, S.isx, S.isz, eltype(F.u)(s))
    F.up, F.u, F.un = F.u, F.un, F.up
    return F
end

function bench(S::Sim{T}, kernel; steps=200, warm=20) where T
    F = Fields(S)
    for i in 1:warm
        step!(S, F, kernel, i * S.dt)
    end
    CUDA.synchronize()
    t = CUDA.@elapsed for i in 1:steps
        step!(S, F, kernel, (warm + i) * S.dt)
    end
    pts = (S.nx - 2H) * (S.nz - 2H)
    ms = 1e3 * t / steps
    gpts = pts / (t / steps) / 1e9
    # Estimated traffic: three arrays read, one written; neighbour reuse comes from cache.
    gbs = 4 * sizeof(T) * pts / (t / steps) / 1e9
    (; kernel, T, n=S.nx, ms, gpts, gbs)
end

function bench_copy(T, n; reps=50)
    a = CUDA.rand(T, n, n)
    b = similar(a)
    copyto!(b, a)
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:reps
        copyto!(b, a)
    end
    2 * sizeof(T) * n * n / (t / reps) / 1e9
end

function demo(; n=1024, out="out", kernel=:rows, T=Float32, snap_at=0.55)
    v = two_layer(T, n, n)
    S = Sim(T, v)
    F = Fields(S)
    nt = round(Int, 1.6 / S.dt)
    1 <= round(Int, snap_at / S.dt) <= nt || throw(ArgumentError("snapshot time is outside the run"))
    izr = S.nb + 6
    rec = zeros(T, n - 2S.nb, nt)
    snap = nothing
    for it in 1:nt
        step!(S, F, kernel, it * S.dt)
        rec[:, it] = Array(@view F.u[S.nb+1:n-S.nb, izr])
        it == round(Int, snap_at / S.dt) && (snap = Array(F.u))
    end
    mkpath(out)
    write_f32(joinpath(out, "snapshot.f32"), snap)
    write_f32(joinpath(out, "gather.f32"), rec)
    write_f32(joinpath(out, "velocity.f32"), v)
    open(joinpath(out, "meta.txt"), "w") do io
        println(io, "nx=$n nz=$n h=$(S.h) dt=$(S.dt) nt=$nt f0=$(S.f0) nb=$(S.nb) nrec=$(size(rec,1)) snap_step=$(round(Int, snap_at / S.dt))")
    end
    @printf "demo: %d×%d, dt=%.4f ms, %d steps, written to %s/\n" n n 1e3 * S.dt nt out
end
