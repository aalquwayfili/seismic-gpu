include(joinpath(@__DIR__, "wave2d.jl"))

function step_rows_range!(un, u, up, r2, gx, gz, c, nx, nzl, z_lo, z_hi)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = z_lo - 1 + (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if H < ix <= nx - H && iz <= z_hi && H < iz <= nzl - H
        @inbounds γ = max(gx[ix], gz[iz])
        @inbounds un[ix, iz] = update(c, u, up, r2, γ, ix, iz, zero(eltype(u)))
    end
    return
end
function launch_range!(un, u, up, r2, gx, gz, c, nx, nzl, z_lo, z_hi)
    nrows = z_hi - z_lo + 1
    nrows <= 0 && return
    @cuda threads=(32, 8) blocks=(cld(nx, 32), cld(nrows, 8)) step_rows_range!(un, u, up, r2, gx, gz, c, nx, nzl, z_lo, z_hi)
end

mutable struct Half{T}
    dev::CuDevice
    nzl::Int
    u::CuMatrix{T}
    up::CuMatrix{T}
    un::CuMatrix{T}
    r2::CuMatrix{T}
    gx::CuVector{T}
    gz::CuVector{T}
    compute::CuStream
    done_edge::CuEvent
end

function Half(T, dev, v, gxh, gzh, r2h, z0, nzh)
    device!(dev)
    nzl = nzh + 2H
    zr = clamp.(z0-H:z0+nzh-1+H, 1, size(v, 2))
    Half{T}(dev, nzl, CUDA.zeros(T, size(v, 1), nzl), CUDA.zeros(T, size(v, 1), nzl), CUDA.zeros(T, size(v, 1), nzl),
            CuArray(r2h[:, zr]), CuArray(gxh), CuArray(gzh[zr]), CuStream(), CuEvent())
end

function exchange!(top::Half, bot::Half)
    copyto!(view(bot.u, :, 1:H), view(top.u, :, top.nzl-2H+1:top.nzl-H))
    copyto!(view(top.u, :, top.nzl-H+1:top.nzl), view(bot.u, :, H+1:2H))
end

function run_split(; n=4096, steps=200, devs=(0, 1), overlap=false, check=false, T=Float32)
    n >= 128 && iseven(n) || throw(ArgumentError("n must be even and at least 128"))
    steps > 0 || throw(ArgumentError("steps must be positive"))
    length(devs) == 2 || throw(ArgumentError("devices must contain two indices"))
    CUDA.functional() || error("CUDA is unavailable; check the NVIDIA driver")
    all(d -> 0 <= d < length(devices()), devs) || throw(ArgumentError("invalid device index"))
    device!(devs[1])
    v = two_layer(T, n, n)
    S = Sim(T, v)
    r2h = Array(S.r2)
    gxh = Array(S.gx)
    gzh = Array(S.gz)
    c = S.c
    nzh = n ÷ 2
    halves = [Half(T, CuDevice(devs[1]), v, gxh, gzh, r2h, 1, nzh), Half(T, CuDevice(devs[2]), v, gxh, gzh, r2h, nzh + 1, nzh)]
    device!(halves[1].dev)
    copy_stream = CuStream()
    CUDA.@allowscalar halves[1].u[n ÷ 2, nzh ÷ 2 + H] = one(T)

    ranges(h) = h === halves[1] ? (lo=2H + 1, hi=h.nzl - H, edge=(h.nzl - 2H + 1, h.nzl - H), inner=(2H + 1, h.nzl - 2H)) :
                                  (lo=H + 1, hi=h.nzl - 2H, edge=(H + 1, 2H), inner=(2H + 1, h.nzl - 2H))
    function step_all!()
        for h in halves
            device!(h.dev)
            r = ranges(h)
            if overlap
                stream!(h.compute) do
                    launch_range!(h.un, h.u, h.up, h.r2, h.gx, h.gz, c, n, h.nzl, r.edge...)
                    record(h.done_edge)
                    launch_range!(h.un, h.u, h.up, h.r2, h.gx, h.gz, c, n, h.nzl, r.inner...)
                end
            else
                stream!(h.compute) do
                    launch_range!(h.un, h.u, h.up, h.r2, h.gx, h.gz, c, n, h.nzl, r.lo, r.hi)
                end
            end
        end
        for h in halves
            device!(h.dev)
            h.up, h.u, h.un = h.u, h.un, h.up
        end
        if overlap
            device!(halves[1].dev)
            # Both edge kernels must finish before either transfer starts.
            for h in halves
                CUDA.wait(h.done_edge, copy_stream)
            end
            stream!(copy_stream) do
                exchange!(halves[1], halves[2])
            end
            synchronize(copy_stream)
            for h in halves
                device!(h.dev)
                synchronize(h.compute)
            end
        else
            for h in halves
                device!(h.dev)
                synchronize(h.compute)
            end
            device!(halves[1].dev)
            exchange!(halves[1], halves[2])
            device!(halves[1].dev)
            synchronize()
        end
    end

    for _ in 1:20
        step_all!()
    end
    for h in halves
        device!(h.dev)
        synchronize()
    end
    t = @elapsed begin
        for _ in 1:steps
            step_all!()
        end
        for h in halves
            device!(h.dev)
            synchronize()
        end
    end
    ms = 1e3 * t / steps
    pts = (n - 2H) * (n - 2H)
    label = "2 GPUs (" * join(devs, ",") * ")" * (overlap ? " overlap" : "")
    @printf "%s: %d×%d  %.3f ms/step  %.2f GPts/s\n" label n n ms pts / (t / steps) / 1e9

    if check
        device!(halves[1].dev)
        F = Fields(S)
        CUDA.@allowscalar F.u[n ÷ 2, nzh ÷ 2] = one(T)
        for _ in 1:(20 + steps)
            launch!(:rows, F.un, F.u, F.up, S.r2, S.gx, S.gz, c, n, n, 0, 0, zero(T))
            F.up, F.u, F.un = F.u, F.un, F.up
        end
        ref = Array(F.u)
        parts = map(halves) do h
            device!(h.dev)
            Array(@view h.u[:, H+1:H+nzh])
        end
        got = hcat(parts...)
        isapprox(got, ref; rtol=1f-5, atol=1f-7) || error("split-grid check failed")
        @printf "check vs one GPU: max|diff| = %.3e  (max|u| = %.3e)\n" maximum(abs, got .- ref) maximum(abs, ref)
    end
end
