include(joinpath(@__DIR__, "..", "src", "rtm.jl"))
using Printf

const OUT = get(ENV, "FIGS", "figs")
w(name, a) = write(joinpath(OUT, name * ".f32"), Float32.(a))
meta(name; kw...) = open(io -> foreach(((k, v),) -> println(io, "$k=$v"), kw), joinpath(OUT, name * ".meta"), "w")

function run!(S::Sim{T}, nt, f; kernel=:rows) where T
    F = Fields(S)
    for it in 1:nt
        step!(S, F, kernel, it * S.dt)
        f(it, F)
    end
    F
end

function figures_main()
    CUDA.functional() || error("CUDA is unavailable; check the NVIDIA driver")
    mkpath(OUT)
    T = Float32

    let n = 768, v = two_layer(T, n, n), S = Sim(T, v), times = (0.15, 0.35, 0.55, 0.75, 0.95, 1.15)
        steps = round.(Int, times ./ S.dt)
        run!(S, maximum(steps), (it, F) -> (k = findfirst(==(it), steps); k === nothing || w("timelapse_$k", Array(F.u))))
        w("timelapse_v", v)
        meta("timelapse"; n, h=S.h, nb=S.nb, times=join(times, ","))
        println("1 timelapse")
    end

    let n = 512, v = fill(T(2000), n, n), t = 0.9
        for (name, γ) in (("edges_off", 0.0), ("edges_on", 0.08))
            S = Sim(T, v; γmax=γ, isz=n ÷ 2)
            F = run!(S, round(Int, t / S.dt), (it, F) -> nothing)
            w(name, Array(F.u))
        end
        meta("edges"; n, h=4.0, nb=40, t)
        println("2 edges")
    end

    let n = 600, v = fill(T(2000), n, n), h = 10.0, f0 = 20.0, t = 1.0
        C2 = (-2.0, 1.0, 0.0, 0.0, 0.0)
        dt = 0.9 * cfl_coeff(2) * h / 2000
        for (name, c) in (("disp_2nd", C2), ("disp_8th", C8))
            S = Sim(T, v; h, f0, c, dt, isz=n ÷ 2)
            F = run!(S, round(Int, t / S.dt), (it, F) -> nothing)
            w(name, Array(F.u))
        end
        meta("disp"; n, h, f0, t, ppw=round(2000 / (2.5f0 * f0) / h, digits=1))
        println("3 dispersion")
    end

    let n = 256, v = fill(T(2000), n, n), h = 4.0, nt = 700
        dtc = cfl_coeff(2) * h / 2000
        for (name, f) in (("cfl_095", 0.95), ("cfl_105", 1.05))
            S = Sim(T, v; dt=f * dtc, isz=n ÷ 2)
            amp = Float32[]
            run!(S, nt, (it, F) -> push!(amp, min(maximum(abs, F.u), 1f30)))
            w(name, amp)
        end
        meta("cfl"; nt, dtc)
        println("4 cfl")
    end

    let n = 512, nshots = 16
        vtrue = two_layer(T, n, n)
        vmig = smooth(vtrue, 12)
        S0 = Sim(T, vtrue)
        nt = round(Int, 1.6 / S0.dt)
        nb = S0.nb
        izr = nb + 6
        rx = nb+1:n-nb
        xs = round.(Int, range(nb + 20, n - nb - 20; length=nshots))
        order = [1, 16, 8, 4, 12, 2, 6, 10, 14, 3, 5, 7, 9, 11, 13, 15]
        I = CUDA.zeros(T, n, n)
        for (k, j) in enumerate(order)
            isx = xs[j]
            _, dtrue, _ = forward!(Sim(T, vtrue; isx), :rows, nt, izr, rx)
            cap = k == 1 ? Dict{Symbol,Any}() : nothing
            migrate_shot!(I, Sim(T, vmig; isx), :rows, nt, izr, rx, dtrue;
                          store=:boundary, capture=(k == 1 ? round(Int, 0.62 / S0.dt) : 0), captured=cap)
            if k == 1
                w("ic_S", cap[:S])
                w("ic_R", cap[:R])
                w("ic_I", cap[:I])
                w("gather_obs", Array(dtrue))
                w("gather_refl", cap[:residual])
            end
            k in (1, 2, 4, 8, 16) && w("rtm_$k", Array(I))
        end
        w("rtm_vtrue", vtrue)
        w("rtm_vmig", vmig)
        meta("rtm"; n, h=S0.h, nb, nt, dt=S0.dt, shot1=xs[order[1]], tcap=0.62, nrec=length(rx))
        println("5+6 rtm")
    end
    println("figures data in $OUT/")
end

abspath(PROGRAM_FILE) == (@__FILE__) && figures_main()
