include(joinpath(@__DIR__, "..", "src", "rtm.jl"))

function rtm_main(args)
    kv = options(args, ("n", "shots", "store", "out", "kernel"))
    n = parse(Int, get(kv, "n", "1024"))
    nshots = parse(Int, get(kv, "shots", "8"))
    store = Symbol(get(kv, "store", "boundary"))
    out = get(kv, "out", "out")
    kernel = Symbol(get(kv, "kernel", "rows"))
    T = Float32
    n >= 128 || throw(ArgumentError("n must be at least 128"))
    nshots > 0 || throw(ArgumentError("shots must be positive"))
    store in (:boundary, :full) || throw(ArgumentError("store must be boundary or full"))
    kernel in (:cols, :rows, :tile, :reg) || throw(ArgumentError("unknown kernel: $kernel"))
    CUDA.functional() || error("CUDA is unavailable; check the NVIDIA driver")
    println("GPU: ", CUDA.name(CUDA.device()))

    vtrue = two_layer(T, n, n)
    vmig = smooth(vtrue, 12)
    S0 = Sim(T, vtrue)
    nt = round(Int, 1.6 / S0.dt)
    nb = S0.nb
    izr = nb + 6
    rx = nb+1:n-nb
    xs = nshots == 1 ? [n ÷ 2] : round.(Int, range(nb + 20, n - nb - 20; length=nshots))
    I = CUDA.zeros(T, n, n)
    total_bytes = 0
    t0 = time()
    err = 0f0
    for (k, isx) in enumerate(xs)
        _, dtrue, _ = forward!(Sim(T, vtrue; isx), kernel, nt, izr, rx)
        bytes, e = migrate_shot!(I, Sim(T, vmig; isx), kernel, nt, izr, rx, dtrue; store, check_at=nt ÷ 2)
        total_bytes = bytes
        err = max(err, e)
        @printf "shot %d/%d at x=%d done\n" k nshots isx
    end
    CUDA.synchronize()
    elapsed = time() - t0
    full_bytes = sizeof(T) * n * n * nt
    @printf "\nstore=%s: %.1f MB saved per shot (storing every snapshot would be %.1f MB, %.0fx more)\n" store total_bytes / 1e6 full_bytes / 1e6 full_bytes / total_bytes
    store == :boundary && @printf "reconstruction error at step %d: %.2e (relative max)\n" nt ÷ 2 err
    @printf "%d shots, %d steps each: %.1f s total\n" nshots nt elapsed
    mkpath(out)
    write(joinpath(out, "image.f32"), Array(I))
    write(joinpath(out, "vmig.f32"), vmig)
    write(joinpath(out, "vtrue.f32"), vtrue)
    open(joinpath(out, "rtm_meta.txt"), "w") do io
        println(io, "nx=$n nz=$n h=$(S0.h) dt=$(S0.dt) nt=$nt nb=$nb shots=$nshots store=$store")
    end
    println("written to $out/")
end

abspath(PROGRAM_FILE) == (@__FILE__) && rtm_main(ARGS)
