include(joinpath(@__DIR__, "..", "src", "wave2d.jl"))

function wave_main(args)
    mode = isempty(args) ? "bench" : args[1]
    mode in ("bench", "demo") || throw(ArgumentError("mode must be bench or demo"))
    kv = options(args[2:end], ("n", "T", "steps", "all", "kernel", "out"))
    n = parse(Int, get(kv, "n", "4096"))
    T = precision(kv)
    steps = parse(Int, get(kv, "steps", "200"))
    n >= 128 || throw(ArgumentError("n must be at least 128"))
    steps > 0 || throw(ArgumentError("steps must be positive"))
    kernel = Symbol(get(kv, "kernel", "rows"))
    kernel in (:cols, :rows, :tile, :reg) || throw(ArgumentError("unknown kernel: $kernel"))
    all_kernels = flag(kv, "all")
    CUDA.functional() || error("CUDA is unavailable; check the NVIDIA driver")
    if mode == "bench"
        println("GPU: ", CUDA.name(CUDA.device()))
        copy_bw = bench_copy(T, n)
        @printf "copy bandwidth (%s, %d×%d): %.0f GB/s\n" T n n copy_bw
        kernels = all_kernels ? (:cols, :rows, :tile, :reg) : (kernel,)
        S = Sim(T, two_layer(T, n, n))
        @printf "%-6s %-8s %8s %9s %9s %7s\n" "kernel" "T" "ms/step" "GPts/s" "GB/s" "%copy"
        for k in kernels
            r = bench(S, k; steps)
            @printf "%-6s %-8s %8.3f %9.2f %9.0f %6.0f%%\n" r.kernel r.T r.ms r.gpts r.gbs 100r.gbs / copy_bw
        end
    elseif mode == "demo"
        demo(; n, out=get(kv, "out", "out"), kernel, T)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && wave_main(ARGS)
