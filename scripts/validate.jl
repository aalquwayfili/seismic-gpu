using Dates

root = dirname(@__DIR__)
out = joinpath(root, "out", "validation-" * Dates.format(now(UTC), "yyyymmddTHHMMSS"))
mkpath(out)

function record(name, args)
    println("Running ", name)
    open(joinpath(out, name * ".txt"), "w") do io
        run(pipeline(`$(Base.julia_cmd()) --project=$root $args`; stdout=io, stderr=io))
    end
end

cd(root) do
    open(joinpath(out, "revision.txt"), "w") do io
        println(io, "UTC: ", now(UTC))
        run(pipeline(`git rev-parse HEAD`; stdout=io))
        run(pipeline(`git status --short`; stdout=io))
    end
    cp(joinpath(root, "Manifest.toml"), joinpath(out, "Manifest.toml"))
    record("environment", ["-e", "using InteractiveUtils, Pkg, CUDA; versioninfo(); Pkg.status(); CUDA.versioninfo()"])
    record("gpu-tests", ["test/gpu.jl"])
    for trial in 1:3
        record("float32-$trial", ["scripts/wave2d.jl", "bench", "n=4096", "steps=300", "all=1"])
        record("float64-$trial", ["scripts/wave2d.jl", "bench", "n=4096", "steps=300", "T=Float64", "kernel=rows"])
    end
end

println("Results: ", out)
