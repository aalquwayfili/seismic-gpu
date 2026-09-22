include(joinpath(@__DIR__, "..", "src", "two_gpu.jl"))

function split_main(args)
    kv = options(args, ("n", "steps", "devices", "overlap", "check"))
    devs = Tuple(parse.(Int, split(get(kv, "devices", "0,1"), ",")))
    run_split(; n=parse(Int, get(kv, "n", "4096")), steps=parse(Int, get(kv, "steps", "200")), devs,
          overlap=flag(kv, "overlap"), check=flag(kv, "check"))
end

abspath(PROGRAM_FILE) == (@__FILE__) && split_main(ARGS)
