using Plots, Statistics

readmeta(path) = Dict(split(word, '='; limit=2) for word in split(read(path, String)))

function read_f32(path, dims)
    filesize(path) == sizeof(Float32) * prod(dims) || error("unexpected array size: $path")
    read!(path, Array{Float32}(undef, dims))
end

function laplacian(a)
    b = zeros(eltype(a), size(a))
    for j in 2:size(a, 2)-1, i in 2:size(a, 1)-1
        b[i, j] = a[i-1, j] + a[i+1, j] + a[i, j-1] + a[i, j+1] - 4a[i, j]
    end
    b
end

function waveplot(x, z, a; title, clip=0.2, limit=nothing)
    lim = isnothing(limit) ? clip * maximum(abs, a) : limit
    lim = max(lim, eps(Float32))
    heatmap(x, z, permutedims(a); title, color=:grays, clims=(-lim, lim),
            yflip=true, xlabel="x (km)", ylabel="depth (km)", colorbar=false)
end

function plot_demo(dir)
    m = readmeta(joinpath(dir, "meta.txt"))
    nx, nz, nt, nrec, nb = (parse(Int, m[k]) for k in ("nx", "nz", "nt", "nrec", "nb"))
    h, dt = parse(Float64, m["h"]), parse(Float64, m["dt"])
    x, z = (0:nx-1) .* h ./ 1000, (0:nz-1) .* h ./ 1000
    v = read_f32(joinpath(dir, "velocity.f32"), (nx, nz))
    u = read_f32(joinpath(dir, "snapshot.f32"), (nx, nz))
    d = read_f32(joinpath(dir, "gather.f32"), (nrec, nt))
    velocity = heatmap(x, z, permutedims(v); title="Velocity (m/s)", color=:viridis,
                       yflip=true, xlabel="x (km)", ylabel="depth (km)")
    t = round(parse(Int, m["snap_step"]) * dt; digits=2)
    snapshot = waveplot(x, z, u; title="Pressure at $t s")
    contour!(snapshot, x, z, permutedims(v); levels=[2000], color=:seagreen,
             linewidth=1, colorbar=false)
    savefig(plot(velocity, snapshot; layout=(1, 2), size=(1100, 440)), joinpath(dir, "snapshot.png"))
    receivers = (nb:nb+nrec-1) .* h ./ 1000
    gather = waveplot(receivers, (1:nt) .* dt, d; title="Shot gather", clip=0.05)
    ylabel!(gather, "time (s)")
    savefig(gather, joinpath(dir, "gather.png"))
end

function plot_rtm(dir)
    m = readmeta(joinpath(dir, "rtm_meta.txt"))
    nx, nz, nb = (parse(Int, m[k]) for k in ("nx", "nz", "nb"))
    h = parse(Float64, m["h"])
    a = read_f32(joinpath(dir, "image.f32"), (nx, nz))
    v = read_f32(joinpath(dir, "vtrue.f32"), (nx, nz))
    xr, zr = nb+1:nx-nb, nb+1:nz-nb
    x, z = (xr .- 1) .* h ./ 1000, (zr .- 1) .* h ./ 1000
    panels = []
    for (data, title) in ((a, "RTM"), (laplacian(a), "RTM, Laplacian filter"))
        crop = data[xr, zr]
        panel = waveplot(x, z, crop; title, limit=3std(crop))
        contour!(panel, x, z, permutedims(v[xr, zr]); levels=[2000], color=:seagreen,
                 linestyle=:dash, linewidth=1, colorbar=false)
        push!(panels, panel)
    end
    savefig(plot(panels...; layout=(1, 2), size=(1100, 440)), joinpath(dir, "rtm.png"))
end

function plot_experiments(dir)
    m = readmeta(joinpath(dir, "timelapse.meta"))
    n, h = parse(Int, m["n"]), parse(Float64, m["h"])
    x = (0:n-1) .* h ./ 1000
    times = split(m["times"], ',')
    panels = [waveplot(x, x, read_f32(joinpath(dir, "timelapse_$k.f32"), (n, n));
                       title="$(times[k]) s", clip=0.12) for k in eachindex(times)]
    savefig(plot(panels...; layout=(2, 3), size=(1200, 760)), joinpath(dir, "timelapse.png"))

    m = readmeta(joinpath(dir, "edges.meta"))
    n, h = parse(Int, m["n"]), parse(Float64, m["h"])
    x = (0:n-1) .* h ./ 1000
    panels = [waveplot(x, x, read_f32(joinpath(dir, "edges_$name.f32"), (n, n));
                       title="Damping $name", clip=0.25) for name in ("off", "on")]
    savefig(plot(panels...; layout=(1, 2), size=(1000, 440)), joinpath(dir, "edges.png"))

    m = readmeta(joinpath(dir, "disp.meta"))
    n, h = parse(Int, m["n"]), parse(Float64, m["h"])
    x = (0:n-1) .* h ./ 1000
    u2 = read_f32(joinpath(dir, "disp_2nd.f32"), (n, n))
    u8 = read_f32(joinpath(dir, "disp_8th.f32"), (n, n))
    limit = 0.35maximum(abs, u8)
    panels = [waveplot(x, x, a; title, limit) for (a, title) in
              ((u2, "Second order"), (u8, "Eighth order"))]
    savefig(plot(panels...; layout=(1, 2), size=(1000, 440)), joinpath(dir, "dispersion.png"))

    m = readmeta(joinpath(dir, "cfl.meta"))
    nt = parse(Int, m["nt"])
    stability = plot(; yscale=:log10, xlabel="Time step", ylabel="Maximum pressure")
    for (name, label) in (("095", "95% of CFL limit"), ("105", "105% of CFL limit"))
        a = read_f32(joinpath(dir, "cfl_$name.f32"), (nt,))
        plot!(stability, 1:nt, clamp.(a, 1f-12, 1f30); label)
    end
    savefig(stability, joinpath(dir, "cfl.png"))

    m = readmeta(joinpath(dir, "rtm.meta"))
    n, nb = parse(Int, m["n"]), parse(Int, m["nb"])
    h, dt = parse(Float64, m["h"]), parse(Float64, m["dt"])
    nt, nrec = parse(Int, m["nt"]), parse(Int, m["nrec"])
    interior = nb+1:n-nb
    x = (interior .- 1) .* h ./ 1000
    panels = []
    for shots in (1, 2, 4, 8, 16)
        a = laplacian(read_f32(joinpath(dir, "rtm_$shots.f32"), (n, n)))[interior, interior]
        push!(panels, waveplot(x, x, a; title="$shots shots", limit=3std(a)))
    end
    savefig(plot(panels...; layout=(1, 5), size=(1500, 350)), joinpath(dir, "rtm_stack.png"))
    panels = [waveplot(x, x, read_f32(joinpath(dir, "ic_$name.f32"), (n, n))[interior, interior];
                       title, clip=0.3) for (name, title) in
              (("S", "Source"), ("R", "Receiver"), ("I", "Image"))]
    savefig(plot(panels...; layout=(1, 3), size=(1200, 440)), joinpath(dir, "imaging_condition.png"))
    d = read_f32(joinpath(dir, "gather_obs.f32"), (nrec, nt))
    gather = waveplot((nb:nb+nrec-1) .* h ./ 1000, (1:nt) .* dt, d; title="Shot gather", clip=0.05)
    ylabel!(gather, "time (s)")
    savefig(gather, joinpath(dir, "gather.png"))
end

function plot_main(args)
    length(args) <= 1 || throw(ArgumentError("usage: plot.jl [output directory]"))
    dir = isempty(args) ? "out" : only(args)
    demo = isfile(joinpath(dir, "meta.txt"))
    rtm = isfile(joinpath(dir, "rtm_meta.txt"))
    experiments = isfile(joinpath(dir, "timelapse.meta"))
    demo || rtm || experiments || error("no simulation output in $dir")
    demo && plot_demo(dir)
    rtm && plot_rtm(dir)
    experiments && plot_experiments(dir)
    println("Plots written to $dir/")
end

abspath(PROGRAM_FILE) == (@__FILE__) && plot_main(ARGS)
