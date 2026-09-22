ENV["GKSwstype"] = "100"
using Test
include(joinpath(@__DIR__, "..", "src", "numerics.jl"))
include(joinpath(@__DIR__, "..", "scripts", "plot.jl"))

@testset "Plot output" begin
    mktempdir() do dir
        n, nt, nb = 32, 16, 4
        model = two_layer(Float32, n, n)
        wave = Float32[sin(i / 3) * cos(j / 4) for i in 1:n, j in 1:n]
        for (name, a) in (("velocity", model), ("vtrue", model), ("snapshot", wave),
                          ("image", wave), ("gather", wave[nb+1:n-nb, 1:nt]))
            write_f32(joinpath(dir, "$name.f32"), a)
        end
        write(joinpath(dir, "meta.txt"), "nx=$n nz=$n nt=$nt nrec=$(n-2nb) nb=$nb h=4 dt=0.001 snap_step=10")
        write(joinpath(dir, "rtm_meta.txt"), "nx=$n nz=$n nb=$nb h=4")
        plot_main([dir])
        for name in ("snapshot", "gather", "rtm")
            @test read(joinpath(dir, "$name.png"))[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        end
        @test_throws ErrorException read_f32(joinpath(dir, "snapshot.f32"), (n+1, n))
        @test laplacian(ones(Float32, 8, 8)) == zeros(Float32, 8, 8)
        quad = Float32[i^2 + j^2 for i in 1:8, j in 1:8]
        @test all(laplacian(quad)[2:7, 2:7] .== 4)
        for k in 1:6
            write_f32(joinpath(dir, "timelapse_$k.f32"), wave)
        end
        write(joinpath(dir, "timelapse.meta"), "n=$n h=4 times=0.1,0.2,0.3,0.4,0.5,0.6")
        write(joinpath(dir, "edges.meta"), "n=$n h=4")
        write(joinpath(dir, "disp.meta"), "n=$n h=4")
        write(joinpath(dir, "cfl.meta"), "nt=$nt")
        write(joinpath(dir, "rtm.meta"), "n=$n h=4 nb=$nb dt=0.001 nt=$nt nrec=$(n-2nb)")
        for name in ("edges_on", "edges_off", "disp_2nd", "disp_8th", "ic_S", "ic_R", "ic_I",
                     "rtm_1", "rtm_2", "rtm_4", "rtm_8", "rtm_16")
            write_f32(joinpath(dir, "$name.f32"), wave)
        end
        for name in ("095", "105")
            write_f32(joinpath(dir, "cfl_$name.f32"), 1:nt)
        end
        write_f32(joinpath(dir, "gather_obs.f32"), wave[nb+1:n-nb, 1:nt])
        plot_experiments(dir)
        for name in ("timelapse", "edges", "dispersion", "cfl", "rtm_stack", "imaging_condition")
            @test filesize(joinpath(dir, "$name.png")) > 0
        end
    end
end
