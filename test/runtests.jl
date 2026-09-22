using Test
include(joinpath(@__DIR__, "..", "src", "numerics.jl"))

@testset "Stencil" begin
    weights = [reverse(collect(C8[2:end])); C8[1]; collect(C8[2:end])]
    offsets = -H:H
    for power in 0:9
        expected = power == 2 ? 2.0 : 0.0
        @test sum(weights .* offsets.^power) ≈ expected atol=1e-10
    end
    @test cfl_coeff(2) ≈ 0.5546324796655889
end

@testset "Source and model" begin
    @test ricker(1 / 15, 15) == 1
    @test ricker(1 / 15 - 0.01, 15) ≈ ricker(1 / 15 + 0.01, 15)
    g = damping(128, 12, 0.08)
    @test g == reverse(g)
    @test all(iszero, g[13:116])
    @test g[1] ≈ 0.08
    @test all(diff(g[1:13]) .<= 0)
    v = two_layer(Float32, 128, 128)
    @test all(v[:, 1] .== 1500)
    @test all(v[:, end] .== 2500)
    @test smooth(fill(7f0, 16, 16), 3) == fill(7f0, 16, 16)
    blurred = smooth(v, 3; passes=1)
    @test extrema(blurred) == (1500f0, 2500f0)
    @test any(1500 .< blurred .< 2500)
end

@testset "Arguments and output" begin
    kv = options(["n=128", "all=0"], ("n", "all"))
    @test kv["n"] == "128"
    @test !flag(kv, "all")
    @test flag(Dict("check" => "1"), "check")
    @test precision(Dict("T" => "Float64")) == Float64
    @test_throws ArgumentError options(["n"], ("n",))
    @test_throws ArgumentError options(["n="], ("n",))
    @test_throws ArgumentError options(["n=128", "n=256"], ("n",))
    @test_throws ArgumentError options(["typo=128"], ("n",))
    @test_throws ArgumentError flag(Dict("all" => "yes"), "all")
    @test_throws ArgumentError precision(Dict("T" => "Float16"))
    mktempdir() do dir
        values = reshape(Float64.(1:12), 3, 4)
        path = joinpath(dir, "values.f32")
        write_f32(path, values)
        @test filesize(path) == 12 * sizeof(Float32)
        @test read!(path, zeros(Float32, 3, 4)) == values
    end
end

@testset "Box smoothing" begin
    function direct_smooth(v, r; passes=3)
        w = copy(v)
        nx, nz = size(v)
        for _ in 1:passes
            out = similar(w)
            for iz in 1:nz, ix in 1:nx
                patch = @view w[max(1, ix-r):min(nx, ix+r), max(1, iz-r):min(nz, iz+r)]
                out[ix, iz] = sum(patch) / length(patch)
            end
            w = out
        end
        w
    end
    for T in (Float32, Float64), r in (0, 1, 4, 30), passes in (0, 1, 3)
        v = T[1500 + 10sin(i + j) for i in 1:21, j in 1:17]
        before = copy(v)
        @test smooth(v, r; passes) ≈ direct_smooth(v, r; passes)
        @test v == before
    end
    @test_throws ArgumentError smooth(ones(3, 3), -1)
end
