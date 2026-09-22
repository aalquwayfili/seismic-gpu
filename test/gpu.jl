using Test, Random
include(joinpath(@__DIR__, "..", "src", "rtm.jl"))
CUDA.functional() || error("GPU tests require a working CUDA driver")
CUDA.allowscalar(false)

@testset "Kernel agreement on a partial block" begin
    Random.seed!(7)
    nx, nz = 137, 133
    S = Sim(Float32, two_layer(Float32, nx, nz); nb=12)
    u = randn(Float32, nx, nz)
    u[1:H, :] .= 0
    u[end-H+1:end, :] .= 0
    u[:, 1:H] .= 0
    u[:, end-H+1:end] .= 0
    up = zeros(Float32, nx, nz)
    next = similar(up)
    r2, gx, gz = Array(S.r2), Array(S.gx), Array(S.gz)
    for _ in 1:12
        fill!(next, 0)
        for iz in H+1:nz-H, ix in H+1:nx-H
            lap = 2S.c[1] * u[ix, iz]
            for k in 1:H
                lap += S.c[k+1] * (u[ix-k, iz] + u[ix+k, iz] + u[ix, iz-k] + u[ix, iz+k])
            end
            g = max(gx[ix], gz[iz])
            next[ix, iz] = (2-g) * u[ix, iz] - (1-g) * up[ix, iz] + r2[ix, iz] * lap
        end
        up, u, next = u, next, up
    end
    reference = copy(u)
    Random.seed!(7)
    initial = randn(Float32, nx, nz)
    initial[1:H, :] .= 0
    initial[end-H+1:end, :] .= 0
    initial[:, 1:H] .= 0
    initial[:, end-H+1:end] .= 0
    for kernel in (:cols, :rows, :tile, :reg)
        F = Fields(CuArray(initial), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz))
        for i in 1:12
            step!(S, F, kernel, i * S.dt; src=false)
        end
        @test Array(F.u) ≈ reference rtol=1f-4
    end
end

@testset "Boundary reconstruction" begin
    n, nb, nt = 128, 12, 180
    v = two_layer(Float32, n, n)
    S = Sim(Float32, smooth(v, 4); nb)
    rx, izr = nb+1:n-nb, nb+6
    _, data, _ = forward!(Sim(Float32, v; nb), :rows, nt, izr, rx)
    boundary, full = CUDA.zeros(Float32, n, n), CUDA.zeros(Float32, n, n)
    bytes, err = migrate_shot!(boundary, S, :rows, nt, izr, rx, data;
                                store=:boundary, check_at=nt÷2)
    full_bytes, _ = migrate_shot!(full, S, :rows, nt, izr, rx, data; store=:full)
    @test isfinite(err) && err < 1f-4
    @test bytes < full_bytes
    @test maximum(abs, Array(full)) > 0
    @test Array(boundary) ≈ Array(full) rtol=1f-3
end

module SplitGrid
include(joinpath(@__DIR__, "..", "src", "two_gpu.jl"))
end

@testset "Split grid" begin
    for overlap in (false, true)
        @test isnothing(SplitGrid.run_split(; n=128, steps=180, devs=(0, 0), overlap, check=true))
    end
    if length(CUDA.devices()) >= 2
        for overlap in (false, true)
            @test isnothing(SplitGrid.run_split(; n=128, steps=180, devs=(0, 1), overlap, check=true))
        end
    else
        @info "Two-device checks skipped: only one CUDA device is available"
    end
end
