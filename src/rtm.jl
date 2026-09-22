include(joinpath(@__DIR__, "wave2d.jl"))

struct Strip{T}
    xr::UnitRange{Int}
    zr::UnitRange{Int}
    left::CuArray{T,3}
    right::CuArray{T,3}
    top::CuArray{T,3}
    bottom::CuArray{T,3}
end
function Strip(T, n, nb, nt)
    xr = nb+1:n-nb
    zr = nb+1:n-nb
    lz = length(zr) + 2H
    lx = length(xr)
    Strip{T}(xr, zr, CuArray{T}(undef, H, lz, nt), CuArray{T}(undef, H, lz, nt),
             CuArray{T}(undef, lx, H, nt), CuArray{T}(undef, lx, H, nt))
end
strip_bytes(s::Strip) = sum(sizeof, (s.left, s.right, s.top, s.bottom))
function save!(s::Strip, u, it)
    xr, zr = s.xr, s.zr
    zw = first(zr)-H:last(zr)+H
    @views begin
        s.left[:, :, it]   .= u[first(xr)-H:first(xr)-1, zw]
        s.right[:, :, it]  .= u[last(xr)+1:last(xr)+H, zw]
        s.top[:, :, it]    .= u[xr, first(zr)-H:first(zr)-1]
        s.bottom[:, :, it] .= u[xr, last(zr)+1:last(zr)+H]
    end
end
function restore!(u, s::Strip, it)
    xr, zr = s.xr, s.zr
    zw = first(zr)-H:last(zr)+H
    @views begin
        u[first(xr)-H:first(xr)-1, zw] .= s.left[:, :, it]
        u[last(xr)+1:last(xr)+H, zw]   .= s.right[:, :, it]
        u[xr, first(zr)-H:first(zr)-1] .= s.top[:, :, it]
        u[xr, last(zr)+1:last(zr)+H]   .= s.bottom[:, :, it]
    end
end

function forward!(S::Sim{T}, kernel, nt, izr, rx; strip=nothing, full=nothing, keep=0) where T
    F = Fields(S)
    d = CUDA.zeros(T, length(rx), nt)
    kept = nothing
    for it in 1:nt
        step!(S, F, kernel, it * S.dt)
        @views d[:, it] .= F.u[rx, izr]
        strip === nothing || save!(strip, F.u, it)
        full === nothing || (@views full[:, :, it] .= F.u)
        it == keep && (kept = copy(F.u))
    end
    F, d, kept
end

function migrate_shot!(I, S::Sim{T}, kernel, nt, izr, rx, dtrue; store, check_at=0, capture=0, captured=nothing) where T
    store in (:boundary, :full) || throw(ArgumentError("store must be boundary or full"))
    strip = store == :boundary ? Strip(T, S.nx, S.nb, nt) : nothing
    full = store == :full ? CuArray{T}(undef, S.nx, S.nz, nt) : nothing
    F, modeled, kept = forward!(S, kernel, nt, izr, rx; strip, full, keep=check_at)
    d = dtrue .- modeled
    captured === nothing || (captured[:residual] = Array(d))
    # Reverse from u[t-1], u[t]; saved strips restore values lost to damping.
    F.u, F.up = F.up, F.u
    R = Fields(S)
    err = zero(T)
    # Only the undamped interior is reconstructed.
    xr = S.nb+1:S.nx-S.nb
    zr = S.nb+1:S.nz-S.nb
    for it in nt:-1:2
        if store == :boundary
            restore!(F.u, strip, it - 1)
            step!(S, F, kernel, it * S.dt)
            Sprev = F.up
        else
            Sprev = @view full[:, :, it-1]
        end
        if it - 1 == check_at && kept !== nothing
            err = @views maximum(abs, Sprev[xr, zr] .- kept[xr, zr]) / maximum(abs, kept[xr, zr])
        end
        # Inject the receiver trace after stepping the receiver wavefield.
        step!(S, R, kernel, zero(T); src=false)
        @views R.u[rx, izr] .+= d[:, it-1]
        @views I[xr, zr] .+= Sprev[xr, zr] .* R.u[xr, zr]
        if it - 1 == capture && captured !== nothing
            captured[:S] = Array(Sprev)
            captured[:R] = Array(R.u)
            captured[:I] = Array(I)
        end
    end
    bytes = store == :boundary ? strip_bytes(strip) : sizeof(full)
    bytes, err
end
