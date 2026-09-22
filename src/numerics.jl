const H = 4  # Stencil radius.
# Centre coefficient followed by offsets 1:4.
const C8 = (-205/72, 8/5, -1/5, 8/315, -1/560)

ricker(t, f0) = (1 - 2 * (π * f0 * (t - 1 / f0))^2) * exp(-(π * f0 * (t - 1 / f0))^2)

cfl_coeff(dim) = sqrt(4 / (dim * (abs(C8[1]) + 2 * sum(abs, C8[2:end]))))

function damping(n, nb, γmax)
    g = zeros(Float32, n)
    for i in 1:n
        d = min(i - 1, n - i)
        d < nb && (g[i] = γmax * (1 - d / nb)^2)
    end
    g
end

function two_layer(T, nx, nz; vtop=1500, vbot=2500)
    v = fill(T(vtop), nx, nz)
    for ix in 1:nx, iz in 1:nz
        iz > nz ÷ 2 + (ix - nx ÷ 2) ÷ 8 && (v[ix, iz] = T(vbot))
    end
    v
end

function smooth(v::Matrix{T}, r; passes=3) where T
    r >= 0 && passes >= 0 || throw(ArgumentError("radius and passes must be nonnegative"))
    w = copy(v)
    nx, nz = size(w)
    # Summed areas make each box average constant work, regardless of its radius.
    sums = zeros(Float64, nx + 1, nz + 1)
    for _ in 1:passes
        sums[2:end, 2:end] .= w
        cumsum!(sums, sums; dims=1)
        cumsum!(sums, sums; dims=2)
        for iz in 1:nz, ix in 1:nx
            x0, x1 = max(1, ix-r), min(nx, ix+r)
            z0, z1 = max(1, iz-r), min(nz, iz+r)
            total = sums[x1+1, z1+1] - sums[x0, z1+1] - sums[x1+1, z0] + sums[x0, z0]
            w[ix, iz] = total / ((x1-x0+1) * (z1-z0+1))
        end
    end
    w
end

function options(args, allowed)
    kv = Dict{String,String}()
    for arg in args
        pair = split(arg, '='; limit=2)
        length(pair) == 2 || throw(ArgumentError("expected key=value, got $arg"))
        key, value = pair
        key in allowed || throw(ArgumentError("unknown option: $key"))
        isempty(value) && throw(ArgumentError("missing value for $key"))
        haskey(kv, key) && throw(ArgumentError("duplicate option: $key"))
        kv[key] = value
    end
    kv
end

function flag(kv, key)
    value = get(kv, key, "0")
    value in ("0", "1") || throw(ArgumentError("$key must be 0 or 1"))
    value == "1"
end

function precision(kv)
    value = get(kv, "T", "Float32")
    value in ("Float32", "Float64") || throw(ArgumentError("T must be Float32 or Float64"))
    value == "Float64" ? Float64 : Float32
end

write_f32(path, a) = write(path, Float32.(a))
