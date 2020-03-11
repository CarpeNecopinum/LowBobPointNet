struct TNet
    layers::Chain
end
Flux.@functor TNet

function TNet(dim)

end


struct LowBobPointNet
    enc1::Chain # goes up to the second nx64 point
    enc2::Chain # Computes the global feature 1024
    decoder::Chain # Takes a point and the global feature
end


struct PointMax end
#(::PointMax)(x) = maxpool(x, (size(x,1), 1))
(::PointMax)(x) = maxpool(x, PoolDims(size(x), (size(x,1), 1)))

# maximum(x, dims=1)

function LowBobPointNet(dim)
    enc1 = Chain(
        Conv((1,1), dim => 16, leakyrelu),
        Conv((1,1), 16 => 32, leakyrelu),
        Conv((1,1), 32 => 64, leakyrelu),
        Conv((1,1), 64 => 64, leakyrelu)
    )
    enc2 = Chain(
        Conv((1,1), 64 => 128, leakyrelu),
        Conv((1,1), 128 => 256, leakyrelu),
        Conv((1,1), 256 => 512, leakyrelu),
        Conv((1,1), 512 => 1024, leakyrelu),
        PointMax()
    )
    dec = Chain(
        Conv((1,1), (1024 + dim) => 1024, leakyrelu),
        Conv((1,1), 1024 => 512, leakyrelu),
        Conv((1,1), 512 => 256, leakyrelu),
        Conv((1,1), 256 => 16, leakyrelu),
        Conv((1,1), 16 => 1)
    )
    LowBobPointNet(enc1, enc2, dec)
end

Flux.@functor LowBobPointNet

onegen = Flux.use_cuda[] ? CuArrays.ones : ones
Flux.Zygote.@adjoint onegen(ps...) = onegen(ps...), _ -> nothing

zerogen = Flux.use_cuda[] ? CuArrays.zeros : zeros


function (n::LowBobPointNet)(scene_points, query_points)
    local_features = model.enc1(scene_points)
    global_feature = model.enc2(local_features)

    global_repeated = global_feature .* onegen(Float32, size(query_points,1), 1, 1, 1)

    #global_repeated = repeat(global_feature; outer = (size(query_points,1), 1, 1, 1))
    combined = cat(query_points, global_repeated; dims = 3)

    model.decoder(combined)
end
