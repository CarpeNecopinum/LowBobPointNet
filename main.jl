cd(@__DIR__)
import Pkg
Pkg.activate(".")

import NNlib
using Makie

using CuArrays
CuArrays.allowscalar(false)
using FileIO
using Flux
using Flux.Optimise
using GeometryTypes
using LinearAlgebra
using BSON
using GeometryTypes

const Edge = Tuple{Vec2f0, Vec2f0}

include("datagen.jl")
include("model.jl")
include("visualization.jl")

Flux.use_cuda[] = true

#---
model = nothing
show_room = random_room()

loss(scan_points, query_points, query_distances) =
    Flux.mse(model(scan_points, query_points), query_distances)

#--- Training with visualization in the plot pane + PNG sequence

Makie.AbstractPlotting.use_display[] = false

model = LowBobPointNet(2) |> gpu
opt = ADAM()

scene = visualize_model(model, show_room)
mkdir("training_gpu")
save("training_gpu/step_0.png", scene; resolution = (1024, 1024)
    )

Juno.@progress for i in 1:1000
    batch = make_batch(16) .|> gpu
    train!(loss, params(model), [batch], opt)
    scene = visualize_model(model, show_room)
    try
        save("training_gpu/step_$i.png", scene; resolution = (1024, 1024))
        display(scene)
    catch e
        println(e)
    end
end

#--- Training with visualization as video

Makie.AbstractPlotting.use_display[] = true

model = LowBobPointNet(2) |> gpu
opt = ADAM()
model_node = Node(model) # wrap the model in a Node to notify Makie.jl about changes
scene = visualize_model(model_node, show_room)

Makie.record(scene, "gpu3.mkv") do io
    Juno.@progress for i in 1:2000
        batch = make_batch(8) .|> gpu
        train!(loss, params(model), [batch], opt)
        push!(model_node, model)
        yield() # give Makie time to update
        recordframe!(io)
    end
end

#---

BSON.@save "model.bson" model
