# LowBobPointNet
Toy example of a PointNet inspired NN learning the unsigned distance function of a square room with a single piece of furniture.

## What is this?

This is a close to minimal piece of code meant to answer the question "Can a PointNet learn anything at all about a point cloud?" and also produce some neat pictures along the way.

It uses [Flux.jl](https://github.com/FluxML/Flux.jl) with [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl) for hardware-accelerated learning and [Makie.jl](https://github.com/JuliaPlots/Makie.jl) for visualization.

The network architecture is inspired by [PointNet](https://arxiv.org/abs/1612.00593), but leaves out all of the non-trivial bits that weren't needed for the network to show a resemblance of success (Norm layers and Transform-Nets).

## How to run

Open the `main.jl` in [Juno](https://junolab.org/) and run the first 2 cells, then run either the 3rd cell (to produce a PNG sequence) or 4th cell (to produce a MKV video).

You can save the model as a bson file by running the last cell.
