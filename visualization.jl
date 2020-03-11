function visualize(edges::Vector{Edge})
    Makie.linesegments(edges; scale_plot = false)
    points = sample_edges(edges)
    Makie.scatter!(points; markersize = 0.01)

    train_points = map(x->x.+0.1f0randn(Vec2f0), sample_edges(edges))
    distances = map(x->distance(x, edges), train_points)

    Makie.scatter!(train_points; color = distances, markersize = 0.01)
end

function visualize_model(model, room)
    scan_points = sample_edges(room)
    scan_data = Flux.unsqueeze(points2mat(scan_points)[:,:,:],2) |> gpu

    xs = range(-1.2, 1.2; length = 128)
    query_points = vec(Vec2f0.(xs, xs'))
    query_data = Flux.unsqueeze(points2mat(query_points)[:,:,:],2) |> gpu

    dsts = model(scan_data, query_data)

    results = reshape(dsts, length(xs), length(xs)) |> cpu

    scene = Makie.Scene(; scale_plot = false, padding = (-0.05, -0.05), show_axis = false)
    scatter!(scene, query_points; color = vec(results), markersize = 0.01)
    linesegments!(scene, room; linewidth = 2)
    contour!(scene, xs, xs, results; levels = 5, linewidth = 2)
    scene
end


function visualize_model(model_node::Node, room)
    scan_points = sample_edges(room)
    scan_data = Flux.unsqueeze(points2mat(scan_points)[:,:,:],2) |> gpu

    xs = range(-1.2, 1.2; length = 128)
    query_points = vec(Vec2f0.(xs, xs'))
    query_data = Flux.unsqueeze(points2mat(query_points)[:,:,:],2) |> gpu

    results = lift(model_node) do model
        dsts = model(scan_data, query_data)
        r = reshape(dsts, length(xs), length(xs)) |> cpu
        r
    end

    scene = Makie.Scene(; scale_plot = false, padding = (-0.05, -0.05), show_axis = false)
    scatter!(scene, query_points; color = lift(vec, results), markersize = 0.005)
    linesegments!(scene, room; linewidth = 4)
    contour!(scene, xs, xs, results; levels = 5, linewidth = 2)
    scene
end

function visualize_batch(batch)
    scene = Makie.Scene(; scale_plot = false, padding = (-0.05, -0.05), show_axis = false)

    # scan points
    Makie.scatter!(scene, batch[1][:,1,:,1]; markersize = 0.01)

    # query points, with distance
    Makie.scatter!(scene, batch[2][:,1,:,1]; color = batch[3][:,1,1], markersize = 0.01)
end

#visualize_batch(make_batch(1))
