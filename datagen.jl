function rect_edges(r)
    [
        (Vec2f0(r.x,r.y), Vec2f0(r.x + r.w, r.y)),
        (Vec2f0(r.x + r.w, r.y), Vec2f0(r.x + r.w, r.y + r.h)),
        (Vec2f0(r.x + r.w, r.y + r.h), Vec2f0(r.x, r.y + r.h)),
        (Vec2f0(r.x, r.y + r.h), Vec2f0(r.x, r.y))
    ]
end

function random_room()
    walls = Rectangle(Vec2f0(-1,-1), Vec2f0(2,2))
    box = Rectangle(rand(Vec2f0) .- Vec2f0(0.75, 0.75), Vec2f0(0.5, 0.5))
    vcat(rect_edges(walls), rect_edges(box))
end

mix(x,y,a) = x + (y - x) * a

function sample_edges(edges)
    samples = Vec2f0[]
    for edge in edges
        len = norm(edge[1] - edge[2])
        for i in 1:round(Int, len / 0.01)
            push!(samples, mix(edge..., rand()))
        end
    end
    samples
end

sqrnorm(x) = x⋅x
function lineseg_distance(point, a, b)
    l2 = sqrnorm(b - a)
    t = clamp((point - a) ⋅ (b - a) / l2, 0f0, 1f0)
    norm(point - mix(a, b, t))
end

function distance(point, edges)
    minimum(edges) do edge
        lineseg_distance(point, edge...)
    end
end


veclen(x) = length(x)
veclen(::Type{Float32}) = 1
function points2mat(pts)
    C = veclen(eltype(pts))
    transpose(reshape(reinterpret(Float32, pts), C, :))
end

mixnd(x,y,a) = x .+ (y .- x) .* a

function make_batch(batch_size)
    scan_mats = Matrix{Float32}[]
    query_mats = Matrix{Float32}[]
    distance_mats = Matrix{Float32}[]

    for _ in 1:batch_size
        room = random_room()
        scan_points = sample_edges(room)

        edge_near_points = map(x->x.+0.1f0randn(Vec2f0), sample_edges(room))
        uniform_points = rand(Vec2f0, length(edge_near_points)) .* 2.2f0 .- (Vec2(1.1f0),)
        query_points = vcat(edge_near_points, uniform_points)

        distances = map(x->distance(x, room), query_points)
        push!(scan_mats, points2mat(scan_points))
        push!(query_mats, points2mat(query_points))
        push!(distance_mats, points2mat(distances))
    end
    Flux.unsqueeze(cat(scan_mats..., dims = 3), 2),
    Flux.unsqueeze(cat(query_mats..., dims = 3), 2),
    Flux.unsqueeze(cat(distance_mats..., dims = 3), 2)
end
