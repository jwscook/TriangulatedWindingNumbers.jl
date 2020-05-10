module TriangulatedWindingNumbers

include("Vertexes.jl")
include("Simplexes.jl")

function solve(f::T, lower::AbstractVector{U}, upper::AbstractVector{V},
               gridsize::Union{Int, Vector{Int}}; kwargs...
               ) where {T<:Function, U<:Number, V<:Number}
  dim = length(lower)
  typeof(gridsize) <: Real && (gridsize = gridsize .* ones(Int, dim))
  position(i) = (i .- 1) ./ (gridsize .- 1) .* (upper .- lower) .+ lower
  index2values = Dict()
  for ii ∈ CartesianIndices(Tuple(gridsize .* ones(Int, dim)))
    index = collect(Tuple(ii))
    x = position(index)
    index2values[index] = f(x)
  end
  simplices = Set{Simplex}()
  VT1 = promote_type(U, V)
  VT2 = typeof(first(index2values)[2])
  function generatesimplices!(simplices, direction)
    for ii ∈ CartesianIndices(Tuple((gridsize .* ones(Int, dim))))
      vertices = Vector{Vertex{VT1, VT2}}()
      index = collect(Tuple(ii))
      for i ∈ 1:dim + 1
        vertexindex = [index[j] + ((j == i) ? direction : 0) for j ∈ 1:dim]
        all(1 .<= vertexindex .<= gridsize) || continue
        vertex = Vertex{VT1, VT2}(position(vertexindex), index2values[vertexindex])
        push!(vertices, vertex)
      end
      length(vertices) != dim + 1 && continue
      push!(simplices, Simplex(vertices))
    end
  end
  totaltime = @elapsed generatesimplices!(simplices, 1)
  totaltime += @elapsed generatesimplices!(simplices, -1)
  @assert length(simplices) == 2 * prod((gridsize .- 1))

  return _solve(f, collect(simplices), totaltime; kwargs...)
end

function _solve(f::T, simplices::Vector{Simplex}, totaltime=0.0;
    kwargs...) where {T<:Function}
  kwargs = Dict(kwargs)
  timelimit = get(kwargs, :timelimit, Inf)
  xtol_abs = get(kwargs, :xtol_abs, eps()) .* ones(Int, dimensionality(first(simplices)))
  xtol_rel = get(kwargs, :xtol_rel, zeros(size(xtol_abs)))
  ftol_rel = get(kwargs, :ftol_rel, eps())
  stopval = get(kwargs, :stopval, 0.0)

  solutions = Vector{Tuple{eltype(simplices), Symbol}}()
  while !isempty(simplices) && totaltime < timelimit
    newsimplices = Vector{eltype(simplices)}()
    for (i, simplex) ∈ enumerate(simplices)
      windings = windingnumber(simplex)
      windings == 0 && continue
      innermost = closestomiddlevertex(simplex)
      Δt = @elapsed centroid = centroidignorevertex(f, simplex, innermost)
      for vertex ∈ simplex
        areidentical(vertex, innermost) && continue
        newsimplex = deepcopy(simplex)
        swap!(newsimplex, vertex, centroid)
        isconverged, returncode = assessconvergence(newsimplex, xtol_abs,
          xtol_rel, ftol_rel, stopval)
        isconverged && push!(solutions, (newsimplex, returncode))
        isconverged || push!(newsimplices, newsimplex)
        (totaltime += Δt) > timelimit && break
      end
    end
    simplices = newsimplices
  end # while
  return solutions
end # solve

end
