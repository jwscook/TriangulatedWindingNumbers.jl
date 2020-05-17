module TriangulatedWindingNumbers

include("Vertexes.jl")
include("Simplexes.jl")

function solve(f::F, lower::AbstractVector{T}, upper::AbstractVector{T},
    gridsizeint::U; kwargs...) where {F<:Function, T<:Number, U<:Integer}
  gridsize = gridsizeint .* ones(typeof(gridsizeint), length(lower))
  return solve(f, lower, upper, gridsize;  kwargs...)
end

function generatesimplices(f::F, lower::AbstractVector{T}, upper::AbstractVector{T},
    gridsize::AbstractVector{<:Integer}) where {F<:Function, T<:Number}
  @assert length(lower) == length(upper) == length(gridsize)
  dim = length(lower)
  index2position(i) = (i .- 1) ./ (gridsize .- 1) .* (upper .- lower) .+ lower
  index2values = Dict()
  totaltime = @elapsed for ii ∈ CartesianIndices(Tuple(gridsize))
    index = collect(Tuple(ii))
    x = index2position(index)
    index2values[index] = f(x)
  end
  U = typeof(first(index2values)[2])
  simplices = Set{Simplex{T, U}}()
  function generatesimplices!(simplices, direction)
    for ii ∈ CartesianIndices(Tuple((gridsize .* ones(Int, dim))))
      vertices = Vector{Vertex{T, U}}()
      index = collect(Tuple(ii))
      for i ∈ 1:dim + 1
        vertexindex = [index[j] + ((j == i) ? direction : 0) for j ∈ 1:dim]
        all(1 .<= vertexindex .<= gridsize) || continue
        vertex = Vertex{T, U}(index2position(vertexindex),
                              index2values[vertexindex])
        push!(vertices, vertex)
      end
      length(vertices) != dim + 1 && continue
      push!(simplices, Simplex{T, U}(vertices))
    end
  end
  totaltime += @elapsed generatesimplices!(simplices, 1)
  totaltime += @elapsed generatesimplices!(simplices, -1)
  @assert length(simplices) == 2 * prod((gridsize .- 1))
  return (simplices, totaltime)
end

function solve(f::F, lower::AbstractVector{T}, upper::AbstractVector{T},
    gridsize::AbstractVector{<:Integer}; kwargs...) where {F<:Function, T<:Number}
  simplices, totaltime = generatesimplices(f, lower, upper, gridsize)
  return solve(f, collect(simplices), totaltime; kwargs...)
end

function convergenceconfig(dim::Int, T::Type; kwargs...)
  kwargs = Dict(kwargs)
  timelimit = get(kwargs, :timelimit, Inf)
  xtol_abs = get(kwargs, :xtol_abs, zeros(T)) .* ones(Bool, dim)
  xtol_rel = get(kwargs, :xtol_rel, eps(T)) .* ones(Bool, dim)
  ftol_rel = get(kwargs, :ftol_rel, eps())
  stopvalroot = get(kwargs, :stopvalroot, eps()) # zero makes it go haywire
  stopvalpole = get(kwargs, :stopvalpole, Inf)
  solutiontype = get(kwargs, :solutiontype, :rootsandpoles)
  any(iszero.(xtol_rel) .& iszero.(xtol_abs)) && error("xtol_rel .& xtol_abs
                                                       must not contain zeros")
  return (timelimit=timelimit, xtol_abs=xtol_abs, xtol_rel=xtol_rel,
          ftol_rel=ftol_rel, stopvalroot=stopvalroot, stopvalpole=stopvalpole,
          solutiontype=solutiontype)
end

function solve(f::F, simplices::AbstractVector{Simplex{T, U}}, totaltime=0.0;
    kwargs...) where {F<:Function, T<:Number, U<:Complex}

  config = convergenceconfig(dimensionality(first(simplices)), T; kwargs...)
  function solutionselector(solutiontype)
    solutiontype == :rootsandpoles && return s->windingnumber(s) == 0 
    solutiontype == :roots && return s -> windingnumber(s) <= 0
    solutiontype == :poles && return s -> windingnumber(s) >= 0
  end
  selector = solutionselector(config[:solutiontype])

  solutions = Vector{Tuple{eltype(simplices), Symbol}}()
  while totaltime < config[:timelimit]
    newsimplices = Vector{eltype(simplices)}()
    for (i, simplex) ∈ enumerate(simplices)
      selector(simplex) && continue
      innermost = closestomiddlevertex(simplex)
      Δt = @elapsed centroidvertex = centroidignorevertex(f, simplex, innermost)
      for vertex ∈ simplex
        areidentical(vertex, innermost) && continue
        newsimplex = deepcopy(simplex)
        swap!(newsimplex, vertex, centroidvertex)
        areidentical(newsimplex, simplex) && continue
        selector(newsimplex) && continue
        isconverged, returncode = assessconvergence(newsimplex, config)
        isconverged && push!(solutions, (newsimplex, returncode))
        isconverged || push!(newsimplices, newsimplex)
      end
      (totaltime += Δt) > config[:timelimit] && break
    end
    simplices = newsimplices
    isempty(simplices) && break
  end # while
  return solutions
end # solve

end
