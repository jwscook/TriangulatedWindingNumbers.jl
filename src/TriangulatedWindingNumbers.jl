module TriangulatedWindingNumbers

include("Vertexes.jl")
include("Simplexes.jl")

function solve(f::T, lower::AbstractVector{U}, upper::AbstractVector{V},
               gridsize::Union{Int, Vector{Int}}; kwargs...
               ) where {T<:Function, U<:Number, V<:Number}
  dim = length(lower)
  typeof(gridsize) <: Real && (gridsize = gridsize .* ones(Int, dim))
  index2position(i) = (i .- 1) ./ (gridsize .- 1) .* (upper .- lower) .+ lower
  index2values = Dict()
  totaltime = @elapsed for ii ∈ CartesianIndices(Tuple(gridsize .* ones(Int, dim)))
    index = collect(Tuple(ii))
    x = index2position(index)
    index2values[index] = f(x)
  end
  VT1 = promote_type(U, V)
  VT2 = typeof(first(index2values)[2])
  simplices = Set{Simplex{VT1, VT2}}()
  function generatesimplices!(simplices, direction)
    for ii ∈ CartesianIndices(Tuple((gridsize .* ones(Int, dim))))
      vertices = Vector{Vertex{VT1, VT2}}()
      index = collect(Tuple(ii))
      for i ∈ 1:dim + 1
        vertexindex = [index[j] + ((j == i) ? direction : 0) for j ∈ 1:dim]
        all(1 .<= vertexindex .<= gridsize) || continue
        vertex = Vertex{VT1, VT2}(index2position(vertexindex),
                                  index2values[vertexindex])
        push!(vertices, vertex)
      end
      length(vertices) != dim + 1 && continue
      push!(simplices, Simplex{VT1, VT2}(vertices))
    end
  end
  totaltime += @elapsed generatesimplices!(simplices, 1)
  totaltime += @elapsed generatesimplices!(simplices, -1)
  @assert length(simplices) == 2 * prod((gridsize .- 1))
  return _solve(f, collect(simplices), totaltime; kwargs...)
end

function convergenceconfig(dim::Int, T::Type; kwargs...)
  kwargs = Dict(kwargs)
  timelimit = get(kwargs, :timelimit, Inf)
  xtol_abs = get(kwargs, :xtol_abs, zeros(T)) .* ones(Bool, dim)
  xtol_rel = get(kwargs, :xtol_rel, eps(T)) .* ones(Bool, dim)
  ftol_rel = get(kwargs, :ftol_rel, eps())
  stopvalroot = get(kwargs, :stopvalroot, eps()) # zero makes it go haywire
  stopvalpole = get(kwargs, :stopvalpole, Inf)
  any(iszero.(xtol_rel) .& iszero.(xtol_abs)) && error("xtol_rel .& xtol_abs
                                                       must not contain zeros")
  return (timelimit=timelimit, xtol_abs=xtol_abs, xtol_rel=xtol_rel,
          ftol_rel=ftol_rel, stopvalroot=stopvalroot, stopvalpole=stopvalpole)
end

function _solve(f::F, simplices::AbstractVector{Simplex{T, U}}, totaltime=0.0;
    kwargs...) where {F<:Function, T<:Number, U<:Complex}

  config = convergenceconfig(dimensionality(first(simplices)), T; kwargs...)

  solutions = Vector{Tuple{eltype(simplices), Symbol}}()
#  iterations = 0
  while totaltime < config[:timelimit]
    newsimplices = Vector{eltype(simplices)}()
    for (i, simplex) ∈ enumerate(simplices)
      windingnumber(simplex) == 0 && continue
      innermost = closestomiddlevertex(simplex)
      Δt = @elapsed centroidvertex = centroidignorevertex(f, simplex, innermost)
#      @show abs(value(maxabsvaluevertex(simplex))); for v ∈ simplex; @show v; end
      for vertex ∈ simplex
        areidentical(vertex, innermost) && continue
        newsimplex = deepcopy(simplex)
        swap!(newsimplex, vertex, centroidvertex)
        all(areidentical.(newsimplex, simplex)) && continue
#        iterations += 1
#        @show abs(value(maxabsvaluevertex(newsimplex))); for v ∈ newsimplex; @show v; end
        windingnumber(newsimplex) == 0 && continue
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
