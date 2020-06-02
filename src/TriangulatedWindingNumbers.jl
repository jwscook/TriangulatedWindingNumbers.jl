module TriangulatedWindingNumbers

include("Vertexes.jl")
include("Simplexes.jl")

function generatesimplices(f::F, lower::AbstractVector{T}, upper::AbstractVector{T},
    gridsize::AbstractVector{<:Integer}) where {F<:Function, T<:Number}
  all(gridsize .> 0) || error("gridsize must .> 0")
  if !(length(lower) == length(upper) == length(gridsize))
    error("The lengths of lower, $lower, upper, $upper, and gridsize $gridsize
          must be the same")
  end
  dim = length(lower)
  index2position(i) = (i .- 1) ./ gridsize .* (upper .- lower) .+ lower
  index2values = Dict()
  totaltime = @elapsed for ii ∈ CartesianIndices(Tuple(gridsize .+ 1))
    index = collect(Tuple(ii))
    x = index2position(index)
    index2values[index] = f(x)
  end
  U = typeof(first(index2values)[2])
  simplices = Set{Simplex{T, U}}()
  function generatesimplices!(simplices, direction)
    for ii ∈ CartesianIndices(Tuple(((gridsize .+ 1) .* ones(Int, dim))))
      vertices = Vector{Vertex{T, U}}()
      index = collect(Tuple(ii))
      for i ∈ 1:dim + 1
        vertexindex = [index[j] + ((j == i) ? direction : 0) for j ∈ 1:dim]
        all(1 .<= vertexindex .<= gridsize .+ 1) || continue
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
  @assert length(simplices) == 2 * prod(gridsize)
  return (simplices, totaltime)
end

function convergenceconfig(dim::Int, T::Type; kwargs...)
  kwargs = Dict(kwargs)
  timelimit = get(kwargs, :timelimit, Inf)
  xtol_abs = get(kwargs, :xtol_abs, zeros(T)) .* ones(Bool, dim)
  xtol_rel = get(kwargs, :xtol_rel, eps(T)) .* ones(Bool, dim)
  ftol_abs = get(kwargs, :ftol_abs, 0)
  ftol_rel = get(kwargs, :ftol_rel, eps())
  stopvalroot = get(kwargs, :stopvalroot, nextfloat(0.0)) # zero makes it go haywire
  stopvalpole = get(kwargs, :stopvalpole, Inf)
  any(iszero.(xtol_rel) .& iszero.(xtol_abs)) && error("xtol_rel .& xtol_abs
                                                       must not contain zeros")
  return (timelimit=timelimit, xtol_abs=xtol_abs, xtol_rel=xtol_rel,
          ftol_abs=ftol_abs, ftol_rel=ftol_rel, 
          stopvalroot=stopvalroot, stopvalpole=stopvalpole)
end


"""
    solve(f, lower, upper, gridsizeint; kwargs...)

Find the roots and poles of function, `f`, in a hyper-rectangle from
`lower` to `upper` with simplices generating in `gridsizeint`
sub-hyper-rectangles, and options passed in via kwargs.

Function `f` accepts an n-dimensional array as an argument and returns a complex
number.

# Keyword Arguments
-  stopval (default sqrt(eps())): stopping criterion when function evaluates
equal to or less than stopval
-  xtol_abs (default zeros(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this absolute tolerance
-  xtol_rel (default eps(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this relative tolerance
-  ftol_abs (default 0): stop if function evaluations at the
vertices are close to one another by this absolute tolerance
-  ftol_rel (default 1000eps(real(U))): stop if function evaluations at the
vertices are close to one another by this relative tolerance
-  timelimit (default Inf): stop if it takes longer than this in seconds
-  stopvalroot (default nextfloat(0.0): stop if any value is less than or
equal to this
-  stopvalpole (default Inf): stop if any value is greater than or equal to this
"""
function solve(f::F, lower::AbstractVector{T}, upper::AbstractVector{T},
    gridsizeint::U; kwargs...) where {F<:Function, T<:Number, U<:Integer}
  gridsize = gridsizeint .* ones(typeof(gridsizeint), length(lower))
  return solve(f, lower, upper, gridsize;  kwargs...)
end

"""
    solve(f, lower, upper, gridsizeint; kwargs...)

Find the roots and poles of function, `f`, in a hyper-rectangle from
`lower` to `upper` with simplices generating in `gridsizeint`
sub-hyper-rectangles, and options passed in via kwargs.

Function `f` accepts an n-dimensional array as an argument and returns a complex
number.

# Keyword Arguments
-  stopval (default sqrt(eps())): stopping criterion when function evaluates
equal to or less than stopval
-  xtol_abs (default zeros(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this absolute tolerance
-  xtol_rel (default eps(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this relative tolerance
-  ftol_abs (default 0): stop if function evaluations at the
vertices are close to one another by this absolute tolerance
-  ftol_rel (default 1000eps(real(U))): stop if function evaluations at the
vertices are close to one another by this relative tolerance
-  timelimit (default Inf): stop if it takes longer than this in seconds
-  stopvalroot (default nextfloat(0.0): stop if any value is less than or
equal to this
-  stopvalpole (default Inf): stop if any value is greater than or equal to this
"""
function solve(f::F, lower::AbstractVector{T}, upper::AbstractVector{T},
    gridsize::AbstractVector{<:Integer}; kwargs...) where {F<:Function, T<:Number}
  simplices, totaltime = generatesimplices(f, lower, upper, gridsize)
  return solve(f, collect(simplices), totaltime; kwargs...)
end

"""
    solve(f, simplices, totaltime=0.0; kwargs...)

Find the roots and poles of function, `f`, by subdividing `simplices` and
checking to see if the child simplices contain a root or a pole by evaluating
the winding number, and options passed in via kwargs.

Function `f` accepts an n-dimensional array as an argument and returns a complex
number.

# Keyword Arguments
-  stopval (default sqrt(eps())): stopping criterion when function evaluates
equal to or less than stopval
-  xtol_abs (default zeros(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this absolute tolerance
-  xtol_rel (default eps(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this relative tolerance
-  ftol_abs (default 0): stop if function evaluations at the
vertices are close to one another by this absolute tolerance
-  ftol_rel (default 1000eps(real(U))): stop if function evaluations at the
vertices are close to one another by this relative tolerance
-  timelimit (default Inf): stop if it takes longer than this in seconds
-  stopvalroot (default nextfloat(0.0): stop if any value is less than or
equal to this
-  stopvalpole (default Inf): stop if any value is greater than or equal to this
"""
function solve(f::F, simplices::AbstractVector{Simplex{T, U}}, totaltime=0.0;
    kwargs...) where {F<:Function, T<:Number, U<:Complex}

  config = convergenceconfig(dimensionality(first(simplices)), T; kwargs...)

  solutions = Vector{Tuple{eltype(simplices), Symbol}}()
  while totaltime < config[:timelimit]
    newsimplices = Vector{eltype(simplices)}()
    for (i, simplex) ∈ enumerate(simplices)
      windingnumber(simplex) == 0 && continue
      innermost = closestomiddlevertex(simplex)
      Δt = @elapsed centroidvertex = centroidignorevertex(f, simplex, innermost)
      for vertex ∈ simplex
        areidentical(vertex, innermost) && continue
        newsimplex = deepcopy(simplex)
        swap!(newsimplex, vertex, centroidvertex)
        areidentical(newsimplex, simplex) && continue
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
