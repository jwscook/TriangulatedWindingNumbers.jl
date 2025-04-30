module TriangulatedWindingNumbers

const Container{T} = Union{AbstractVector{T}, NTuple{N, T}} where N

include("Vertexes.jl")
include("Simplexes.jl")

function generatesimplices(f::F, lower::Container{N}, upper::Container{N},
    gridsize::Container{<:Integer}) where {F<:Function, N<:Number}
  all(gridsize .> 0) || throw(ArgumentError("gridsize must .> 0"))
  if !(length(lower) == length(upper) == length(gridsize))
    throw(ArgumentError("The lengths of lower, $lower, upper, $upper, and gridsize
                        $gridsize must be the same"))
  end
  if !(all(lower .< upper))
    throw(ArgumentError("lower, $lower must .< upper, $upper"))
  end
  dim = length(lower)
  index2position(i) = (i .- 1) ./ gridsize .* (upper .- lower) .+ lower
  index2values = Dict()
  totaltime = @elapsed for ii ∈ CartesianIndices(Tuple(gridsize .+ 1))
    index = collect(Tuple(ii))
    x = index2position(index)
    index2values[index] = f(x)
  end
  T = typeof(first(index2values)[2])
  V = typeof(lower)
  simplices = Set{Simplex{T, Vector{Vertex{T, V}}}}()
  function generatesimplices!(simplices, direction)
    for ii ∈ CartesianIndices(Tuple(((gridsize .+ 1) .* ones(Int, dim))))
      vertices = Vector{Vertex{T, V}}()
      index = collect(Tuple(ii))
      for i ∈ 1:dim + 1
        vertexindex = [index[j] + ((j == i) ? direction : 0) for j ∈ 1:dim]
        all(1 .<= vertexindex .<= gridsize .+ 1) || continue
        vertex = Vertex{T, V}(index2position(vertexindex),
                              index2values[vertexindex])
        push!(vertices, vertex)
      end
      length(vertices) != dim + 1 && continue
      push!(simplices, Simplex(vertices))
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
  if any(iszero.(xtol_rel) .& iszero.(xtol_abs))
    throw(ArgumentError("xtol_rel .& xtol_abs must not contain zeros"))
  end
  return (timelimit=timelimit, xtol_abs=xtol_abs, xtol_rel=xtol_rel,
          ftol_abs=ftol_abs, ftol_rel=ftol_rel, 
          stopvalroot=stopvalroot, stopvalpole=stopvalpole)
end


"""
    solve(f, lower, upper, gridsizeint; kwargs...)

Find the roots and poles of function, `f`, in a hyper-rectangle from
`lower` to `upper` with simplices generating in `gridsizeint`
sub-hyper-rectangles, and options passed in via kwargs.
"""
function solve(f::F, lower::Container{T}, upper::Container{T},
    gridsizeint::U; kwargs...) where {F<:Function, T<:Number, U<:Integer}
  gridsize = gridsizeint .* ones(typeof(gridsizeint), length(lower))
  return solve(f, lower, upper, gridsize;  kwargs...)
end

"""
    solve(f, lower, upper, gridsize; kwargs...)

Find the roots and poles of function, `f`, in a hyper-rectangle from
`lower` to `upper` with simplices generating in `gridsize`
sub-hyper-rectangles, and options passed in via kwargs.
"""
function solve(f::F, lower::Container{T}, upper::Container{T},
    gridsize::Container{<:Integer}; kwargs...) where {F<:Function, T<:Number}
  simplices, totaltime = generatesimplices(f, lower, upper, gridsize)
  return solve(f, collect(simplices); totaltime=totaltime, kwargs...)
end

"""
    solve(f, simplices, totaltime=0.0; targetwindingnumber=iszero, kwargs...)

Find the roots and poles of function, `f`, by subdividing `simplices` and
checking to see if the child simplices contain a root or a pole by evaluating
the winding number, and options passed in via kwargs.

Function `f` accepts an n-dimensional array as an argument and returns a complex
number.

Arguments
-  f: function to find the roots and poles of
-  simplices: collection of simplices to check
-  totaltime (kwarg) default 0: the taken to run the calculation, used to compare
against the timelimit kwarg.
-  targetwindingnumber (kwarg) default iszero: a function that accepts a winding number
of a simplex and allows the search to continue for that simplex if it returns true.

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
function solve(f::F, simplices::Container{Simplex{T, V}};
    totaltime=0.0, targetwindingnumber::W=!iszero,
    kwargs...) where {F<:Function, T<:Complex, V, W<:Function}

  config = convergenceconfig(dimensionality(first(simplices)), float(real(T));
                             kwargs...)

  solutions = Set{Tuple{eltype(simplices), Int64, Symbol}}()
  while totaltime < config[:timelimit]
    newsimplices = Vector{eltype(simplices)}()
    for (i, simplex) ∈ enumerate(simplices)
      targetwindingnumber(windingnumber(simplex)) || continue
      innermost = closestomiddlevertex(simplex)
      Δt = @elapsed centroidvertex = centroidignorevertex(f, simplex, innermost)
      for vertex ∈ simplex
        isequal(vertex, innermost) && continue
        swap!(simplex, vertex, centroidvertex)
        if targetwindingnumber(windingnumber(simplex))
          isconverged, returncode = assessconvergence(simplex, config)
          newsimplex = deepcopy(simplex)
          isconverged && push!(solutions, (newsimplex, windingnumber(simplex), returncode))
          isconverged || push!(newsimplices, newsimplex)
        end
        swap!(simplex, centroidvertex, vertex) # swap back
      end
      (totaltime += Δt) > config[:timelimit] && break
    end
    simplices = newsimplices
    isempty(simplices) && break
  end # while
  return solutions
end # solve

end
