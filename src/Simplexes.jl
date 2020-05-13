struct Simplex{T<:Number, U<:Complex}
  vertices::AbstractVector{Vertex{T,U}}
  function Simplex{T,U}(vertices::AbstractVector{Vertex{T,U}}
      ) where {T<:Number, U<:Complex}
    sort!(vertices, by=v->angle(value(v)))
    return new{T,U}(vertices)
  end
end

import Base.length, Base.iterate, Base.push!, Base.iterate, Base.getindex
import Base.eachindex, Base.sort!
Base.length(s::Simplex) = length(s.vertices)
Base.push!(s::Simplex, v::Vertex) = push!(s.vertices, v)
Base.iterate(s::Simplex) = iterate(s.vertices)
Base.iterate(s::Simplex, counter) = iterate(s.vertices, counter)
Base.getindex(s::Simplex, index) = s.vertices[index]
Base.eachindex(s::Simplex) = eachindex(s.vertices)
Base.sort!(s::Simplex; kwargs...) = sort!(s.vertices; kwargs...)

remove!(s::Simplex, v::Vertex) = filter!(x -> !areidentical(x, v), s.vertices)

sortbyangle(s::Simplex) = sort!(s, by=v->angle(value(v)))
issortedbyangle(s::Simplex) = issorted(s, by=v->angle(value(v)))

dimensionality(s::Simplex) = length(s) - 1

function getvertex(s::Simplex, i::Int)
  @assert 1 <= i <= length(s)
  @assert issortedbyangle(s)
  return s[i]
end

function findabsvaluevertex(s::Simplex, op::T) where {T<:Function}
  _, index = op(map(v -> abs(value(v)), s))
  return s[index]
end

minabsvaluevertex(s::Simplex) = findabsvaluevertex(s, findmin)
maxabsvaluevertex(s::Simplex) = findabsvaluevertex(s, findmax)

function centroidignorevertex(f::T, s::Simplex, vertextoignore::Vertex
    ) where {T<:Function, U<:Function}
  g(v) = areidentical(v, vertextoignore) ? zero(position(v)) : position(v)
  x = mapreduce(v -> g(v) / (length(s) - 1), +, s)
  return Vertex(x, f(x))
end

centroid(s::Simplex) = mapreduce(v -> position(v) ./ length(s), +, s)
function closestomiddlevertex(s::Simplex)
  mid = centroid(s)
  _, index = findmin(map(v->sum((position(v) - mid).^2), s))
  return s[index]
end

function swap!(s::Simplex, this::Vertex, forthat::Vertex)
  lengthbefore = length(s)
  remove!(s, this)
  @assert length(s) == lengthbefore - 1
  push!(s, forthat)
  sortbyangle(s)
  @assert length(s) == lengthbefore
  return nothing
end

function assessconvergence(simplex, config::NamedTuple)
  if abs(value(minabsvaluevertex(simplex))) <= config[:stopvalroot]
    return true, :STOPVAL_ROOT_REACHED
  elseif abs(value(maxabsvaluevertex(simplex))) >= config[:stopvalpole]
    return true, :STOPVAL_POLE_REACHED
  end

  toprocess = Set{Int}(1)
  processed = Set{Int}()
  @inbounds while !isempty(toprocess)
    vi = pop!(toprocess)
    v = getvertex(simplex, vi)
    connectedto = Set{Int}()
    for (qi, q) ∈ enumerate(simplex)
      thisxtol = true
      for (i, (pv, pq)) ∈ enumerate(zip(position(v), position(q)))
        thisxtol &= isapprox(pv, pq, rtol=config[:xtol_rel][i],
                                     atol=config[:xtol_abs][i])
      end
      thisxtol && push!(connectedto, qi)
      thisxtol && for i in connectedto if i ∉ processed push!(toprocess, i) end end
    end
    push!(processed, vi)
  end
  allxtol = all(i ∈ processed for i ∈ 1:length(simplex))
  allxtol && return true, :XTOL_REACHED

  allftol = true
  @inbounds for (vi, v) ∈ enumerate(simplex)
    for qi ∈ vi+1:length(simplex)
      q = getvertex(simplex, qi)
      allftol &= all(isapprox(value(v), value(q), rtol=config[:ftol_rel], atol=0))
    end
  end
  allftol && return true, :FTOL_REACHED

  return false, :CONTINUE
end

function _πtoπ(ϕ)
  ϕ < -π && return _πtoπ(ϕ + 2π)
  ϕ > π && return _πtoπ(ϕ - 2π)
  return ϕ
end
import Base.angle
angle(a::T, b::T) where {T<:Complex} = _πtoπ(angle(b) - angle(a))

function windingangle(s::Simplex)
  return sum(angle.(value.(s.vertices), value.(circshift(s.vertices, -1))))
end

function windingnumber(s::Simplex)
  radians = windingangle(s)
  return isfinite(radians) ? Int64(round(radians / 2π)) : Int64(0)
end


