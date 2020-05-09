struct Simplex{T<:Number, U<:Complex}
  vertices::Vector{Vertex{T,U}}
  function Simplex(vertices::Vector{Vertex{T,U}}) where {T<:Number, U<:Complex}
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

function getvertex(s::Simplex, i::Int)
  @assert 1 <= i <= length(s)
  @assert issortedbyangle(s)
  return s[i]
end

bestvertex(s::Simplex) = getvertex(s, 1)

function centroidignorevertex(f::T, s::Simplex, vertextoignore::Vertex
    ) where {T<:Function, U<:Function}
  g(v) = areidentical(v, vertextoignore) ? zero(position(v)) : position(v)
  x = mapreduce(g, +, s) / (length(s) - 1)
  return Vertex(x, f(x))
end

centroid(s::Simplex) = mapreduce(position, +, s) / length(s)
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

function assessconvergence(simplex, xtol_abs, xtol_rel, ftol_rel, stopval)
  abs(value(bestvertex(simplex))) <= stopval && return true, :STOPVAL_REACHED
  allxtol = true
  allftol = true
  @inbounds for vi ∈ eachindex(simplex)
    v = getvertex(simplex, vi)
    for qi ∈ vi+1:length(simplex)
      q = getvertex(simplex, qi)
      for (i, (pv, pq)) ∈ enumerate(zip(position(v), position(q)))
        allxtol &= all(isapprox.(pv, pq, rtol=xtol_rel[i], atol=xtol_abs[i]))
      end
      allftol &= all(isapprox(value(v), value(q), rtol=ftol_rel, atol=0))
    end
  end
  allxtol && return true, :XTOL_REACHED
  allftol && return true, :FTOL_REACHED
  return false, :CONTINUE
end

function _πtoπ(ϕ)
  ϕ < -π && return _πtoπ(ϕ + 2π)
  ϕ > π && return _πtoπ(ϕ - 2π)
  return ϕ
end
angle(a) = atan(imag(a), real(a))
angle(a, b) = _πtoπ(angle(b) - angle(a))

function windingangle(s::Simplex)
  return sum(angle.(value.(s.vertices), value.(circshift(s.vertices, -1))))
end

function windingnumber(s::Simplex)
  radians = windingangle(s)
  isnan(radians) && return Int64(0)
  return Int64(round(radians / 2π))
end


