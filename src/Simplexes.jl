struct Simplex{T<:Complex, V<:AbstractVector{<:Vertex}}
  vertices::V
  function Simplex(vertices::AbstractVector{Vertex{T,U}}
      ) where {T<:Complex, U}
    sort!(vertices, by=v->sortby(v, centroid(vertices)))
    return new{T,typeof(vertices)}(vertices)
  end
end

sortby(v::Vertex, c) = atan(reverse(position(v) .- c)...)

import Base: length, iterate, push!, iterate, getindex
import Base: eachindex, sort!, setindex!, hash, isequal
Base.length(s::Simplex) = length(s.vertices)
Base.iterate(s::Simplex) = iterate(s.vertices)
Base.iterate(s::Simplex, counter) = iterate(s.vertices, counter)
Base.getindex(s::Simplex, index) = s.vertices[index]
function Base.setindex!(s::Simplex, entry, index)
  s.vertices[index] = entry
  sortbyvertexangle!(s)
  return nothing
end
Base.hash(s::Simplex) = hash(s, hash(:Simplex))
Base.hash(s::Simplex, h::UInt64) = hash(hash.(s), h)
Base.isequal(a::Simplex, b::Simplex) = all(isequal.(a, b))

sortbyvertexangle!(s::Simplex) = sort!(s.vertices, lt=(a, b)->sortby(a, centroid(s)) < sortby(b, centroid(s)))
verticesaresorted(s::Simplex) = issorted(s.vertices, lt=(a, b)->sortby(b, centroid(s)) < sortby(b, centroid(s)))

dimensionality(s::Simplex) = length(s) - 1

function getvertex(s::Simplex, i::Int)
  @assert 1 <= i <= length(s)
  return s[i]
end

function findabsvaluevertex(s::Simplex, op::T) where {T<:Function}
  _, index = op(map(v -> abs(value(v)), s))
  return s[index]
end

minabsvaluevertex(s::Simplex) = findabsvaluevertex(s, findmin)
maxabsvaluevertex(s::Simplex) = findabsvaluevertex(s, findmax)

function centroidignorevertex(f::T, s::Simplex, vertextoignore::Vertex
    ) where {T<:Function}
  g(v) = isequal(v, vertextoignore) ? zero(position(v)) : position(v)
  x = mapreduce(g, +, s) ./ (length(s) - 1)
  return Vertex(x, f(x))
end

centroid(s) = mapreduce(position, +, s) ./ length(s)
function closestomiddlevertex(s::Simplex)
  mid = centroid(s)
  _, index = findmin(map(v->sum((position(v) - mid).^2), s))
  return s[index]
end

function swap!(s::Simplex, this::Vertex, forthat::Vertex)
  index = findfirst(x->isequal(x, this), s.vertices)
  s[index] = forthat
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
  while !isempty(toprocess)
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
  for (vi, v) ∈ enumerate(simplex)
    for qi ∈ vi+1:length(simplex)
      q = getvertex(simplex, qi)
      allftol &= all(isapprox(value(v), value(q), rtol=config[:ftol_rel],
                              atol=config[:ftol_abs]))
    end
  end
  allftol && return true, :FTOL_REACHED

  return false, :CONTINUE
end

function _πtoπ(ϕ::T) where {T}
  ϕ <= -T(π) && return _πtoπ(ϕ + 2π)
  ϕ > T(π) && return _πtoπ(ϕ - 2π)
  return ϕ
end

function windingangle(s::Simplex{T}) where {T}
  θ = zero(real(T))
  @inbounds for i in 1:length(s)
    θ += _πtoπ(angle(value(s[mod1(i+1, length(s))])) - angle(value(s[i])))
  end
  return θ
end

function windingnumber(s::Simplex)
  radians = windingangle(s)
  return isfinite(radians) ? Int64(round(radians / 2π)) : Int64(0)
end


