struct Vertex{T<:Complex, V<:AbstractVector{<:Number}}
  position::V
  value::T
end
value(v::Vertex) = v.value
position(v::Vertex) = v.position
import Base.isequal
function Base.isequal(a::Vertex, b::Vertex)
  values_equal = value(a) == value(b) || (isnan(value(a)) && isnan(value(b)))
  positions_equal = all(position(a) .== position(b))
  return values_equal && positions_equal
end
import Base.hash
Base.hash(v::Vertex) = hash(v, hash(:Vertex))
Base.hash(v::Vertex, h::UInt64) = hash(hash.(v.position), hash(v.value, h))
