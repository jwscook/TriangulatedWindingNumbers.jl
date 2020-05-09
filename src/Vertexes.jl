struct Vertex{T<:Number, U<:Complex}
  x::AbstractVector{T}
  value::U
end
value(v::Vertex) = v.value
position(v::Vertex) = v.x
function areidentical(a::Vertex, b::Vertex)
  return all(position(a) .== position(b)) && value(a) == value(b)
end
