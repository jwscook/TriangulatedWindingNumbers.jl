struct Vertex{T<:Number, U<:Complex}
  x::AbstractVector{T}
  value::U
end
value(v::Vertex) = v.value
position(v::Vertex) = v.x
function areidentical(a::Vertex, b::Vertex)
  return all(position(a) .== position(b)) && value(a) == value(b)
end

import Base: isless, +, -
Base.:isless(a::Vertex, b::Vertex) = value(a) < value(b)
Base.:+(a::Vertex, b) = position(a) .+ b
Base.:-(a::Vertex, b::Vertex) = position(a) .- position(b)

function cosinelaw(va::Vertex, vb::Vertex)
  a = abs(value(va))
  b = abs(value(vb))
  c = abs(value(va) - value(vb))
  return acos((a^2 + b^2 - c^2) / (2 * a * b))
end


