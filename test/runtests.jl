using Random, Test, TriangulatedWindingNumbers
using TriangulatedWindingNumbers: Vertex, Simplex, windingnumber, windingangle, centroid

@testset "TriangulatedWindingNumbers tests" begin

  Random.seed!(0)

  @testset "Simplex tests" begin
    irrelevant = [0.0, 0.0]
    @testset "Simplex encloses zero" begin
      v1 = Vertex(irrelevant, 1.0 - im)
      v2 = Vertex(irrelevant, 0.0 + im)
      v3 = Vertex(irrelevant, -1.0 - im)
      encloseszero = Simplex([v1, v2, v3])
      @test isapprox(1, abs(windingangle(encloseszero)) / (2π))
      @test windingnumber(encloseszero) == 1
    end
    @testset "Simplex doesn't enclose zero" begin
      v1 = Vertex(irrelevant, 1.0 + im)
      v2 = Vertex(irrelevant, 2.0 + im)
      v3 = Vertex(irrelevant, 1.0 + im * 2)
      doesntenclosezero = Simplex([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
      v1 = Vertex(irrelevant, 1.0 + im)
      v2 = Vertex(irrelevant, 0.0 + 2*im)
      v3 = Vertex(irrelevant, -1.0 + im)
      doesntenclosezero = Simplex([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
      v1 = Vertex(irrelevant, 1.0 + 0im)
      v2 = Vertex(irrelevant, 0.0 + im)
      v3 = Vertex(irrelevant, 1.0 + 0im)
      doesntenclosezero = Simplex([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
    end
  end

  @testset "End-to-end tests" begin
    for i in 1:10
      gridisize = [rand(2:10), rand(2:10)]
      xtol_abs=10.0^(-rand(3:15))
      function mock(x::Vector, root)
        return (x[1] + im * x[2]) - root
      end
      root = rand(ComplexF64)
      objective(x) = mock(x, root)
      lower = collect(reim(root)) .- rand(2)
      upper = collect(reim(root)) .+ rand(2)
      solutions = TriangulatedWindingNumbers.solve(objective, lower, upper,
        gridisize, xtol_abs=xtol_abs)
      for (s, reason) ∈ solutions
        @test isapprox(centroid(s)[1], real(root), atol=xtol_abs)
        @test isapprox(centroid(s)[2], imag(root), atol=xtol_abs)
      end
    end
  end
end

