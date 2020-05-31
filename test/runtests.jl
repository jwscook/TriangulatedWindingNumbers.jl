using Random, Test, TriangulatedWindingNumbers
using TriangulatedWindingNumbers: Vertex, Simplex, windingnumber, windingangle
using TriangulatedWindingNumbers: centroid, assessconvergence, position, value

@testset "TriangulatedWindingNumbers tests" begin

  Random.seed!(0)

  @testset "Simplex tests" begin

    irrelevant = [0.0, 0.0]
    @testset "Simplex encloses zero" begin
      v1 = Vertex(irrelevant, 1.0 - im)
      v2 = Vertex(irrelevant, 0.0 + im)
      v3 = Vertex(irrelevant, -1.0 - im)
      encloseszero = Simplex{Float64, ComplexF64}([v1, v2, v3])
      @test isapprox(1, abs(windingangle(encloseszero)) / (2π))
      @test windingnumber(encloseszero) == 1
    end

    @testset "Simplex doesn't enclose zero" begin
      v1 = Vertex(irrelevant, 1.0 + im)
      v2 = Vertex(irrelevant, 2.0 + im)
      v3 = Vertex(irrelevant, 1.0 + im * 2)
      doesntenclosezero = Simplex{Float64, ComplexF64}([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
      v1 = Vertex(irrelevant, 1.0 + im)
      v2 = Vertex(irrelevant, 0.0 + 2*im)
      v3 = Vertex(irrelevant, -1.0 + im)
      doesntenclosezero = Simplex{Float64, ComplexF64}([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
      v1 = Vertex(irrelevant, 1.0 + 0im)
      v2 = Vertex(irrelevant, 0.0 + im)
      v3 = Vertex(irrelevant, 1.0 + 0im)
      doesntenclosezero = Simplex{Float64, ComplexF64}([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
    end

    @testset "Simplices with identical vertices is converged" begin
      dim = 2
      T = Float64
      U = ComplexF64
      pos = T[1.0, 1.0]
      val = rand(U)
      v1 = Vertex(pos, val)
      v2 = Vertex(pos, val)
      v3 = Vertex(pos, val)
      s = Simplex{T,U}([v1, v2, v3])

      defaults = TriangulatedWindingNumbers.convergenceconfig(dim, T)
      isconverged, returncode = assessconvergence(s, defaults)
      @test isconverged
      @test returncode == :XTOL_REACHED
    end

    @testset "Simplices in a simplex eps apart are converged" begin
      dim = 2
      T = Float64
      U = ComplexF64
      v1 = Vertex([one(T), one(T)], rand(U))
      v2 = Vertex([one(T), one(T) + eps(T)], rand(U))
      v3 = Vertex([one(T) + eps(T), one(T)], rand(U))
      s = Simplex{T,U}([v1, v2, v3])
      defaults = TriangulatedWindingNumbers.convergenceconfig(dim, T)
      isconverged, returncode = assessconvergence(s, defaults)
      @test isconverged
      @test returncode == :XTOL_REACHED
    end

    @testset "Simplices in a chain eps apart are converged" begin
      dim = 2
      T = Float64
      U = ComplexF64
      v1 = Vertex([one(T), one(T)], rand(U))
      v2 = Vertex([one(T), one(T) + eps(T)], rand(U))
      v3 = Vertex([one(T), one(T) + eps(T) + eps(T)], rand(U))
      s = Simplex{T,U}([v1, v2, v3])
      defaults = TriangulatedWindingNumbers.convergenceconfig(dim, T)
      isconverged, returncode = assessconvergence(s, defaults)
      @assert all(isapprox(position(v1)[d], position(v2)[d],
                        atol=defaults[:xtol_abs][d],
                        rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
      @assert all(isapprox(position(v2)[d], position(v3)[d],
                        atol=defaults[:xtol_abs][d],
                        rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
      @assert !all(isapprox(position(v1)[d], position(v3)[d],
                        atol=defaults[:xtol_abs][d],
                        rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
      @test isconverged
      @test returncode == :XTOL_REACHED
    end

  end

  @testset "End-to-end tests roots" begin
    @testset "Single root" begin
      for i in 1:10
        gridsize = [rand(1:10), rand(1:10)]
        xtol_abs=10.0^(-rand(3:15))
        function mock(x::Vector, root)
          return (x[1] + im * x[2]) - root
        end
        root = rand(ComplexF64)
        objective(x) = mock(x, root)
        lower = collect(reim(root)) .- rand(2)
        upper = collect(reim(root)) .+ rand(2)
        solutions = TriangulatedWindingNumbers.solve(objective, lower, upper,
          gridsize, xtol_abs=xtol_abs)
        @test length(solutions) == 1
        for (s, reason) ∈ solutions
          @test isapprox(centroid(s)[1], real(root), atol=xtol_abs)
          @test isapprox(centroid(s)[2], imag(root), atol=xtol_abs)
        end
      end
    end

    @testset "Single pole" begin
      for i in 1:10
        gridsize = [rand(1:10), rand(1:10)]
        xtol_abs=10.0^(-rand(3:15))
        function mock(x::Vector, root)
          return 1 / ((x[1] + im * x[2]) - root)
        end
        root = rand(ComplexF64)
        objective(x) = mock(x, root)
        lower = collect(reim(root)) .- rand(2)
        upper = collect(reim(root)) .+ rand(2)
        solutions = TriangulatedWindingNumbers.solve(objective, lower, upper,
          gridsize, xtol_abs=xtol_abs)
        @test length(solutions) == 1
        for (s, reason) ∈ solutions
          @test isapprox(centroid(s)[1], real(root), atol=xtol_abs)
          @test isapprox(centroid(s)[2], imag(root), atol=xtol_abs)
        end
      end
    end

    @testset "Multiple roots" begin
      for i in 1:100
        gridsize = [rand(5:20), rand(5:20)]
        xtol_abs=10.0^(-rand(3:15))
        function multimock(x::Vector, roots)
          return mapreduce(root->(x[1] + im * x[2]) - root, *, roots)
        end
        lower = [0.0, 0.0]
        upper = [1.0, 1.0]
        roots = [rand(ComplexF64) for i ∈ 1:rand(1:10)]
        objective(x) = multimock(x, roots)
        solutions = TriangulatedWindingNumbers.solve(objective, lower, upper,
          gridsize, xtol_abs=xtol_abs, stopvalroot=1e-20)
        for (s, reason) ∈ solutions
          passed = false
          for root ∈ roots
            passed |= (isapprox(centroid(s)[1], real(root), atol=xtol_abs) &&
                       isapprox(centroid(s)[2], imag(root), atol=xtol_abs))
          end
          @test passed
        end
      end
    end

    @testset "Multiple poles" begin
      for i in 1:100
        gridsize = [rand(5:20), rand(5:20)]
        xtol_abs=10.0^(-rand(3:15))
        function multimock(x::Vector, roots)
          return mapreduce(root->1 / ((x[1] + im * x[2]) - root), *, roots)
        end
        lower = [0.0, 0.0]
        upper = [1.0, 1.0]
        roots = [rand(ComplexF64) for i ∈ 1:rand(1:10)]
        objective(x) = multimock(x, roots)
        solutions = TriangulatedWindingNumbers.solve(objective, lower, upper,
          gridsize, xtol_abs=xtol_abs, stopvalpole=1e20)
        for (s, reason) ∈ solutions
          passed = false
          for root ∈ roots
            passed |= (isapprox(centroid(s)[1], real(root), atol=xtol_abs) &&
                       isapprox(centroid(s)[2], imag(root), atol=xtol_abs))
          end
          @test passed
        end
      end
    end

    @testset "xtol_rel is eps and xtol_abs zero stopvalroot is eps" begin
      for i in 1:10
        gridsize = [rand(1:10), rand(1:10)]
        function mock(x::Vector, root)
          return (x[1] + im * x[2]) - root
        end
        root = rand(ComplexF64)
        objective(x) = mock(x, root)
        lower = collect(reim(root)) .- rand(2)
        upper = collect(reim(root)) .+ rand(2)
        solutions = TriangulatedWindingNumbers.solve(objective, lower, upper,
          gridsize, xtol_abs=0.0, xtol_rel=eps(), stopvalroot=eps())
        @test !isempty(solutions)
        for (s, reason) ∈ solutions
          @test reason == :XTOL_REACHED || reason == :STOPVAL_ROOT_REACHED
          reason == :STOPVAL_ROOT_REACHED && continue
          difference = sqrt(sum(centroid(s) - [real(root), imag(root)]).^2)
          @test isapprox(difference, 0, atol=eps(), rtol=0)
          @test isapprox(centroid(s)[1], real(root), atol=0, rtol=eps())
          @test isapprox(centroid(s)[2], imag(root), atol=0, rtol=eps())
        end
      end
    end

  end

end

