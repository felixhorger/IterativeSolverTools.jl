
module IterativeSolverTools

	using IterativeSolvers
	using LinearAlgebra
	import Roots

	"""
		debug_iterative_solver(iterator, debugger, types)

	Evaluate a function after each iteration of a given solver and store the result

	# Arguments
	- `iterator`: doing one iteration of any solver;
	- `debugger`: function to be evaluated at each iteration;
	- `types`: output types of `debugger`, can be single type (for single output of `debugger`)
	  or `Tuple` of types.

	# Output
	- `history`: Tuple of vectors holding the outputs of `debugger`.
	"""
	function debug_iterative_solver(iterator, debugger::Function, type::Type)
		debugger_tuple(iterator, v) = (debugger(iterator, v),)
		history = debug_iterative_solver(iterator, debugger_tuple, (type,)) |> first
		return history
	end
	function debug_iterative_solver(iterator, debugger::Function, types::NTuple{N, Type}) where N
		history = ntuple(n -> Vector{types[n]}(undef, iterator.maxiter), Val(N))
		j = 0
		for v in iterator
			j += 1
			d = debugger(iterator, v)
			for n in 1:N
				history[n][j] = d[n]
			end
		end
		history = ntuple(n -> (@view history[n][1:j]), Val(N))
		return history
	end


	"""
		cg_debugger(A, x::AbstractVector{<: Number}, y::AbstractVector{<: Number})

	Relevant quantities indicating the convergence of conjugate gradients applied to `A * y = b`:
	- A induced norm ``||x - y||^2_A`` which should be decreasing monotonously, where ``x`` is the current estimate.
	- root mean squared error ``||x - y||_2``.

	# Arguments
	- `A`: linear operator, must support matrix vector product;
	- `x` => `AbstractVector{< Number}`: current estimate;
	- `y` => `AbstractVector{<: Number}`: solution of ``A y = b``;
	  is evaluated several times with `y` staying the same.

	# Outputs
	- A induced norm of difference ``x - y``;
	- root mean squared error.
	"""
	function cg_debugger(A, x::AbstractVector{<: Number}, y::AbstractVector{<: Number})
		Δx = x .- y
		A_norm = real(dot(Δx, A, Δx)) # Dot should give a real result, dot() conjugates and A is Hermitian
		rmse = norm(Δx)
		return A_norm, rmse
	end

	"""
		debug_cg!(
			x::AbstractVector{<: Number},
			A,
			b::AbstractVector{<: Number},
			xhat::AbstractVector{<: Number};
			kwargs...
		)

	Apply conjugate gradients to `A * xhat = b`, logging the quantities computed by `cg_debugger` in every iteration.
	Useful for checking that the linear operator `A` works as intended or when to stop the algorithm
	early to be closer to the solution `xhat` in a non-least-squares sense.

	# Arguments
	- `A`: linear operator, must support matrix vector product;
	- `b` => `AbstractVector{< Number}`: current estimate;
	- `x0` => `AbstractVector{< Number}`: current estimate;
	- `xhat` => `AbstractVector{<: Number}`: solution of A * xhat = b;

	## Keywords
	Same as the ones for `cg_iterator!`.

	# Outputs
	- Solution found by conjugate gradients;
	- residuals;
	- history of A induced norm of difference `x - xhat`;
	- history of root mean squared error of `x` and `xhat`.
	"""
	function debug_cg!(
		x::AbstractVector{<: Number},
		A,
		b::AbstractVector{<: Number},
		xhat::AbstractVector{<: Number};
		maxiter, kwargs...
	)
		cg_iter = cg_iterator!(x, A, b; maxiter, kwargs...)
		debugger(iterator, residual) = begin
			A_norm, rmse = cg_debugger(A, iterator.x, xhat)
			return residual, A_norm, rmse
		end
		history = debug_iterative_solver(cg_iter, debugger, NTuple{3, Float64})
		iterations = length(history)
		# Split debugged quantities
		residuals = Vector{Float64}(undef, iterations)
		A_norm = Vector{Float64}(undef, iterations)
		nrmse = Vector{Float64}(undef, iterations)
		for i = 1:iterations
			residuals[i], A_norm[i], nrmse[i] = history[i]
		end
		nrmse ./= norm(xhat)
		return cg_iter.x, residuals, A_norm, nrmse
	end

	"""
		computes f(κ, n) in ||e^n|| ≤ f(κ, n) ||e^0||
		with f(κ, n) = 2 √κ ⋅ ( (√κ - 1) / (√κ + 1) )^n
	"""
	cg_convergence(κ::Real, n::Integer) = 2√κ ⋅ ( (√κ - 1) / (√κ + 1) )^n

	"""
		cg_required_conditioning(error::Real, n::Integer)

		error = relative error, n = iterations

		Derivation:
		let a = √κ
		error ≤ 2a ⋅ ( (a - 1) / (a + 1) )^n
		error / 2 ≤ ( a^1/n ⋅ (a - 1) / (a + 1) )^n
		which has the form x ≤ y^n. Assuming that x < 1 and y < 1
		gives x^(1/n) ≥ y

		(error / 2)^1/n ≥ a^1/n ⋅ (a - 1) / (a + 1)
		b ≥ a^1/n ⋅ (a - 1) / (a + 1)
		b ⋅ (a + 1) ≥ a^1/n ⋅ (a - 1)
		b ≥ a^1/n ⋅ (a - 1) - b⋅a
		b ≥ a^(n+1)/n - b⋅a - a^1/n
		b ≥ (a^1/n)^(n+1) - b⋅(a^1/n)^n - (a^1/n)^1
		0 ≥ c^(n+1) - b⋅c^n - c - b
		solve for c
		TODO: change variable names
	"""
	function cg_required_conditioning(error::Real, n::Integer; max_conditioning::Real=1000.0)
		b = (0.5error)^(1.0/n)
		polynomial(x) = begin
			x_pow_n = x^n
			x * x_pow_n - b * x_pow_n - x - b
		end
		sqrt_n_half_κ = Roots.find_zero(polynomial, (1.0, max_conditioning^(1/2n)), Roots.Bisection())
		κ = sqrt_n_half_κ^(2n)
		return κ
	end

end

