# AssignmentSolver

Collection of algorithms for solving asymmetric linear sum assignment problems including several forward/reverse [Auction algorithms](https://en.wikipedia.org/wiki/Auction_algorithm), as well as an implementation of the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm).  The Auction algorithms are typically more efficient for sparse problems.

## Installation
```julia
pkg> add https://github.com/brendanstats/AssignmentSolver.jl
```

## Examples
```julia
using AssignmentSolver
cost = [ii * jj for ii in 1:8, jj in 1:9]
reward = Float64.(reward2cost(cost)) #convert cost matrix to a reward matrix or vice versa

#Find maximal assignment
auctionsol, lambda = auction_assignment(reward) #Find maximal assignment
compute_objective(auctionsol, cost) #Compute objective
auctionsol.r2c  #Report column each row is assigned to

#Find minimal assignment
hungariansol = hungarian_assignment(cost) #Find minimal assignment
compute_objective(hungariansol, cost) #Compute objective
hungariansol.r2c #Report column each row is assigned to
```
By default auction algorithms will find a maximal assignment while the Hungarian algorithm will find a minimal assignment.  The `reward2cost` function will convert a reward matrix to a cost matrix (or vice versa) allowing either class of algorithm to be used. See `test/runtests.jl` for additional examples.

## AssignmentState Type
Both classes of algorithm return an object of the `AssignmentState` type defined below.
```julia
mutable struct AssignmentState{G<:Integer, T<:Real}
    r2c::Array{G, 1} #Map row to assigned column, zero otherwise
    c2r::Array{G, 1} #Map column to assigned row, zero otherwise
    rowPrices::Array{T, 1} #Row dual variable
    colPrices::Array{T, 1} #Column dual variable
    nassigned::G #Number of assignments
    nrow::G #Number of rows
    ncol::G #Number of columns
    nexcesscols::G #ncol - nrow
    sym::Bool #nrow == ncol
end
```
Row assignments are stored in the `r2c` field.  The additional ``\lambda`` value returned by the auction algorithms is an internal parameter used in solving asymmetric assignment problems, see references for additional details.

# Sparse Assignment Problems
The provided Hungarian algorithm is implemented only for dense problems (any row can be assigned to any column).  In contrast auction algorithms are implemented to solve both dense and sparse problems (by supplying a sparse reward matrix). 

## Existsence of a Feasible Assignment
For sparse problems there is no guarantee that a feasible assignment, where each row is assigned to a different column, exists.  Methods exist to detect in feasibility in auction algorithms automatically (see references) but are not currently implemented.  A simple way to ensure that a feasible assignment exists is to add a number of dummy columns to the reward matrix so that every row is guaranteed to have a column to which it can be assigned.  This can be done explicitly with the `pad_matrix` function which will return a new reward matrix or implicitly setting `pad = true` within the auction algorithm.  
```julia
auction_assignment(pad_matrix(reward, dfltReward = 0.0))
auction_assignment(reward, pad = true, dfltReward = 0.0)
```
Setting `pad = true` does not generate a new reward matrix but instead implicitly adds possible assignments `row, ncol + row` with rewards of `dfltReward`.  An alternative interpretation of this approach is that rows may be left unassigned  and receive a reward of `dfltReward`. Setting `dfltReward` lower (negative with larger absolute value) encourages the algorithm to find an assignment using the arcs in the supplied reward matrix to be used.  Setting `dfltReward = -Inf` is equivalent to adding no padded (dummy) entries (setting `pad=false`).  


# Auction Algorithm Details

## Tuning Parameters and ``\epsilon-scaling``
Solutions to assignment problems found by an auction algorithm are only approximate, the objective value of the returned assignment will be within ``n * \epsilon`` of the optimal objective value, where ``n`` is the number of rows, for a given tolerance ``\epsilon``. If the rewards are integers then selecting ``\epsilon < 1/n`` guarantees that the assignment is optimal. For non-integer rewards optimality can be guaranteed by setting ``\epsilon < \delta/n`` where ``\delta`` is the smallest difference between non-identical values of the reward matrix. The complexity of auction algorithms is enhanced through the use of ``\epsilon-scaling``, solving the assignment problem repeatedly for successively smaller values of ``\epsilon``.  This processes can be controlled through several tuning parameters:
- `epsi0`: starting value for ``\epsilon``
- `epsitol`: final value for ``\epsilon``
- `epsiscale`: rate to shrink ``\epsilon``, ``\epsilon_t = \epsilon_{t - 1} * epsiscale``
- `dfltTwo`: Default second best reward, default value is `-Inf` but in principle any large (relative to other reward values) negative value should work if `Inf` value dual variables are problematic

See Bertsekas and Castañon (1992) for additional details.

## Forward / Reverse Algorithms
A specific auction algorithm can be specified by setting the `algorithm` parameter to one of "as", "asfr1", "asfr2" or "syfr" (e.g. `auction_assignment(reward, algorithm = "as")`).  The "syfr" algorithm is useful only for symmetric (equal number of rows and columns in the reward matrix) and cannot be used with padding, however it yields substantial runtime improvements for symmetric problems. The `auction_assignment` function is a wrapper which calls one of the following functions:
- `auction_assignment_as`
- `auction_assignment_asfr1`
- `auction_assignment_asfr2`
- `auction_assignment_syfr`
- `auction_assignment_padas`
- `auction_assignment_padasfr1`
- `auction_assignment_padasfr2`

Finer grained control is available by calling these functions directly. Definitions are consistent with those used in Bertsekas and Castañon (1992) with `pad` indicating a version including dummy entries to ensure feasibility.  In practice `asfr1` and `asfr2` display similar performance although `asfr1` is thought to be more efficient in some scenarios.  Performance for `as` is generally slower.

## Algorithm Complexity
The worst case computational complexity for the Hungarian algorithm of ``O(n^3)``, for a cost matrix with n rows, is well known.  For auction algorithms, with integer rewards, the worst case computational complexity is ``O(nA\log(nC/\epsilon))``, where A is the number of arcs in the underlying graph of the assignment problem, C is the maximum absolute value of the assignment rewards, and ``\epsilon`` is the tolerance to which the problem is solved.  While I am unaware of findings on the typical complexity for the Hungarian algorithm, other than that it is faster than ``O(n^3)``, the average case for auction algorithms appears to grow like in simulations ``O(A\log(n)\log(nC))``.  See references Bertsekas (1998), 7.1 for auction algorithm complexity results.


## References
1. Bertsekas, D.P. and Castañon, D.A., 1992. A forward/reverse auction algorithm for asymmetric assignment problems. Computational Optimization and Applications, 1(3), pp.277-297.
2. Bertsekas, D.P., 1998. Network optimization: continuous and discrete models (pp. 467-511). Belmont: Athena Scientific.
3. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html