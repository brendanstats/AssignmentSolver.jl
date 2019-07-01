module AssignmentSolver

using DataStructures: Queue, enqueue!, dequeue!
using SparseArrays

export AssignmentState,
    clear_assignment!,
    adjust_inf!,
    remove_padded!,
    flip,
    compute_objective,
    pad_matrix,
    reward2cost
export adjusted_cost,
    zero_cost
export hungarian_assignment,
    hungarian_assignment!,
    step3,
    step4,
    step5,
    step6!
export maxtwo_column,
    maxtwo_row,
    get_zeros,
    get_openrows,
    get_opencols,
    get_opencolsabove,
    get_nbelow_opencolsabove,
    findrowmax,
    findrowmin,
    findcolmax,
    findcolmin,
    dimmaximums,
    forward_rewardmatrix,
    scale_assignment!,
    min_assigned_colprice,
    check_epsilons,
    check_epsilon_slackness,
    tuple_r2c
export forward_bid,
    reverse_bid,
    forward_update!,
    forward_update_nbelow!,
    reverse_update!,
    reverse_update_nbelow!,
    forward_iteration!,
    forward_iteration_nbelow!,
    reverse_iteration!,
    reverse_iteration_nbelow!,
    scaling_as!,
    auction_assignment_as,
    auction_assignment_padas,
    scaling_asfr1!,
    auction_assignment_asfr1,
    auction_assignment_padasfr1,
    scaling_asfr2!,
    auction_assignment_asfr2,
    auction_assignment_padasfr2,
    scaling_syfr!,
    auction_assignment_syfr,
    auction_assignment

include("../src/assignmentstate.jl")
include("../src/hungarian_utils.jl")
include("../src/hungarian.jl")
include("../src/auction_utils.jl")
include("../src/auction_as.jl")

end
