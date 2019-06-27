
using AssignmentSolver, Test, SparseArrays, DelimitedFiles

########################################
##Test standard worst case (for hungarian algorithm)
########################################

nrow = 8
ncol = 9
cost = [ii * jj for ii in 1:nrow, jj in 1:ncol]
reward = Float64.(reward2cost(cost))

costPad = pad_matrix(cost, maximum(cost))
rewardSp = sparse(reward)
rewardSpF = pad_matrix(rewardSp)

solfull = reverse(collect(1:nrow))
solsq = copy(solfull)
solpad = copy(solfull)

objfull = compute_objective(solfull, reward)
objsq = compute_objective(solsq, reward)
objpad = compute_objective(solfull, rewardSpF)

@testset "Basic" begin
    @testset "Full" begin
        @testset "Low-level" begin
            @test compute_objective(hungarian_assignment(cost), reward) == objfull
            @test compute_objective(auction_assignment_as(reward)[1], reward) == objfull
            @test compute_objective(auction_assignment_asfr1(reward)[1], reward) == objfull
            @test compute_objective(auction_assignment_asfr2(reward)[1], reward) == objfull
        end

        @testset "Wrapper" begin
            @test compute_objective(auction_assignment(reward, algorithm = "as")[1], reward) == objfull
            @test compute_objective(auction_assignment(reward, algorithm = "asfr1")[1], reward) == objfull
            @test compute_objective(auction_assignment(reward, algorithm = "asfr2")[1], reward) == objfull
        end

        @testset "Square" begin
            @test compute_objective(hungarian_assignment(cost[1:nrow, 1:nrow]), reward) == objsq
            @test compute_objective(auction_assignment(reward[1:nrow, 1:nrow], algorithm = "syfr")[1], reward) == objsq
            @test compute_objective(auction_assignment_syfr(reward[1:nrow, 1:nrow])[1], reward) == objsq
        end
    end
    @testset "Padded" begin
        @testset "Padded Input" begin
            @test compute_objective(hungarian_assignment(costPad), reward) == objpad
            @test compute_objective(auction_assignment_as(rewardSpF)[1], reward) == objpad
            @test compute_objective(auction_assignment_asfr1(rewardSpF)[1], reward) == objpad
            @test compute_objective(auction_assignment_asfr2(rewardSpF)[1], reward) == objpad
        end
        
        @testset "Implicit Padded" begin
            @testset "Low-level" begin
                @test compute_objective(auction_assignment_padas(rewardSp)[1], reward) == objpad
                @test compute_objective(auction_assignment_padasfr1(rewardSp)[1], reward) == objpad
                @test compute_objective(auction_assignment_padasfr2(rewardSp)[1], reward) == objpad
            end

            @testset "Wrapper" begin
                @test compute_objective(auction_assignment(rewardSp, algorithm = "as")[1], reward) == objpad
                @test compute_objective(auction_assignment(rewardSp, algorithm = "asfr1")[1], reward) == objpad
                @test compute_objective(auction_assignment(rewardSp, algorithm = "asfr2")[1], reward) == objpad
            end
        end
    end
end

########################################
##Test Example 1
########################################

cost = readdlm(joinpath(@__DIR__, "example1_costs.txt"))
nrow = size(cost, 1)
reward = reward2cost(cost)

costPad = pad_matrix(cost, maximum(cost))
rewardSp = sparse(reward)
rewardSpF = pad_matrix(rewardSp)

solfull = vec(readdlm(joinpath(@__DIR__, "example1_solfull.txt"), Int))
solsq = vec(readdlm(joinpath(@__DIR__, "example1_solsq.txt"), Int))
solpad = vec(readdlm(joinpath(@__DIR__, "example1_solpad.txt"), Int))

objfull = compute_objective(solfull, reward)
objsq = compute_objective(solsq, reward)
objpad = compute_objective(solfull, rewardSpF)

@testset "Example 1" begin
    @testset "Full" begin
        @testset "Low-level" begin
            @test compute_objective(hungarian_assignment(cost), reward) == objfull
            @test compute_objective(auction_assignment_as(reward)[1], reward) == objfull
            @test compute_objective(auction_assignment_asfr1(reward)[1], reward) == objfull
            @test compute_objective(auction_assignment_asfr2(reward)[1], reward) == objfull
        end

        @testset "Wrapper" begin
            @test compute_objective(auction_assignment(reward, algorithm = "as")[1], reward) == objfull
            @test compute_objective(auction_assignment(reward, algorithm = "asfr1")[1], reward) == objfull
            @test compute_objective(auction_assignment(reward, algorithm = "asfr2")[1], reward) == objfull
        end

        @testset "Square" begin
            @test compute_objective(hungarian_assignment(cost[1:nrow, 1:nrow]), reward) == objsq
            @test compute_objective(auction_assignment(reward[1:nrow, 1:nrow], algorithm = "syfr")[1], reward) == objsq
            @test compute_objective(auction_assignment_syfr(reward[1:nrow, 1:nrow])[1], reward) == objsq
        end
    end
    @testset "Padded" begin
        @testset "Padded Input" begin
            @test compute_objective(hungarian_assignment(costPad), reward) == objpad
            @test compute_objective(auction_assignment_as(rewardSpF)[1], reward) == objpad
            @test compute_objective(auction_assignment_asfr1(rewardSpF)[1], reward) == objpad
            @test compute_objective(auction_assignment_asfr2(rewardSpF)[1], reward) == objpad
        end
        
        @testset "Implicit Padded" begin
            @testset "Low-level" begin
                @test compute_objective(auction_assignment_padas(rewardSp)[1], reward) == objpad
                @test compute_objective(auction_assignment_padasfr1(rewardSp)[1], reward) == objpad
                @test compute_objective(auction_assignment_padasfr2(rewardSp)[1], reward) == objpad
            end

            @testset "Wrapper" begin
                @test compute_objective(auction_assignment(rewardSp, algorithm = "as")[1], reward) == objpad
                @test compute_objective(auction_assignment(rewardSp, algorithm = "asfr1")[1], reward) == objpad
                @test compute_objective(auction_assignment(rewardSp, algorithm = "asfr2")[1], reward) == objpad
            end
        end
    end
end

########################################
##Test Example 2
########################################

cost = readdlm(joinpath(@__DIR__, "example2_costs.txt"))
nrow = size(cost, 1)
reward = reward2cost(cost)
tol = minimum(sort(unique(reward))[2:end] - sort(unique(reward))[1:end-1]) / nrow

costPad = pad_matrix(cost, maximum(cost))
rewardSp = sparse(reward)
rewardSpF = pad_matrix(rewardSp)

solfull = vec(readdlm(joinpath(@__DIR__, "example2_solfull.txt"), Int))
solsq = vec(readdlm(joinpath(@__DIR__, "example2_solsq.txt"), Int))
solpad = vec(readdlm(joinpath(@__DIR__, "example2_solpad.txt"), Int))

objfull = compute_objective(solfull, reward)
objsq = compute_objective(solsq, reward)
objpad = compute_objective(solfull, rewardSpF)

@testset "Example 2" begin
    @testset "Full" begin
        @testset "Low-level" begin
            @test compute_objective(hungarian_assignment(cost), reward) == objfull
            @test isapprox(compute_objective(auction_assignment_as(reward)[1], reward), objfull)
            @test isapprox(compute_objective(auction_assignment_asfr1(reward)[1], reward), objfull)
            @test isapprox(compute_objective(auction_assignment_asfr2(reward)[1], reward), objfull)
        end

        @testset "Wrapper" begin
            @test isapprox(compute_objective(auction_assignment(reward, algorithm = "as")[1], reward), objfull)
            @test isapprox(compute_objective(auction_assignment(reward, algorithm = "asfr1")[1], reward), objfull)
            @test isapprox(compute_objective(auction_assignment(reward, algorithm = "asfr2")[1], reward), objfull)
        end

        @testset "Square" begin
            @test isapprox(compute_objective(hungarian_assignment(cost[1:nrow, 1:nrow]), reward), objsq)
            @test isapprox(compute_objective(auction_assignment(reward[1:nrow, 1:nrow], algorithm = "syfr")[1], reward), objsq)
            @test isapprox(compute_objective(auction_assignment_syfr(reward[1:nrow, 1:nrow])[1], reward), objsq)
        end
    end
    @testset "Padded" begin
        @testset "Padded Input" begin
            @test compute_objective(hungarian_assignment(costPad), reward) == objpad
            @test compute_objective(auction_assignment_as(rewardSpF)[1], reward) == objpad
            @test compute_objective(auction_assignment_asfr1(rewardSpF)[1], reward) == objpad
            @test compute_objective(auction_assignment_asfr2(rewardSpF)[1], reward) == objpad
        end
        
        @testset "Implicit Padded" begin
            @testset "Low-level" begin
                @test compute_objective(auction_assignment_padas(rewardSp)[1], reward) == objpad
                @test compute_objective(auction_assignment_padasfr1(rewardSp)[1], reward) == objpad
                @test compute_objective(auction_assignment_padasfr2(rewardSp)[1], reward) == objpad
            end

            @testset "Wrapper" begin
                @test isapprox(compute_objective(auction_assignment(rewardSp, algorithm = "as")[1], reward), objpad)
                @test isapprox(compute_objective(auction_assignment(rewardSp, algorithm = "asfr1")[1], reward), objpad)
                @test isapprox(compute_objective(auction_assignment(rewardSp, algorithm = "asfr2")[1], reward), objpad)
            end
        end
    end
end

########################################
##Test Example 3
########################################

cost = readdlm(joinpath(@__DIR__, "example3_costs.txt"))
nrow = size(cost, 1)
reward = reward2cost(cost)
tol = minimum(sort(unique(reward))[2:end] - sort(unique(reward))[1:end-1]) / nrow

costPad = pad_matrix(cost, maximum(cost))
rewardSp = sparse(reward)
rewardSpF = pad_matrix(rewardSp)

solfull = vec(readdlm(joinpath(@__DIR__, "example3_solfull.txt"), Int))
solsq = vec(readdlm(joinpath(@__DIR__, "example3_solsq.txt"), Int))
solpad = vec(readdlm(joinpath(@__DIR__, "example3_solpad.txt"), Int))

objfull = compute_objective(solfull, reward)
objsq = compute_objective(solsq, reward)
objpad = compute_objective(solfull, rewardSpF)

@testset "Example 3" begin
    @testset "Full" begin
        @testset "Low-level" begin
            @test compute_objective(hungarian_assignment(cost), reward) == objfull
            @test isapprox(compute_objective(auction_assignment_as(reward)[1], reward), objfull)
            @test isapprox(compute_objective(auction_assignment_asfr1(reward)[1], reward), objfull)
            @test isapprox(compute_objective(auction_assignment_asfr2(reward)[1], reward), objfull)
        end

        @testset "Wrapper" begin
            @test isapprox(compute_objective(auction_assignment(reward, algorithm = "as")[1], reward), objfull)
            @test isapprox(compute_objective(auction_assignment(reward, algorithm = "asfr1")[1], reward), objfull)
            @test isapprox(compute_objective(auction_assignment(reward, algorithm = "asfr2")[1], reward), objfull)
        end

        @testset "Square" begin
            @test isapprox(compute_objective(hungarian_assignment(cost[1:nrow, 1:nrow]), reward), objsq)
            @test isapprox(compute_objective(auction_assignment(reward[1:nrow, 1:nrow], algorithm = "syfr")[1], reward), objsq)
            @test isapprox(compute_objective(auction_assignment_syfr(reward[1:nrow, 1:nrow])[1], reward), objsq)
        end
    end
    @testset "Padded" begin
        @testset "Padded Input" begin
            @test compute_objective(hungarian_assignment(costPad), reward) == objpad
            @test compute_objective(auction_assignment_as(rewardSpF)[1], reward) == objpad
            @test compute_objective(auction_assignment_asfr1(rewardSpF)[1], reward) == objpad
            @test compute_objective(auction_assignment_asfr2(rewardSpF)[1], reward) == objpad
        end
        
        @testset "Implicit Padded" begin
            @testset "Low-level" begin
                @test compute_objective(auction_assignment_padas(rewardSp)[1], reward) == objpad
                @test compute_objective(auction_assignment_padasfr1(rewardSp)[1], reward) == objpad
                @test compute_objective(auction_assignment_padasfr2(rewardSp)[1], reward) == objpad
            end

            @testset "Wrapper" begin
                @test isapprox(compute_objective(auction_assignment(rewardSp, algorithm = "as")[1], reward), objpad)
                @test isapprox(compute_objective(auction_assignment(rewardSp, algorithm = "asfr1")[1], reward), objpad)
                @test isapprox(compute_objective(auction_assignment(rewardSp, algorithm = "asfr2")[1], reward), objpad)
            end
        end
    end
end
