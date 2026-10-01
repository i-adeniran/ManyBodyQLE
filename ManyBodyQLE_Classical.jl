# ========================================================
# The Crowded Proton Wire (Classical Zero-Flux Gridlock)
# Classical Overdamped Langevin Dynamics
# Includes Statistical Ensemble Averaging (50 Runs)
#
# This code is based on Jose Antonio Forne's MATLAB code
# for the Many Body QLE in:
# Fornés, J. A. Quantum Ratchets. In Principles of 
# Brownian and Molecular Motors; Fornés, J. A., Ed.; 
# Springer International Publishing: Cham, 2021; 
# pp 123–148. https://doi.org/10.1007/978-3-030-64957-9_8.
# ========================================================

using Plots
using LinearAlgebra
using Random
using Statistics

default(fontfamily="Arial_Bold")

function simulate_classical_wire()
  # Parameters
  screen_length     = 0.974  
  steric_core       = 0.137  
  A_coulomb         = 3.085  
  A_steric          = 0.841  

  # Classical Limit
  lambda_smearing   = 0.0 

  # Setup
  dt                = 1e-4   
  N_steps           = 30000  
  therm_steps       = 10000  # Discard initial thermalization
  N_protons         = 15     
  F_ext             = 1.5    

  noise_amp         = sqrt(2.0 * 1.0 * dt)
  
  N_runs            = 50
  fluxes            = zeros(Float64, N_runs)
  first_history     = zeros(Float64, N_steps, N_protons)

  for run in 1:N_runs
      Random.seed!(42 + run) 

      x                 = collect(range(0.0, step=0.2, length=N_protons))
      dx                = zeros(Float64, N_protons)
      safe_r            = zeros(Float64, N_protons, N_protons)
      total_force  = zeros(Float64, N_protons, N_protons)
      
      cm_therm = 0.0

      for step in 1:N_steps
        two_pi_x                                    = @. 2.0 * π * x
        four_pi_x                                   = @. 4.0 * π * x
        Dd1U                                        = @. -0.5 * (cos(two_pi_x) - 0.5 * cos(four_pi_x))

        bottleneck_x                                = 6.0
        
        safe_dist                                   = max.(1e-3, bottleneck_x .- x) 
        bottleneck_force                            = @. -12.0 * A_steric * (steric_core^12) * safe_dist / (safe_dist^2 + 1e-4)^7

        diff                                        = x .- x' 
        r                                           = abs.(diff)
        safe_r                                      .= r .+ I(N_protons) 
        sign_diff                                   = sign.(diff)

        yukawa_force                                = @. A_coulomb * exp(-safe_r / screen_length) * ((1.0 / screen_length) / safe_r + 1.0 / safe_r^2) * sign_diff
        
        gauss_exp                                   = @. exp(-0.5 * (safe_r / steric_core)^2)

        steric_classical_force                      = @. (A_steric * safe_r / (steric_core^2)) * gauss_exp * sign_diff

        total_force                                 .= steric_classical_force .+ yukawa_force

        total_force[diagind(total_force)]           .= 0.0 
        interaction_force                           = dropdims(sum(total_force, dims=2), dims=2)

        dx                                          .= @. (-Dd1U + interaction_force + bottleneck_force + F_ext) * dt + noise_amp * randn()
        dx                                          .= clamp.(dx, -0.05, 0.05)
        x                                           .+= dx
        
        if run == 1
          first_history[step, :] .= x
        end
        
        if step == therm_steps
          cm_therm = mean(x)
        end
      end
      
      cm_end      = mean(x)
      delta_t     = (N_steps - therm_steps) * dt
      fluxes[run] = (cm_end - cm_therm) / delta_t
  end

  mean_flux = mean(fluxes)
  std_flux  = std(fluxes)
  sem_flux  = std_flux / sqrt(N_runs)

  println("===================================")
  println("\nCLASSICAL STATISTICS (N = $N_runs)")
  println("===================================")
  println("Mean Flux: ", round(mean_flux, digits=4))
  println("Std Dev:   ", round(std_flux, digits=4))
  println("SEM:       ", round(sem_flux, digits=4))

  time_axis = (1:N_steps) .* dt
  p         = plot(
    time_axis, 
    first_history, 
    legend=false, 
    linewidth=1.5, 
    palette=:Reds_9, 
    bg_inside=:whitesmoke,
    xguidefontsize = 16, 
    yguidefontsize = 16, 
    xtickfontsize  = 14, 
    ytickfontsize  = 14, 
    legendfontsize = 8,  
    dpi=300
  )
  title!("Classical Limit: Zero-Flux Gridlock")
  xlabel!("Dimensionless Time (t)")
  ylabel!("Dimensionless Position (x)")
  
  stats_text = "Flux = $(round(mean_flux, digits=2)) ± $(round(sem_flux, digits=3))"
  annotate!(time_axis[15000], 8.5, text("Trapped at Steric Bottleneck\n" * stats_text, :red, 10, :center))
  hline!([6.0], color=:black, linestyle=:dash, linewidth=2, label="Bottleneck")

  plot!(size=(800, 600), dpi=300, margin=5Plots.mm)
  display(p)
  savefig(p, "classical_gridlock_stats.png")
end

simulate_classical_wire()
