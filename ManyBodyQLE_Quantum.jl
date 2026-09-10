# ===================================================
# The Crowded Proton Wire (Grotthuss Cascades)
# Quantum Spatial Smearing and Correlated Percolation
# ===================================================

using Plots
using LinearAlgebra
using Random

function simulate_biological_wire()
  # Parameters
  screen_length     = 0.974  
  steric_core       = 0.137  
  A_coulomb         = 3.085  
  A_steric          = 0.841  

  lambda_smearing   = 0.2 

  # Setup
  dt                = 1e-4   
  N_steps           = 30000  
  N_protons         = 15     
  F_ext             = 1.5    

  noise_amp         = sqrt(2.0 * 1.0 * dt)
  x                 = collect(range(0.0, step=0.2, length=N_protons))
  history           = zeros(Float64, N_steps, N_protons)

  dx                = zeros(Float64, N_protons)
  safe_r            = zeros(Float64, N_protons, N_protons)
  total_int_matrix  = zeros(Float64, N_protons, N_protons)

  Random.seed!(42) 

  for step in 1:N_steps
    two_pi_x                                    = @. 2.0 * π * x
    four_pi_x                                   = @. 4.0 * π * x
    Dd1U                                        = @. -0.5 * (cos(two_pi_x) - 0.5 * cos(four_pi_x))

    bottleneck_x                                = 6.0
    
    safe_dist                                   = max.(1e-3, bottleneck_x .- x) 
    
    bottleneck_force                            = @. -12.0 * A_steric * (steric_core^12) * safe_dist / (safe_dist^2 + lambda_smearing^2)^7

    diff                                        = x .- x' 
    r                                           = abs.(diff)
    safe_r                                      .= r .+ I(N_protons) 
    sign_diff                                   = sign.(diff)

    steric_force                                = @. 12.0 * A_steric * (steric_core^12 / safe_r^13) * sign_diff
    yukawa_force                                = @. A_coulomb * exp(-safe_r / screen_length) * ((1.0 / screen_length) / safe_r + 1.0 / safe_r^2) * sign_diff
    
    total_int_matrix                            .= steric_force .+ yukawa_force
    total_int_matrix[diagind(total_int_matrix)] .= 0.0 
    interaction_force                           = dropdims(sum(total_int_matrix, dims=2), dims=2)

    dx                                          .= @. (-Dd1U + interaction_force + bottleneck_force + F_ext) * dt + noise_amp * randn()
    dx                                          .= clamp.(dx, -0.05, 0.05)
    x                                           .+= dx
    history[step, :]                            .= x
  end

  time_axis = (1:N_steps) .* dt
  p         = plot(time_axis, history, legend=false, linewidth=1.5, palette=:Greens_9, bg_inside=:whitesmoke)
  title!("Quantum QLE: Correlated Percolation")
  xlabel!("Dimensionless Time (t)")
  ylabel!("Dimensionless Position (x)")
  annotate!(time_axis[15000], 12.0, text("Steady Macroscopic Flux", :darkgreen, 10, :center))
  hline!([6.0], color=:black, linestyle=:dash, linewidth=2, label="Bottleneck")

  plot!(size=(800, 600), dpi=300, margin=5Plots.mm)
  display(p)
  savefig(p, "quantum_percolation.png")
end

simulate_biological_wire()
