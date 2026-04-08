/*
 * ============================================================================
 * FluidX3D Setup: Archimedes Spiral Wind Turbine — TEST CASE  v3
 * ============================================================================
 *
 *   Wind Tunnel : 8.0 m (x) x 3.0 m (y) x 3.0 m (z)
 *   Inlet Speed : 5.0 m/s in the -x direction
 *   Turbine D   : 1.0 m, axis along +x (STL file: arch.stl, in mm)
 *   TSR         : 2.5
 *   Resolution  : Ny = 240  (~80 voxels per rotor diameter)
 *   Rotation    : Clockwise when viewed from +x
 *
 * How to use
 * ----------
 *   1. Copy this file over  FluidX3D/src/setup.cpp
 *   2. Place  arch.stl  inside  FluidX3D/stl/
 *   3. In  FluidX3D/src/defines.hpp  ensure:
 *        #define FORCE_FIELD
 *        #define MOVING_BOUNDARIES
 *   4. Build & run
 *
 * v3 changes (radical restructure):
 *   - Force measurement decoupled from revox cadence:
 *       revox every 5°, forces measured every 60° (6× per revolution)
 *       This cuts GPU→CPU transfers from 72× to 6× per rev → ~100 MLUPS
 *   - tau floor raised to 0.51 (Re_eff ≈ 2,400) for ~1.6 voxel boundary
 *     layers at 80 voxels/D.  Previous tau=0.503 gave sub-voxel BL.
 *   - Redundant flags.read_from_device() eliminated from set_wall_velocities
 *   - set_wall_velocities no longer reads flags; caller must ensure
 *     flags are already on CPU
 *
 * Author : David Isaac
 * Date   : 2026-04-07
 * ============================================================================
 */

#include "setup.hpp"
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// ============================================================================
//  1.  PHYSICAL & NUMERICAL PARAMETERS
// ============================================================================

static constexpr float SI_RHO   = 1.204f;       // air density           [kg/m³]
static constexpr float SI_NU    = 1.516e-5f;     // kinematic viscosity   [m²/s]

static constexpr float TUNNEL_LX = 8.0f;         // streamwise            [m]
static constexpr float TUNNEL_LY = 3.0f;         // vertical              [m]
static constexpr float TUNNEL_LZ = 3.0f;         // spanwise              [m]

static constexpr float ROTOR_D      = 1.0f;      // rotor diameter        [m]
static constexpr float ROTOR_R      = ROTOR_D * 0.5f;
static constexpr float STL_DIAM_MM  = 914.233f;  // Y-Z diameter in thickened STL [mm]

static constexpr float SI_U  = 5.0f;             // freestream velocity   [m/s]
static constexpr float TSR   = 2.5f;             // tip-speed ratio

// Resolution — change this for grid convergence study:
//   Ny=360  → 1.4 voxels/blade,  124M cells, ~19 GB VRAM (RTX 3090)
//   Ny=480  → 1.9 voxels/blade,  295M cells, ~44 GB VRAM (A100 40GB)
//   Ny=600  → 2.4 voxels/blade,  576M cells, ~86 GB VRAM (A100 80GB)
static constexpr uint  NY    = 480u;             // voxels across tunnel height
static constexpr float U_LB  = 0.1f;            // lattice reference velocity

static constexpr float CONV_TOL      = 0.02f;    // 2% convergence threshold
static constexpr uint  MIN_CONV_REVS = 10u;
static constexpr uint  SAMPLE_REVS   = 5u;
static constexpr uint  MAX_REVS      = 50u;

static constexpr float UPSTREAM_D    = 3.0f;     // turbine dist from inlet [D]

// ============================================================================
//  2.  main_setup()
// ============================================================================
void main_setup() {

    const float PI = 3.14159265358979f;

    // ----------------------------------------------------------------
    //  Derived quantities
    // ----------------------------------------------------------------
    const float dx = TUNNEL_LY / (float)NY;
    const uint  Nx = (uint)std::round(TUNNEL_LX / dx);
    const uint  Ny = NY;
    const uint  Nz = (uint)std::round(TUNNEL_LZ / dx);

    const float dt         = U_LB * dx / SI_U;
    const float nu_lb_phys = SI_NU * dt / (dx * dx);
    const float tau_min    = 0.51f;                                 // Re_eff ≈ 2,400
    const float nu_lb_min  = (tau_min - 0.5f) / 3.0f;
    const float nu_lb      = fmax(nu_lb_phys, nu_lb_min);
    const float tau        = 3.0f * nu_lb + 0.5f;

    const float omega_si = TSR * SI_U / ROTOR_R;                   // [rad/s]
    const float omega_lb = omega_si * dt;                           // [rad/step]

    const float C_F = SI_RHO * (dx*dx*dx*dx) / (dt*dt);            // force  [N]
    const float C_T = C_F * dx;                                     // torque [N·m]

    const uint  steps_per_rev = (uint)std::round(2.0f * PI / omega_lb);

    // Revox cadence: 5° rotation per re-voxelisation
    const float deg_per_revox  = 5.0f;
    const uint  revox_interval = max(1u,
        (uint)std::round(deg_per_revox * PI / 180.0f / omega_lb));

    // Force measurement cadence: every 60° (= every 12 revox steps)
    //   6 measurements per revolution, giving good angular sampling
    //   while cutting GPU→CPU transfers by 12×
    const float deg_per_force  = 60.0f;
    const uint  force_every_n  = max(1u,
        (uint)std::round(deg_per_force / deg_per_revox));

    // Turbine placement
    const float x_turb_f = (float)Nx - 1.0f - UPSTREAM_D * ROTOR_D / dx;
    const uint  x_turb   = (uint)std::round(x_turb_f);
    const uint  y_turb   = Ny / 2u;
    const uint  z_turb   = Nz / 2u;
    const float3 turb_center = float3((float)x_turb, (float)y_turb, (float)z_turb);

    // Reference quantities
    const float A_swept   = PI * ROTOR_R * ROTOR_R;
    const float q_A       = 0.5f * SI_RHO * A_swept * SI_U * SI_U; // [N]
    const float P_wind    = q_A * SI_U;                              // [W]
    const float Re_D      = SI_U * ROTOR_D / SI_NU;
    const float D_lattice = ROTOR_D / dx;
    const float Re_eff    = U_LB * D_lattice / nu_lb;
    const float blockage  = A_swept / (TUNNEL_LY * TUNNEL_LZ) * 100.0f;

    // ----------------------------------------------------------------
    //  Banner
    // ----------------------------------------------------------------
    print_info("============================================================");
    print_info("  ARCHIMEDES WIND TURBINE  —  TEST CASE  v3");
    print_info("============================================================");
    print_info("  Tunnel       : " + to_string(TUNNEL_LX) + "m x "
               + to_string(TUNNEL_LY) + "m x " + to_string(TUNNEL_LZ) + "m");
    print_info("  U_inlet      : " + to_string(SI_U) + " m/s  (-x)");
    print_info("  Rotor D      : " + to_string(ROTOR_D) + " m");
    print_info("  TSR          : " + to_string(TSR));
    print_info("  omega        : " + to_string(omega_si) + " rad/s");
    print_info("  Re_D (phys)  : " + to_string(Re_D));
    print_info("  Re_eff (LBM) : " + to_string(Re_eff));
    if (nu_lb > nu_lb_phys) {
        print_info("  *** tau floor: tau=" + to_string(tau)
                  + "  nu_lb=" + to_string(nu_lb)
                  + "  (physical: " + to_string(nu_lb_phys) + ") ***");
    }
    print_info("  Blockage     : " + to_string(blockage) + " %");
    print_info("------------------------------------------------------------");
    print_info("  Grid         : " + to_string(Nx) + "x" + to_string(Ny)
               + "x" + to_string(Nz) + " = "
               + to_string((ulong)Nx * Ny * Nz / 1000000.0f) + " M voxels");
    print_info("  dx           : " + to_string(dx * 1000.0f) + " mm");
    print_info("  dt           : " + to_string(dt * 1.0e6f) + " us");
    print_info("  u_lb         : " + to_string(U_LB)
               + "  (Ma ~ " + to_string(U_LB * 1.7321f) + ")");
    print_info("  nu_lb        : " + to_string(nu_lb));
    print_info("  tau          : " + to_string(tau));
    print_info("  Voxels / D   : " + to_string(D_lattice));
    print_info("  Steps / rev  : " + to_string(steps_per_rev));
    print_info("  Revox every  : " + to_string(revox_interval) + " steps  (~"
               + to_string(deg_per_revox) + " deg)");
    print_info("  Force every  : " + to_string(force_every_n) + " revox steps  (~"
               + to_string(deg_per_force) + " deg)");
    print_info("  Turbine at   : (" + to_string(x_turb) + ", "
               + to_string(y_turb) + ", " + to_string(z_turb) + ")");
    print_info("============================================================");

    // ================================================================
    //  3.  CREATE LBM & SET BOUNDARY CONDITIONS
    // ================================================================
    LBM lbm(Nx, Ny, Nz, nu_lb);

    const ulong N = lbm.get_N();
    for (ulong n = 0ull; n < N; n++) {
        uint x = 0u, y = 0u, z = 0u;
        lbm.coordinates(n, x, y, z);

        if (x == Nx - 1u) {
            lbm.flags[n] = TYPE_E;
            lbm.u.x[n] = -U_LB;  lbm.u.y[n] = 0.0f;  lbm.u.z[n] = 0.0f;
            lbm.rho[n] = 1.0f;
        }
        else if (x == 0u) {
            lbm.flags[n] = TYPE_E;
            lbm.u.x[n] = -U_LB;  lbm.u.y[n] = 0.0f;  lbm.u.z[n] = 0.0f;
            lbm.rho[n] = 1.0f;
        }
        else if (y == 0u || y == Ny - 1u || z == 0u || z == Nz - 1u) {
            lbm.flags[n] = TYPE_S;
        }
        else {
            lbm.u.x[n] = -U_LB;  lbm.u.y[n] = 0.0f;  lbm.u.z[n] = 0.0f;
            lbm.rho[n] = 1.0f;
        }
    }

    // ================================================================
    //  4.  LOAD TURBINE MESH
    // ================================================================
    const float scale_factor = D_lattice / STL_DIAM_MM;
    Mesh* turbine = read_stl(get_exe_path() + "../stl/arch_12mm.stl");
    turbine->scale(scale_factor);
    turbine->translate(turb_center - turbine->get_center());

    const float3 omega_vec = float3(+omega_lb, 0.0f, 0.0f);

    print_info("Turbine loaded: scale=" + to_string(scale_factor)
              + "  D_lat=" + to_string(D_lattice)
              + "  omega_lb=" + to_string(omega_lb));

    // ================================================================
    //  5.  HELPER: set wall velocities on turbine cells
    // ================================================================
    //  Assumes flags are ALREADY on CPU (caller must ensure this).
    //  omega = (+omega_lb, 0, 0) → u = omega × r = (0, -ω·rz, +ω·ry)

    auto set_wall_velocities = [&]() {
        for (ulong n = 0ull; n < N; n++) {
            if (lbm.flags[n] & TYPE_S) {
                uint cx = 0u, cy = 0u, cz = 0u;
                lbm.coordinates(n, cx, cy, cz);
                if (cx == 0u || cx == Nx - 1u ||
                    cy == 0u || cy == Ny - 1u ||
                    cz == 0u || cz == Nz - 1u) continue;
                const float ry = (float)cy - turb_center.y;
                const float rz = (float)cz - turb_center.z;
                lbm.u.x[n] =  0.0f;
                lbm.u.y[n] = -omega_lb * rz;
                lbm.u.z[n] =  omega_lb * ry;
            }
        }
        lbm.u.write_to_device();
        lbm.update_moving_boundaries();
    };

    // ================================================================
    //  6.  HELPER: compute forces on turbine (CPU loop, NaN-filtered)
    // ================================================================
    //  Returns {torque_x_SI, thrust_x_SI}

    auto compute_forces = [&]() -> std::pair<float, float> {
        lbm.update_force_field();
        lbm.F.read_from_device();
        lbm.flags.read_from_device();

        float3 F_sum = float3(0.0f, 0.0f, 0.0f);
        float3 T_sum = float3(0.0f, 0.0f, 0.0f);

        for (ulong n = 0ull; n < N; n++) {
            if (!(lbm.flags[n] & TYPE_S)) continue;
            uint cx = 0u, cy = 0u, cz = 0u;
            lbm.coordinates(n, cx, cy, cz);
            if (cx == 0u || cx == Nx - 1u ||
                cy == 0u || cy == Ny - 1u ||
                cz == 0u || cz == Nz - 1u) continue;

            const float fx = lbm.F.x[n];
            const float fy = lbm.F.y[n];
            const float fz = lbm.F.z[n];
            if (std::isnan(fx) || std::isnan(fy) || std::isnan(fz) ||
                std::isinf(fx) || std::isinf(fy) || std::isinf(fz)) continue;

            F_sum.x += fx;  F_sum.y += fy;  F_sum.z += fz;

            const float rx = (float)cx - turb_center.x;
            const float ry = (float)cy - turb_center.y;
            const float rz = (float)cz - turb_center.z;
            T_sum.x += ry * fz - rz * fy;
            T_sum.y += rz * fx - rx * fz;
            T_sum.z += rx * fy - ry * fx;
        }

        return { T_sum.x * C_T, F_sum.x * C_F };
    };

    // ================================================================
    //  7.  BUFFERS
    // ================================================================
    std::vector<float> rev_torque, rev_thrust, rev_Cp;
    std::vector<float> curr_torques, curr_thrusts;

    bool   converged        = false;
    uint   convergence_rev  = 0u;
    float  convergence_time = 0.0f;
    uint   total_revs       = 0u;

    std::ofstream csv(get_exe_path() + "../data/test_timeseries.csv");
    csv << "step,time_s,angle_deg,torque_Nm,thrust_N,Cp_inst,Cd_inst\n";

    // ================================================================
    //  8.  INIT: transfer IC to GPU, then voxelise
    // ================================================================
    lbm.run(0u);

    // Read flags once for initial voxelisation + wall velocity setup
    lbm.voxelize_mesh_on_device(turbine, TYPE_S, turb_center, float3(0.0f), omega_vec);
    lbm.flags.read_from_device();
    set_wall_velocities();

    // Sanity check
    {
        float max_uy = 0.0f, max_uz = 0.0f;
        ulong n_turb = 0ull;
        for (ulong n = 0ull; n < N; n++) {
            if (lbm.flags[n] & TYPE_S) {
                uint cx = 0u, cy = 0u, cz = 0u;
                lbm.coordinates(n, cx, cy, cz);
                if (cx == 0u || cx == Nx - 1u ||
                    cy == 0u || cy == Ny - 1u ||
                    cz == 0u || cz == Nz - 1u) continue;
                n_turb++;
                max_uy = fmax(max_uy, fabs(lbm.u.y[n]));
                max_uz = fmax(max_uz, fabs(lbm.u.z[n]));
            }
        }
        print_info("Voxelised: " + to_string(n_turb) + " turbine cells"
                  + "  |u.y|_max=" + to_string(max_uy)
                  + "  |u.z|_max=" + to_string(max_uz)
                  + "  tip~" + to_string(omega_lb * D_lattice * 0.5f));
    }

    // ================================================================
    //  9.  SIMULATION LOOP
    // ================================================================
    //
    //  Architecture:
    //    OUTER:  loop until convergence or max_revs
    //    INNER:  revoxelise every 5°, run LBM steps
    //    FORCE:  compute forces only every 60° (every 12 revox steps)
    //    REPORT: once per revolution — average, convergence, status
    //
    //  This decoupling is critical for performance.  The CPU force loop
    //  reads ~500 MB from GPU each call.  At 6× per rev (not 72×),
    //  the overhead drops from ~35 GB/rev to ~3 GB/rev.

    const uint   max_steps = MAX_REVS * steps_per_rev;
    uint         step = 0u;
    float        accum_angle = 0.0f;
    uint         revox_count = 0u;   // counts revox steps within current rev

    print_info("Starting simulation  (max " + to_string(MAX_REVS) + " revolutions)");

    Clock clock;
    clock.start();

    while (step < max_steps) {

        // ----------------------------------------------------------
        //  9a.  Rotate mesh & re-voxelise (every 5°)
        // ----------------------------------------------------------
        const float delta_angle = omega_lb * (float)revox_interval;
        accum_angle += delta_angle;

        lbm.unvoxelize_mesh_on_device(turbine, TYPE_S);
        turbine->rotate(float3x3(float3(1.0f, 0.0f, 0.0f), -delta_angle));
        lbm.voxelize_mesh_on_device(turbine, TYPE_S, turb_center, float3(0.0f), omega_vec);

        // Read flags once (needed by both set_wall_velocities and force loop)
        lbm.flags.read_from_device();
        set_wall_velocities();

        // ----------------------------------------------------------
        //  9b.  Advance LBM
        // ----------------------------------------------------------
        lbm.run(revox_interval);
        step += revox_interval;
        revox_count++;

        // ----------------------------------------------------------
        //  9c.  Force measurement (every 60° = every force_every_n revox steps)
        // ----------------------------------------------------------
        if (revox_count % force_every_n == 0u) {
            auto [torque_si, thrust_si] = compute_forces();
            const float t_phys  = (float)step * dt;
            const float Cp_inst = (torque_si * omega_si) / P_wind;
            const float Cd_inst = fabs(thrust_si) / q_A;

            curr_torques.push_back(torque_si);
            curr_thrusts.push_back(thrust_si);

            csv << step << "," << t_phys << ","
                << accum_angle * (180.0f / PI) << ","
                << torque_si << "," << thrust_si << ","
                << Cp_inst << "," << Cd_inst << "\n";
        }

        // ----------------------------------------------------------
        //  9d.  Revolution boundary
        // ----------------------------------------------------------
        const float revs_done = accum_angle / (2.0f * PI);
        if ((uint)revs_done > total_revs) {
            total_revs = (uint)revs_done;
            revox_count = 0u;

            // Skip if no force samples yet (shouldn't happen, but guard)
            if (curr_torques.empty()) continue;

            // Revolution average
            float sum_T = 0.0f, sum_F = 0.0f;
            for (size_t i = 0; i < curr_torques.size(); i++) {
                sum_T += curr_torques[i];
                sum_F += curr_thrusts[i];
            }
            const float avg_T = sum_T / (float)curr_torques.size();
            const float avg_F = sum_F / (float)curr_thrusts.size();
            const uint  n_samples = (uint)curr_torques.size();
            curr_torques.clear();
            curr_thrusts.clear();

            rev_torque.push_back(avg_T);
            rev_thrust.push_back(avg_F);
            rev_Cp.push_back((avg_T * omega_si) / P_wind);

            // Convergence
            float conv_change = -1.0f;
            if (!converged && total_revs >= MIN_CONV_REVS && rev_torque.size() >= 3u) {
                const size_t nn = rev_torque.size();
                const float d2 = fabs(rev_torque[nn - 2]);
                const float d3 = fabs(rev_torque[nn - 3]);
                if (d2 > 1.0e-12f && d3 > 1.0e-12f) {
                    const float c1 = fabs(rev_torque[nn-1] - rev_torque[nn-2]) / d2;
                    const float c2 = fabs(rev_torque[nn-2] - rev_torque[nn-3]) / d3;
                    conv_change = fmax(c1, c2);
                    if (c1 < CONV_TOL && c2 < CONV_TOL) {
                        converged        = true;
                        convergence_rev  = total_revs;
                        convergence_time = (float)step * dt;
                    }
                }
            }

            // Status
            const double elapsed = clock.stop();
            const float progress = (float)step / (float)max_steps * 100.0f;
            const float mlups    = (float)step * (float)N / (float)elapsed / 1.0e6f;
            const float eta_s    = (float)elapsed / (float)step * (float)(max_steps - step);
            const int eta_min = (int)(eta_s / 60.0f), eta_sec = (int)(eta_s) % 60;
            const int el_min  = (int)(elapsed / 60.0),  el_sec  = (int)(elapsed) % 60;

            print_info("============================================================");
            print_info("  Rev " + to_string(total_revs) + " / " + to_string(MAX_REVS)
                      + "   |   Step " + to_string(step) + " / " + to_string(max_steps)
                      + "  (" + to_string((int)progress) + "%)"
                      + "   [" + to_string(n_samples) + " force samples]");
            print_info("  t = " + to_string((float)step * dt) + " s"
                      + "   angle = " + to_string(accum_angle * (180.0f / PI)) + " deg");
            print_info("------------------------------------------------------------");
            print_info("  Torque (avg) : " + to_string(avg_T) + " N·m");
            print_info("  Thrust (avg) : " + to_string(avg_F) + " N");
            print_info("  Power        : " + to_string(avg_T * omega_si) + " W");
            print_info("  Cp           : " + to_string(rev_Cp.back()));
            print_info("  Cd           : " + to_string(fabs(avg_F) / q_A));
            print_info("------------------------------------------------------------");

            if (converged) {
                if (total_revs == convergence_rev)
                    print_info("  >>> CONVERGED at rev " + to_string(convergence_rev)
                              + "  |  Sampling " + to_string(SAMPLE_REVS) + " more ...");
                else
                    print_info("  CONVERGED  |  Sampling " + to_string(total_revs - convergence_rev)
                              + " / " + to_string(SAMPLE_REVS));
            } else if (conv_change >= 0.0f) {
                print_info("  Conv: " + to_string(conv_change * 100.0f) + "%  (< "
                          + to_string(CONV_TOL * 100.0f) + "%)");
            } else {
                print_info("  Conv: waiting (min " + to_string(MIN_CONV_REVS) + " revs)");
            }

            print_info("  Perf: " + to_string(mlups) + " MLUPS"
                      + "   Elapsed: " + to_string(el_min) + "m " + to_string(el_sec) + "s"
                      + "   ETA: " + to_string(eta_min) + "m " + to_string(eta_sec) + "s");
            print_info("============================================================");

            if (converged && total_revs >= convergence_rev + SAMPLE_REVS) {
                print_info(">>> Sampling complete.  Stopping.");
                break;
            }
        }
    }

    csv.close();

    // ================================================================
    //  10.  POST-PROCESS
    // ================================================================
    const uint use_revs = converged
        ? min(SAMPLE_REVS, (uint)rev_torque.size())
        : min(5u, (uint)rev_torque.size());

    float final_torque = 0.0f, final_thrust = 0.0f;
    for (uint i = 0; i < use_revs; i++) {
        final_torque += rev_torque[rev_torque.size() - 1 - i];
        final_thrust += rev_thrust[rev_thrust.size() - 1 - i];
    }
    final_torque /= (float)use_revs;
    final_thrust /= (float)use_revs;

    const float power    = final_torque * omega_si;
    const float Cp_final = power / P_wind;
    const float Cd_final = fabs(final_thrust) / q_A;
    const float wall_s   = clock.stop();

    print_info("============================================================");
    print_info("  RESULTS  —  TEST CASE");
    print_info("============================================================");
    print_info("  Avg  Torque  : " + to_string(final_torque) + " N·m");
    print_info("  Avg  Thrust  : " + to_string(final_thrust) + " N");
    print_info("  Power output : " + to_string(power) + " W");
    print_info("  Cp           : " + to_string(Cp_final));
    print_info("  Cd           : " + to_string(Cd_final));
    print_info("------------------------------------------------------------");
    print_info("  Converged    : " + string(converged ? "YES" : "NO"));
    if (converged) {
        print_info("    at rev     : " + to_string(convergence_rev));
        print_info("    at time    : " + to_string(convergence_time) + " s");
    }
    print_info("  Total revs   : " + to_string(total_revs));
    print_info("  Total steps  : " + to_string(step));
    print_info("  Wall-clock   : " + to_string(wall_s) + " s  ("
              + to_string(wall_s / 60.0f) + " min)");
    print_info("============================================================");

    // Summary file
    {
        std::ofstream f(get_exe_path() + "../data/test_summary.txt");
        f << "ARCHIMEDES WIND TURBINE — TEST CASE v3\n";
        f << "=======================================\n\n";
        f << "Configuration\n";
        f << "  Tunnel        : " << TUNNEL_LX << " x " << TUNNEL_LY
                                  << " x " << TUNNEL_LZ << " m\n";
        f << "  U_inlet       : " << SI_U      << " m/s\n";
        f << "  Rotor D       : " << ROTOR_D   << " m\n";
        f << "  TSR           : " << TSR        << "\n";
        f << "  omega         : " << omega_si   << " rad/s\n";
        f << "  Re_D          : " << Re_D       << "\n";
        f << "  Re_eff        : " << Re_eff     << "\n";
        f << "  tau           : " << tau         << "\n";
        f << "  tau_min       : " << tau_min     << "\n";
        f << "  nu_lb         : " << nu_lb       << "\n";
        f << "  nu_lb (phys)  : " << nu_lb_phys  << "\n";
        f << "  Blockage      : " << blockage   << " %\n";
        f << "  Ny            : " << Ny         << "\n";
        f << "  Voxels/D      : " << D_lattice  << "\n";
        f << "  Grid          : " << Nx << "x" << Ny << "x" << Nz << "\n\n";
        f << "Results\n";
        f << "  Avg Torque    : " << final_torque << " N·m\n";
        f << "  Avg Thrust    : " << final_thrust << " N\n";
        f << "  Power         : " << power        << " W\n";
        f << "  Cp            : " << Cp_final     << "\n";
        f << "  Cd            : " << Cd_final     << "\n\n";
        f << "Convergence\n";
        f << "  Converged     : " << (converged ? "Yes" : "No") << "\n";
        f << "  Conv. rev     : " << convergence_rev  << "\n";
        f << "  Conv. time    : " << convergence_time << " s\n";
        f << "  Wall-clock    : " << wall_s << " s\n";
        f.close();
    }

    // Per-revolution CSV
    {
        std::ofstream f(get_exe_path() + "../data/test_rev_averages.csv");
        f << "revolution,avg_torque_Nm,avg_thrust_N,Cp,Cd\n";
        for (size_t i = 0; i < rev_torque.size(); i++) {
            f << (i + 1) << ","
              << rev_torque[i] << "," << rev_thrust[i] << ","
              << rev_Cp[i] << "," << fabs(rev_thrust[i]) / q_A << "\n";
        }
        f.close();
    }

    print_info("Data → data/test_timeseries.csv, test_rev_averages.csv, test_summary.txt");
    delete turbine;
}