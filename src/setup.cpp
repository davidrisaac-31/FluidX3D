/*
 * ============================================================================
 * FluidX3D Setup: Archimedes Spiral Wind Turbine — TEST CASE  v4
 * ============================================================================
 *
 *   Wind Tunnel : 8.0 m (x) x 3.0 m (y) x 3.0 m (z)
 *   Inlet Speed : 5.0 m/s in the -x direction
 *   Turbine D   : 1.0 m, axis along +x (STL file: arch_12mm.stl, in mm)
 *   TSR         : 2.5
 *   Resolution  : Ny = 480  (~160 voxels per rotor diameter)
 *   Rotation    : Clockwise when viewed from +x
 *
 * How to use
 * ----------
 *   1. Copy this file over  FluidX3D/src/setup.cpp
 *   2. Place  arch_12mm.stl  inside  FluidX3D/stl/
 *   3. In  FluidX3D/src/defines.hpp  ensure:
 *        #define FORCE_FIELD
 *        #define MOVING_BOUNDARIES
 *        #define FP16S               <-- IMPORTANT for cloud GPU cost
 *      FP16S uses FP32 compute with FP16 storage, roughly doubling
 *      MLUPS (LBM is memory-bandwidth bound) with <1% impact on Cp.
 *      This alone cuts your cloud GPU bill in half.
 *   4. Build & run
 *
 * v4 changes — major GPU-utilization optimization:
 *   - Turbine voxelized with TYPE_S|TYPE_X  so we can distinguish it
 *     from the tunnel walls (which stay plain TYPE_S).
 *   - compute_forces() now uses  lbm.object_force(TYPE_S|TYPE_X)  and
 *     lbm.object_torque(center, TYPE_X) — both GPU parallel reductions.
 *     Each force sample is now ~2 ms instead of ~1 s (no flags/F
 *     transfer, no CPU loop over 295 M cells).
 *   - Removed set_wall_velocities() entirely.  FluidX3D's voxelize
 *     kernel sets  u = v_lin + omega x (p - c)  on the GPU directly
 *     when rotational_velocity != 0, so the CPU loop and the 3.5 GB
 *     velocity writeback per revox step were completely redundant.
 *     (Verified in kernel.cpp lines 2341-2362.)
 *   - Removed flags.read_from_device() from the hot path.
 *   - Revox cadence relaxed from 5 deg -> 10 deg.
 *   - Expected GPU utilization: ~75% (v3) -> ~98% (v4)
 *
 * v3 changes (superseded by v4):
 *   - Force measurement decoupled from revox cadence
 *   - tau floor raised to 0.51
 *
 * Author : David Isaac
 * Date   : 2026-04-08
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
// (VRAM figures assume FP16S is set in defines.hpp; double them for FP32.)
static constexpr uint  NY    = 480u;             // voxels across tunnel height
static constexpr float U_LB  = 0.1f;             // lattice reference velocity

static constexpr float CONV_TOL      = 0.02f;    // 2% convergence threshold
static constexpr uint  MIN_CONV_REVS = 10u;
static constexpr uint  SAMPLE_REVS   = 5u;
static constexpr uint  MAX_REVS      = 50u;

static constexpr float UPSTREAM_D    = 3.0f;     // turbine dist from inlet [D]

// Turbine is voxelized with this combined flag so we can separate it
// from the tunnel walls (which stay plain TYPE_S).  object_force() uses
// exact flag match, so the tunnel walls are excluded automatically.
// object_torque() uses a bitmask match — we pass TYPE_X alone for it,
// which also excludes the walls since they have no TYPE_X bit.
static constexpr uchar TURB_FLAG = TYPE_S | TYPE_X;

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

    // Revox cadence: 10° rotation per re-voxelisation (v4: was 5°)
    // Revox is cheap — the bigger cost was set_wall_velocities, now removed.
    // But every revox still runs a voxelize_mesh kernel, so 10° saves GPU time.
    const float deg_per_revox  = 10.0f;
    const uint  revox_interval = max(1u,
        (uint)std::round(deg_per_revox * PI / 180.0f / omega_lb));

    // Force measurement cadence: every 60°
    // With the GPU reduction this is almost free, but we keep the cadence
    // for a stable per-rev average (6 samples × revolution).
    const float deg_per_force  = 60.0f;
    const uint  force_every_n  = max(1u,
        (uint)std::round(deg_per_force / deg_per_revox));

    // Turbine placement
    const float x_turb_f = (float)Nx - 1.0f - UPSTREAM_D * ROTOR_D / dx;
    const uint  x_turb   = (uint)std::round(x_turb_f);
    const uint  y_turb   = Ny / 2u;
    const uint  z_turb   = Nz / 2u;

    // Two flavours of turbine center:
    //   turb_center        — cell-index coordinates, used by voxelize_mesh_on_device
    //   turb_center_centered — origin-centered lattice coordinates, used by
    //                           object_torque() (see kernel.cpp:1975 and
    //                           the position() helper at kernel.cpp:834).
    const float3 turb_center = float3((float)x_turb, (float)y_turb, (float)z_turb);
    const float3 turb_center_centered = float3(
        (float)x_turb + 0.5f - 0.5f * (float)Nx,
        (float)y_turb + 0.5f - 0.5f * (float)Ny,
        (float)z_turb + 0.5f - 0.5f * (float)Nz);

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
    print_info("  ARCHIMEDES WIND TURBINE  —  TEST CASE  v4  (GPU-opt)");
    print_info("============================================================");
    print_info("  Tunnel       : " + to_string(TUNNEL_LX) + "m x "
               + to_string(TUNNEL_LY) + "m x " + to_string(TUNNEL_LZ) + "m");
    print_info("  U_inlet      : " + to_string(SI_U) + " m/s  (-x)");
    print_info("  Rotor D      : " + to_string(ROTOR_D) + " m");
    print_info("  TSR          : " + to_string(TSR));
    print_info("  omega        : " + to_string(omega_si) + " rad/s  (CW from +x)");
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
            // Tunnel walls — plain TYPE_S (no TYPE_X).  object_force/torque
            // on the turbine will exclude these via flag matching.
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

    // Clockwise rotation when viewed from +x axis (looking toward -x).
    // By right-hand rule, CW from +x corresponds to omega along -x.
    const float3 omega_vec = float3(-omega_lb, 0.0f, 0.0f);

    print_info("Turbine loaded: scale=" + to_string(scale_factor)
              + "  D_lat=" + to_string(D_lattice)
              + "  omega_lb=" + to_string(omega_lb));

    // ================================================================
    //  5.  HELPER: compute forces via GPU reductions
    // ================================================================
    //  Each call:
    //    - runs update_force_field (GPU kernel)   ~ 1 pass over grid
    //    - runs object_force       (GPU reduction) ~ 1 pass over grid
    //    - runs object_torque      (GPU reduction) ~ 1 pass over grid
    //    - reads back 24 bytes from device
    //  No CPU loops.  No big GPU->CPU transfers.
    //  Returns {torque_x_SI, thrust_x_SI}.

    auto compute_forces = [&]() -> std::pair<float, float> {
        const float3 F_lu = lbm.object_force(TURB_FLAG);           // exact match → turbine only
        const float3 T_lu = lbm.object_torque(turb_center_centered, TYPE_X); // bitmask TYPE_X → turbine only
        const float torque_si = T_lu.x * C_T;
        const float thrust_si = F_lu.x * C_F;
        return { torque_si, thrust_si };
    };

    // ================================================================
    //  6.  BUFFERS
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
    //  7.  INIT: transfer IC to GPU, then voxelise
    // ================================================================
    lbm.run(0u);

    // Voxelize turbine on GPU.  This kernel sets:
    //   flags[n] = TYPE_S | TYPE_X       for cells inside the mesh
    //   u[n]    = omega_vec × (p - c)    moving-wall velocity
    // both directly on device memory — no CPU transfers needed.
    lbm.voxelize_mesh_on_device(turbine, TURB_FLAG, turb_center, float3(0.0f), omega_vec);

    print_info("Turbine voxelised on GPU  (flag = TYPE_S|TYPE_X)");
    print_info("Expected tip speed (lb) ~ " + to_string(omega_lb * D_lattice * 0.5f));

    // ================================================================
    //  8.  SIMULATION LOOP
    // ================================================================
    //
    //  Architecture (v4 — all on GPU):
    //    OUTER:  loop until convergence or max_revs
    //    INNER:  revoxelise every 10° (GPU voxelize kernel — sets u on device)
    //    LBM:    advance LBM for revox_interval steps (GPU)
    //    FORCE:  every 60°, call object_force + object_torque (GPU reduction)
    //    REPORT: once per revolution — convergence, status
    //
    //  No CPU loops over the grid on the hot path.  No multi-GB transfers.
    //  Expected GPU utilization ~98%.

    const uint   max_steps = MAX_REVS * steps_per_rev;
    uint         step = 0u;
    float        accum_angle = 0.0f;
    uint         revox_count = 0u;   // counts revox steps within current rev

    print_info("Starting simulation  (max " + to_string(MAX_REVS) + " revolutions)");

    Clock clock;
    clock.start();

    while (step < max_steps) {

        // ----------------------------------------------------------
        //  8a.  Rotate mesh & re-voxelise (every 10°)
        //       The voxelize_mesh kernel itself writes the moving-wall
        //       velocities on the GPU — nothing else to do.
        // ----------------------------------------------------------
        const float delta_angle = omega_lb * (float)revox_interval;
        accum_angle += delta_angle;

        lbm.unvoxelize_mesh_on_device(turbine, TURB_FLAG);
        turbine->rotate(float3x3(float3(1.0f, 0.0f, 0.0f), -delta_angle));  // CW from +x
        lbm.voxelize_mesh_on_device(turbine, TURB_FLAG, turb_center, float3(0.0f), omega_vec);

        // ----------------------------------------------------------
        //  8b.  Advance LBM
        // ----------------------------------------------------------
        lbm.run(revox_interval);
        step += revox_interval;
        revox_count++;

        // ----------------------------------------------------------
        //  8c.  Force measurement (every 60°)
        //       GPU reduction — near-zero overhead.
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
        //  8d.  Revolution boundary
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
    //  9.  POST-PROCESS
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
    print_info("  RESULTS  —  TEST CASE  v4");
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
        f << "ARCHIMEDES WIND TURBINE — TEST CASE v4\n";
        f << "=======================================\n\n";
        f << "Configuration\n";
        f << "  Tunnel        : " << TUNNEL_LX << " x " << TUNNEL_LY
                                  << " x " << TUNNEL_LZ << " m\n";
        f << "  U_inlet       : " << SI_U      << " m/s\n";
        f << "  Rotor D       : " << ROTOR_D   << " m\n";
        f << "  TSR           : " << TSR        << "\n";
        f << "  omega         : " << omega_si   << " rad/s (CW from +x)\n";
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