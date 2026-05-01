import numpy as np
import trajectory_planning_helpers as tph
import sys
import matplotlib.pyplot as plt


def prep_track(reftrack_imp: np.ndarray,
               reg_smooth_opts: dict,
               stepsize_opts: dict,
               debug: bool = True,
               min_width: float = None,
               *,
               normals_crossing_horizon: int = 10,
               auto_increase_smoothing: bool = False,
               auto_smooth_max_iters: int = 6,
               auto_smooth_s_reg_mult: float = 2.0,
               auto_shrink_width_on_crossing: bool = False,
               auto_shrink_width_max_iters: int = 25,
               auto_shrink_width_mult: float = 0.97,
               auto_shrink_width_min_total: float = None,
               plot_on_fail: bool = True) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function prepares the inserted reference track for optimization.

    Inputs:
    reftrack_imp:               imported track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    reg_smooth_opts:            parameters for the spline approximation
    stepsize_opts:              dict containing the stepsizes before spline approximation and after spline interpolation
    debug:                      boolean showing if debug messages should be printed
    min_width:                  [m] minimum enforced track width (None to deactivate)

    Outputs:
    reftrack_interp:            track after smoothing and interpolation [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvec_normalized_interp:  normalized normal vectors on the reference line [x_m, y_m]
    a_interp:                   LES coefficients when calculating the splines
    coeffs_x_interp:            spline coefficients of the x-component
    coeffs_y_interp:            spline coefficients of the y-component
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INTERPOLATE REFTRACK AND CALCULATE INITIAL SPLINES ---------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # We may need to retry spline approximation with higher smoothing if spline normals cross.
    # This commonly happens when the reference track is noisy or has sharp kinks.
    reg_smooth_local = dict(reg_smooth_opts)

    def _patch_tph_spline_approximation_dist_to_p() -> bool:
        """Patch TPH dist_to_p to accept 1-element array inputs from SciPy optimizers.

        Some trajectory_planning_helpers versions define dist_to_p(t_glob, ...) assuming t_glob is scalar,
        but scipy.optimize.fmin passes a shape-(1,) array. That can lead to SciPy distance errors.
        """
        try:
            from scipy import interpolate, spatial
            import trajectory_planning_helpers.spline_approximation as sa
        except Exception:
            return False

        def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
            t = float(np.atleast_1d(t_glob)[0])
            s = interpolate.splev(t, path)
            s = np.asarray(s, dtype=float).reshape(-1)
            p_vec = np.asarray(p, dtype=float).reshape(-1)
            return spatial.distance.euclidean(p_vec, s)

        sa.dist_to_p = dist_to_p
        return True

    last_normals_crossing = True
    last_reftrack_interp = None
    last_normvec_normalized_interp = None
    last_coeffs_x_interp = None
    last_coeffs_y_interp = None
    last_a_interp = None

    num_smooth_attempts = max(int(auto_smooth_max_iters), 1) if auto_increase_smoothing else 1

    for attempt in range(num_smooth_attempts):
        # smoothing and interpolating reference track
        try:
            reftrack_interp = tph.spline_approximation. \
                spline_approximation(track=reftrack_imp,
                                     k_reg=reg_smooth_local["k_reg"],
                                     s_reg=reg_smooth_local["s_reg"],
                                     stepsize_prep=stepsize_opts["stepsize_prep"],
                                     stepsize_reg=stepsize_opts["stepsize_reg"],
                                     debug=debug)
        except ValueError as e:
            # Work around buggy dist_to_p implementations in some TPH versions.
            if "Input vector should be 1-D" in str(e) and _patch_tph_spline_approximation_dist_to_p():
                reftrack_interp = tph.spline_approximation. \
                    spline_approximation(track=reftrack_imp,
                                         k_reg=reg_smooth_local["k_reg"],
                                         s_reg=reg_smooth_local["s_reg"],
                                         stepsize_prep=stepsize_opts["stepsize_prep"],
                                         stepsize_reg=stepsize_opts["stepsize_reg"],
                                         debug=debug)
            else:
                raise

        # calculate splines
        refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

        coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.\
            calc_splines(path=refpath_interp_cl)

        # ------------------------------------------------------------------------------------------------------------------
        # CHECK SPLINE NORMALS FOR CROSSING POINTS -------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        normals_crossing = tph.check_normals_crossing.check_normals_crossing(
            track=reftrack_interp,
            normvec_normalized=normvec_normalized_interp,
            horizon=normals_crossing_horizon,
        )

        last_normals_crossing = normals_crossing
        last_reftrack_interp = reftrack_interp
        last_normvec_normalized_interp = normvec_normalized_interp
        last_coeffs_x_interp = coeffs_x_interp
        last_coeffs_y_interp = coeffs_y_interp
        last_a_interp = a_interp

        if not normals_crossing:
            break

        # First, optionally try shrinking widths (shortening the normal segments) before increasing smoothing.
        if auto_shrink_width_on_crossing and last_reftrack_interp is not None:
            reftrack_shrunk = np.array(last_reftrack_interp, copy=True)
            for w_attempt in range(max(int(auto_shrink_width_max_iters), 1)):
                reftrack_shrunk[:, 2] *= float(auto_shrink_width_mult)
                reftrack_shrunk[:, 3] *= float(auto_shrink_width_mult)

                if auto_shrink_width_min_total is not None:
                    min_total = float(auto_shrink_width_min_total)
                    total = reftrack_shrunk[:, 2] + reftrack_shrunk[:, 3]
                    too_small = total < min_total
                    if np.any(too_small):
                        delta = (min_total - total[too_small]) / 2.0
                        reftrack_shrunk[too_small, 2] += delta
                        reftrack_shrunk[too_small, 3] += delta

                normals_crossing_shrunk = tph.check_normals_crossing.check_normals_crossing(
                    track=reftrack_shrunk,
                    normvec_normalized=last_normvec_normalized_interp,
                    horizon=normals_crossing_horizon,
                )

                if debug:
                    print(
                        "WARNING: Spline normals crossed; shrinking track widths "
                        f"(attempt {w_attempt + 1}/{auto_shrink_width_max_iters}, mult={auto_shrink_width_mult}).",
                        file=sys.stderr,
                    )

                if not normals_crossing_shrunk:
                    last_normals_crossing = False
                    last_reftrack_interp = reftrack_shrunk
                    break

            if not last_normals_crossing:
                break

        # If shrinking didn't help (or is disabled), optionally increase smoothing and retry.
        if not auto_increase_smoothing:
            break

        try:
            s_reg = reg_smooth_local.get("s_reg", None)
            if s_reg is None:
                s_reg = 1.0
            s_reg = float(s_reg)
            if s_reg <= 0.0:
                s_reg = 1e-3
            reg_smooth_local["s_reg"] = s_reg * float(auto_smooth_s_reg_mult)
        except Exception:
            reg_smooth_local["s_reg"] = 1.0

        if debug:
            print(
                "WARNING: Spline normals crossed; retrying prep_track with higher smoothing "
                f"(attempt {attempt + 1}/{auto_smooth_max_iters}, s_reg={reg_smooth_local['s_reg']}).",
                file=sys.stderr,
            )

    if last_normals_crossing:
        if plot_on_fail and last_reftrack_interp is not None and last_normvec_normalized_interp is not None:
            bound_1_tmp = last_reftrack_interp[:, :2] + last_normvec_normalized_interp * np.expand_dims(
                last_reftrack_interp[:, 2], axis=1
            )
            bound_2_tmp = last_reftrack_interp[:, :2] - last_normvec_normalized_interp * np.expand_dims(
                last_reftrack_interp[:, 3], axis=1
            )

            plt.figure()

            plt.plot(last_reftrack_interp[:, 0], last_reftrack_interp[:, 1], 'k-')
            for i in range(bound_1_tmp.shape[0]):
                temp = np.vstack((bound_1_tmp[i], bound_2_tmp[i]))
                plt.plot(temp[:, 0], temp[:, 1], "r-", linewidth=0.7)

            plt.grid()
            ax = plt.gca()
            ax.set_aspect("equal", "datalim")
            plt.xlabel("east in m")
            plt.ylabel("north in m")
            plt.title("Error: at least one pair of normals is crossed!")

            plt.show()

        raise IOError(
            "At least two spline normals are crossed, check input or increase smoothing factor!"
        )

    # Unpack last successful iteration's outputs
    reftrack_interp = last_reftrack_interp
    normvec_normalized_interp = last_normvec_normalized_interp
    a_interp = last_a_interp
    coeffs_x_interp = last_coeffs_x_interp
    coeffs_y_interp = last_coeffs_y_interp

    # ------------------------------------------------------------------------------------------------------------------
    # ENFORCE MINIMUM TRACK WIDTH (INFLATE TIGHTER SECTIONS UNTIL REACHED) ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    manipulated_track_width = False

    if min_width is not None:
        for i in range(reftrack_interp.shape[0]):
            cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]

            if cur_width < min_width:
                manipulated_track_width = True

                # inflate to both sides equally
                reftrack_interp[i, 2] += (min_width - cur_width) / 2
                reftrack_interp[i, 3] += (min_width - cur_width) / 2

    if manipulated_track_width:
        print("WARNING: Track region was smaller than requested minimum track width -> Applied artificial inflation in"
              " order to match the requirements!", file=sys.stderr)

    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
