k_glob = 50

#  add dataset "hyqk_points_d1_d2_Y.csv"


# centr_compu_meth = "MEDIAN"  #  ["MEAN", "MEDIAN"]
scale_Q = True
remove_unimprove_points = True
N_INIT = 1
for BASE_DIR, lambda_factors, algo_improving_with_qubo, SEED, drop_cdiag, init_after_kmeans in zip([
                                                                    r"PATH_TO_DIR",


                                                                    ],
                                                                    [
                                                                        (-0.15, )
                                                                    ],
                                                                    [
                                                                        "KMEANS"
                                                                    ],
                                                                    [
                                                                        4,
                                                                     ],
                                                                     [
                                                                         True
                                                                    ],
                                                                    [
                                                                        True
                                                                    ]
                                                                        ):

    # for timeout in [round(i/100, 2) for i in range(1, 100)]:
    for timeout in [60]:
        print(f"Another run with timeout = {timeout}")




        """
        auto_run_HYQK.py - end-to-end batch runner
        • Reads cleaned datasets from CLEAN_DIR instead of generating new ones
        • Shuffles rows while keeping target aligned
        • Runs K-Means and HYQK, records metrics
        • Updates results.txt with a fixed-width line per run
        Resumes automatically after crash by skipping lines already in results.txt
        ---------------------------------------------------------------------------
        author:
        """

        # lambda_chose_minus = True
        with_alpha_in_Q = False


        import time

        # -----------------  imports  -----------------

        # ---- LightSolver SDK (leave commented if not available) ----
        import matplotlib.pyplot as plt
        from laser_mind_client_meta import MessageKeys
        from laser_mind_client import LaserMind
        from sklearn.decomposition import PCA
        # 1. Initialisation (K-Means++)
        import numpy as np
        from typing import Tuple
        import os, csv, pathlib, random
        import os

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_blobs
        from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                     calinski_harabasz_score, homogeneity_score)
        from sklearn.cluster import KMeans
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import mean_squared_error

        # ------------------------------------------------------------------
        #  K-Means++ (return = centroids, assignments) - no tricks
        # ------------------------------------------------------------------
        import numpy as np
        from typing import Tuple
        from sklearn.cluster import kmeans_plusplus


        def qubo_bruteforce(Q):
            import itertools
            best_x, best_e = None, np.inf
            for bits in itertools.product([0, 1], repeat=Q.shape[0]):
                x = np.array(bits, dtype=float)
                e = float(x @ Q @ x)
                if e < best_e:
                    best_e = e
                    best_x = x.copy()
            return best_x.astype(int), best_e


        def HYQK_initialize_sklearn(data: np.ndarray,
                                    k: int,
                                    # rng: np.random.Generator
                                    ) -> Tuple[np.ndarray, np.ndarray]:

            centroids, indices = kmeans_plusplus(
                data,
                n_clusters=k,
                random_state=SEED
            )

            distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # (n,k)
            assignments = np.argmin(distances, axis=1)

            return centroids, assignments


        # 2. Helpers ------------------------------------------------
        def distance_matrix(data, centroids):
            n, k = data.shape[0], centroids.shape[0]
            dist = np.zeros((n, k))
            for i in range(n):
                for j in range(k):
                    diff = data[i] - centroids[j]
                    dist[i, j] = diff @ diff
            return dist

        def two_closest_centroids(D):
            A = np.argmin(D, axis=1)
            B = np.argsort(D, axis=1)[:, 1]
            return A, B

        # in those where the algorithm is better
        def centroid_shift_tables(data, assignments, A, B, centroids):
            n, d = data.shape
            k   = centroids.shape[0]
            clusters = [np.where(assignments == c)[0] for c in range(k)]
            sums     = [data[idxs].sum(0) for idxs in clusters]
            c = np.zeros((n, n)); d_ = np.zeros((n, n))
            for i in range(n):
                clsA, clsB = A[i], B[i]
                if len(clusters[clsA]) <= 1:
                    continue
                sizeA, sizeB = len(clusters[clsA]), len(clusters[clsB])
                new_mu_A = (sums[clsA] - data[i]) / (sizeA - 1)
                new_mu_B = (sums[clsB] + data[i]) / (sizeB + 1)
                for j in clusters[clsA]:
                    # if j == i:
                    #     continue  # do not fill c[i, i]

                    c[i, j] = (data[j] - new_mu_A) @ (data[j] - new_mu_A) - \
                              (data[j] - centroids[clsA]) @ (data[j] - centroids[clsA])
                for j in clusters[clsB]:
                    d_[i, j] = (data[j] - new_mu_B) @ (data[j] - new_mu_B) - \
                               (data[j] - centroids[clsB]) @ (data[j] - centroids[clsB])
            return c, d_

        # ---------- hybrid Q + lambda-annealing ---------------------------------------
        # + ALPHA TO H TO THE QUBO!!!!
        def _add_cardinality(Q0, max_frac=0.20, alpha_scale=1.3):
            """Q0 = c - d_  ->  plus alpha - (sum X - m)^2 (single block, once)"""
            n        = Q0.shape[0]
            m        = int(np.ceil(max_frac * n))
            Amax     = np.max(np.abs(Q0))
            alpha    = alpha_scale * Amax

            Q = Q0.copy()
            diag_add = alpha - 2 * alpha * m
            np.fill_diagonal(Q, Q.diagonal() + diag_add)

            tri = np.triu_indices(n, k=1)
            Q[tri] += 2 * alpha
            Q[(tri[1], tri[0])] = Q[tri]
            return Q, Amax

        # def solve_HYQK_iter(c, d_):
        #     """
        #     Build a hybrid matrix once and solve a few times with decreasing lambda.
        #     """
        #     Q0 = c - d_
        #     Q0 = 0.5 * (Q0 + Q0.T)  # enforce symmetry
        #     Q = Q0.copy()
        #     Amax = np.max(np.abs(Q0))  # keep scaling for lambda
        #
        #     X = np.zeros(Q.shape[0], dtype=int)
        #     if lambda_chose_minus:
        #         lambda_factors = (-1.4, -0.9, -0.6, -0.3)
        #     else:
        #         lambda_factors = (1.4, 0.9, 0.6, 0.3)
        #     for f in lambda_factors:
        #         lam  = f * 0.7 * Amax
        #         Ql   = Q.copy()
        #         np.fill_diagonal(Ql, Ql.diagonal() + d_.sum(1) + lam)
        #         print(f"no Error Q is {Q.shape}")
        #         X, _ = get_X_light_computer(Ql)
        #         if X is None:
        #             break
        #         freeze = np.where(X == 1)[0]
        #         Q[freeze, :] = Q[:, freeze] = 0
        #     return X


        def solve_HYQK_iter(c, d_, cur_lamb_i):

            Q0 = c - d_

            # --- enforce symmetry, no penalty ---
            Q0 = 0.5 * (Q0 + Q0.T)  # ← single-line fix
            Q = Q0.copy()  # ← no cardinality penalty
            if drop_cdiag:
                np.fill_diagonal(Q, 0.0)
            # no alpha
            # Amax = np.max(np.abs(Q0))  # keep scaling for λ
            # with alpha
            if with_alpha_in_Q:
                Q, Amax  = _add_cardinality(Q0)
            else:
                Amax = np.max(np.abs(Q0))  # keep scaling for λ
            X        = np.zeros(Q.shape[0], dtype=int)
            print(f"lambda_factors: {lambda_factors}")
            print(f"BASE_DIR: {BASE_DIR}")
            # count = 0
            # for i in range(n):
            #     if np.all(Q[i, :] == 0) and np.all(Q[:, i] == 0):
            #         count += 1
            # print(f"num items in Q not 0 = {count}")
            # print(f"num got froze (line i and column i are 0) = {np.count_nonzero(Q[0, :] == Q[:, 0])}")
            f = lambda_factors[cur_lamb_i]
            lam  = f * 0.7 * Amax                # λ_t
            Qλ   = Q.copy()
            np.fill_diagonal(Qλ, Qλ.diagonal() + d_.sum(1) + lam)

            try:
                # SCALING FOR IMAGE showing
                _save_qubo_matrix_figure(1000 * Qλ, stage_tag=f"03_Q_iter_{cur_lamb_i + 1}")
                # _save_qubo_matrix_figure(Qλ, stage_tag=f"03_Q_iter_{cur_lamb_i + 1}")
            except Exception as e:
                print("save 03_Q failed:", e)


            print(f"no Error Q is {Q.shape}")
            X, _ = get_X_light_computer(Qλ)       # ← LightSolver
            # X, _ = qubo_bruteforce(Qλ)       # ← LightSolver
            print("Qλ:", Qλ)
            # print("Q[8,3]:", Q[8,3])
            print(f"lambda={f} (actually {lam})")
            print(f"type(X)={type(X)}")
            # X = None
            # if X is None:
            #     break                            
            # return None
            print(f"X returned = {X}")
            print(f"% num points changed cluster lambda {lambda_factors[cur_lamb_i]}: {(np.sum(X) * 100) / Q.shape[0]}")
            return X



        # KMeans:
        def kmeans_one_step(data, centroids):
            D = distance_matrix(data, centroids)
            return np.argmin(D, axis=1)


        def recompute_centroids_kmeans(data, assignments, k):
            d = data.shape[1]
            C = np.zeros((k, d))
            for cls in range(k):
                pts = data[assignments == cls]
                if len(pts):
                    C[cls] = pts.mean(0)
                else:
                    C[cls] = np.nan

            if np.isnan(C).any():
                nearest = C[np.nan_to_num(assignments, nan=0).astype(int)]
                sq = np.sum((data - nearest) ** 2, axis=1)
                far_order = np.argsort(-sq)

                empty_ids = np.where(np.isnan(C).any(axis=1))[0].tolist()
                used = set()
                for i in far_order:
                    if not empty_ids:
                        break
                    if i in used:
                        continue
                    cls = empty_ids.pop(0)
                    C[cls] = data[i]
                    used.add(i)

            return C

        def _pairwise_l2(a, b):
            # a: (n_a, d), b: (n_b, d)
            a2 = np.sum(a * a, axis=1, keepdims=True)
            b2 = np.sum(b * b, axis=1, keepdims=True).T
            sq = a2 + b2 - 2 * np.dot(a, b.T)
            sq = np.maximum(sq, 0.0)
            return np.sqrt(sq)


        from sklearn.mixture import GaussianMixture

        import numpy as np

        def snap_centroids_to_data(centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
            D = _pairwise_l2(centroids, X)
            nearest_idx = np.argmin(D, axis=1)
            return X[nearest_idx]


        import numpy as np


        # ------------------------------------------------------------------
        #  Plot helper (signature preserved)
        # ------------------------------------------------------------------
        def _plot_clusters(data,
                           centroids,
                           assignments,
                           title,
                           old_centroids=None,
                           highlight_idx=None,
                           extra_centroids=None,      # dict: {"ALG": ndarray(k,d), ...}
                           max_dim=3):
            """
            Draw a scatter-plot of the clustering result.
            """
            from matplotlib import pyplot as plt
            from sklearn.decomposition import PCA

            d_orig = data.shape[1]

            # optional PCA reduction
            if d_orig > max_dim:
                pca        = PCA(n_components=max_dim, random_state=0)
                data_p     = pca.fit_transform(data)
                cent_p     = pca.transform(centroids)
                old_cent_p = pca.transform(old_centroids) if old_centroids is not None else None
                extra_p    = {name: pca.transform(C) for name, C in (extra_centroids or {}).items()}
            else:
                data_p, cent_p, old_cent_p = data, centroids, old_centroids
                extra_p = extra_centroids or {}

            k        = centroids.shape[0]
            colours  = plt.get_cmap("tab10").colors

            # choose 2-D or 3-D axes
            is_3d = data_p.shape[1] == 3 and max_dim == 3
            if is_3d:
                fig = plt.figure(figsize=(6, 4))
                ax  = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))

            # points and their centroids
            for cid in range(k):
                pts = data_p[assignments == cid]
                ax.scatter(*pts.T[:2] if not is_3d else pts.T,
                           s=18,
                           color=colours[cid % 10],
                           label=f"C{cid}")
                ax.scatter(*cent_p[cid][:2] if not is_3d else cent_p[cid],
                           marker="X",
                           s=150,
                           color=colours[cid % 10],
                           edgecolors="k")

            # centroid shift arrows
            if old_cent_p is not None:
                for oc, nc in zip(old_cent_p, cent_p):
                    if is_3d:
                        ax.quiver(oc[0], oc[1], oc[2],
                                  *(nc - oc),
                                  arrow_length_ratio=0.3,
                                  color="black",
                                  linewidth=3.0,
                                  alpha=0.8)
                    else:
                        ax.arrow(oc[0], oc[1],
                                 *(nc - oc)[:2],
                                 width=0.004,
                                 head_width=0.04,
                                 head_length=0.02,
                                 color="black",
                                 length_includes_head=True,
                                 alpha=0.8)

            # highlighted points
            if highlight_idx is not None and highlight_idx.size:
                hp = data_p[highlight_idx]
                ax.scatter(*hp.T[:2] if not is_3d else hp.T,
                           s=60,
                           facecolors="none",
                           edgecolors="red",
                           linewidths=1.4)

            # extra centroid sets
            if extra_p:
                marker_cycle = ["^", "s", "P", "D", "*", "v", "<", ">", "H"]
                for m, (name, C) in zip(marker_cycle, extra_p.items()):
                    for cid, cvec in enumerate(C):
                        ax.scatter(*cvec[:2] if not is_3d else cvec,
                                   marker=m,
                                   s=130,
                                   facecolors="none",
                                   edgecolors="black",
                                   linewidths=1.2,
                                   label=f"{name}_C{cid}")

            ax.set_title(title)
            ax.legend(loc="best", labelspacing=0.8, fontsize=8)
            plt.tight_layout()
            plt.show()

        # ------------------------------------------------------------------
        #  align_centroids - Hungarian mapping to keep cluster colors stable
        # ------------------------------------------------------------------
        def align_centroids(old_C, new_C, labels):
            """
            Returns:
              new_C_aligned - same order as old_C
              labels_mapped - labels remapped to the old IDs
            """
            cost = np.linalg.norm(old_C[:, None] - new_C[None, :], axis=2)
            row, col = linear_sum_assignment(cost)          # old_i <-> new_col[i]

            new_C_aligned = new_C[col]
            inv = np.empty_like(col); inv[col] = row        # new-idx -> old-id
            return new_C_aligned, inv[labels]

        def HYQK_full(data, k, viewing=False, max_iter=7, tol=1e-4, user_token=""):

            rng = np.random.default_rng(SEED)
            d = data.shape[1]
            global CURRENT_Q_POINT_IDS
            # centroids, assignments = HYQK_initialize(data, k, rng)
            if init_after_kmeans:
                assignments, km = run_kmeans(data, k)  # assignments = labels
                centroids = km.cluster_centers_.copy()  # centroids = centers
            else:
                centroids, assignments = HYQK_initialize_sklearn(data, k)


            # plot init if requested
            try:
                if viewing and d == 2:
                    _plot_clusters(data, centroids, assignments, title="Init", max_dim=2)
            except Exception:
                pass

            
            # try:
            #     if viewing and d == 2:
            #         _save_cluster_stage_figure(
            #             data=data,
            #             centroids=centroids,
            #             assignments=assignments,
            #             stage_tag="01_init",
            #             title="INIT"# seed=4. In this example it converges after 1 QUBO iteration, and a good example that HYQK is not perfect, but performs usually as KMEANS and better. it belongs to "Clusters imbalance <= 0.1" and won in all metrics.
            #         )
            # except Exception as e:
            #     print("save 01_init failed:", e)
            prev_cleanup_assignments = assignments.copy()

            # ---------- MAIN LOOP ----------
            for it in range(max_iter):
                assignments_prev = assignments.copy()

                centroids_for_tables = centroids

                D = distance_matrix(data, centroids_for_tables)
                A, B = two_closest_centroids(D)
                c, d_ = centroid_shift_tables(data, assignments, A, B, centroids_for_tables)
                if remove_unimprove_points:
                    Amax = np.max(np.abs(c - d_))
                    mask = np.max(np.abs(c - d_), axis=1) >  0.5 * Amax
                    idxs = np.where(mask)[0]

                    # try:
                    #     if viewing and d == 2:
                    #         _save_cluster_stage_figure(
                    #             data=data,
                    #             centroids=centroids_for_tables,
                    #             assignments=assignments,
                    #             stage_tag=f"02_suspected_iter_{it + 1}",
                    #             title=f"Suspected points for QUBO, iter {it + 1}",
                    #             suspect_idx=idxs,
                    #             draw_ab=True,
                    #             A=A,
                    #             B=B
                    #         )
                    # except Exception as e:
                    #     print("save 02_suspected failed:", e)

                    print("idxs:", idxs)
                    X = np.zeros(len(mask), dtype=int)
                    print(f"% not included in the QUBO: (1-len(idxs)/n)*100 = {(1-len(idxs)/n)*100}%")
                    if len(idxs) >= 3:
                        CURRENT_Q_POINT_IDS = _display_ids_from_local_idxs(idxs)
                        X_sub = solve_HYQK_iter(c[np.ix_(idxs, idxs)], d_[np.ix_(idxs, idxs)], it)
                        X[idxs] = X_sub if X_sub is not None else 0
                # X = solve_HYQK_iter(c, d_)
                else:
                    CURRENT_Q_POINT_IDS = CURRENT_PLOT_POINT_IDS.copy()
                    X = solve_HYQK_iter(c, d_, it)
                switches = np.where(X == 1)[0]
                assignments[switches] = B[switches]

                centroids_old = centroids.copy()

                if algo_improving_with_qubo == "KMEANS":
                    centroids = recompute_centroids_kmeans(data, assignments, k)
                    # try:
                    #     if viewing and d == 2:
                    #         _save_cluster_stage_figure(
                    #             data=data,
                    #             centroids=centroids,
                    #             assignments=assignments,
                    #             stage_tag=f"04_after_qubo_iter_{it + 1}",
                    #             title=f"After QUBO assignments (X=[1 0 1 1 0]), iter {it + 1}",
                    #             suspect_idx=idxs if remove_unimprove_points else np.arange(len(data)),
                    #             moved_idx=switches
                    #         )
                    # except Exception as e:
                    #     print("save 04_after_qubo failed:", e)
                    changed_qubo = np.where(assignments != assignments_prev)[0]
                    try:
                        if viewing:
                            _plot_clusters(
                                data,
                                centroids,
                                assignments,
                                f"Iter {it + 1} - Post-QUBO",
                                old_centroids=centroids_old,
                                highlight_idx=changed_qubo
                            )
                    except Exception:
                        pass

                    assignments_before_cleanup = assignments.copy()
                    for _ in range(3):
                        assignments = kmeans_one_step(data, centroids)
                        centroids = recompute_centroids_kmeans(data, assignments, k)
                    print("finished 5 times kmeans")
                else:
                    raise ValueError("algo_improving_with_qubo must be KMEANS, KMEDOID, or GMM")

                centroids, assignments = align_centroids(centroids_old, centroids, assignments)

                changed_cleanup = np.where(assignments != assignments_before_cleanup)[0]
                try:
                    if viewing:
                        _plot_clusters(
                            data,
                            centroids,
                            assignments,
                            f"Iter {it + 1} - Post-Cleanup",
                            old_centroids=centroids_old,
                            highlight_idx=changed_cleanup
                        )
                except Exception:
                    pass

                if np.array_equal(assignments, prev_cleanup_assignments):
                    print("CONVERGED!")
                    break
                prev_cleanup_assignments = assignments.copy()

            return assignments, centroids
        def get_X_light_computer(Q):
            """
            Send QUBO matrix Q to LightSolver quantum-hybrid computer.

            Args:
                Q: QUBO matrix (symmetric)

            Returns:
                solution: Binary vector X (0=stay, 1=switch)
                objval: Objective value achieved
            """
            # Scale Q for numerical stability
            if scale_Q:
                if np.max(np.abs(Q)) > 0:
                    Q = Q / np.max(np.abs(Q))
                    print("Q scaled by max absolute value for numerical stability")

                print(f"Q max abs = {np.max(np.abs(Q))}")
            print(f"Q symmetry error = {np.max(np.abs(Q - Q.T))}")

            try:
                # Connect to LightSolver Cloud
                lsClient = LaserMind(userToken=userToken)

                # Solve QUBO problem
                res = lsClient.solve_qubo(matrixData=Q, timeout=timeout)

                assert MessageKeys.SOLUTION in res, "Solution not found in response"

                print(f"LightSolver response: \n{res}")
                return np.asarray(res["solution"], dtype=int), float(res["objval"])

            except Exception as e:
                if "total number of variables must be between" in str(e):
                    print("ERROR: The total number of variables must be between 10-10000")
                else:
                    print(f"ERROR: {e}")
                return None, None










        ARTICLE_IMG_DIR = pathlib.Path(
            r"ENTER_DIR"
        )
        ARTICLE_IMG_DIR.mkdir(parents=True, exist_ok=True)

        CURRENT_PLOT_DATASET_NAME = ""
        CURRENT_PLOT_SHAPES = None
        CURRENT_PLOT_POINT_IDS = None
        CURRENT_PLOT_TRUE_CENTROIDS = None
        CURRENT_PLOT_KMEANS_CENTROIDS = None
        CURRENT_Q_POINT_IDS = None

        PLOT_SHAPE_BANK = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h', '*']


        def _safe_plot_name(s):
            s = str(s)
            return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


        def _article_plot_path(stage_tag):
            ds = _safe_plot_name(CURRENT_PLOT_DATASET_NAME if CURRENT_PLOT_DATASET_NAME else "dataset")
            # return ARTICLE_IMG_DIR / f"{ds}_{stage_tag}.png"
            return ARTICLE_IMG_DIR / f"{ds}_{stage_tag}.pdf"


        def _display_ids_from_local_idxs(idxs):
            idxs = np.asarray(idxs, dtype=int)
            if CURRENT_PLOT_POINT_IDS is None:
                return idxs.copy()
            return np.asarray(CURRENT_PLOT_POINT_IDS, dtype=int)[idxs].copy()


        def _save_qubo_matrix_figure(Q, stage_tag):
            Q = np.asarray(Q, dtype=float)
            point_ids = CURRENT_Q_POINT_IDS
            if point_ids is None or len(point_ids) != Q.shape[0]:
                point_ids = np.arange(Q.shape[0], dtype=int)
            else:
                point_ids = np.asarray(point_ids, dtype=int).copy()

            vmax = np.max(np.abs(Q))
            if vmax == 0:
                vmax = 1.0

            fig, ax = plt.subplots(figsize=(7.0, 5.8))
            im = ax.imshow(Q, cmap="coolwarm", vmin=-vmax, vmax=vmax)

            ax.set_xticks(np.arange(len(point_ids)))
            ax.set_yticks(np.arange(len(point_ids)))
            ax.set_xticklabels(point_ids, fontsize=Q_TICK_LABEL_FONTSIZE)
            ax.set_yticklabels(point_ids, fontsize=Q_TICK_LABEL_FONTSIZE)
            ax.set_xlabel("point index", fontsize=Q_AXIS_LABEL_FONTSIZE, labelpad=6)
            ax.set_ylabel("point index", fontsize=Q_AXIS_LABEL_FONTSIZE, labelpad=6)
            ax.set_title("Q matrix", fontsize=Q_TITLE_FONTSIZE)

            if Q.shape[0] <= 25:
                for i in range(Q.shape[0]):
                    for j in range(Q.shape[1]):
                        val = 0.0 if abs(Q[i, j]) < 0.0005 else Q[i, j]

                        # if abs(val) < 1e-12:
                        #     label = "0"
                        # else:
                        #     mantissa, exponent = f"{val:.4e}".split("e")
                        #     exponent = int(exponent)
                        #     label = rf"${mantissa}\times10^{{{exponent}}}$"
                        if abs(val) < 1e-12:
                            label = "0"
                        else:
                            label = f"{val:.4f}".rstrip("0").rstrip(".")
                        ax.text(
                            j, i, label,
                            ha="center",
                            va="center",
                            fontsize=Q_CELL_FONTSIZE,
                            color="black"
                        )

            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=Q_COLORBAR_TICK_FONTSIZE)

            plt.tight_layout()
            fig.savefig(_article_plot_path(stage_tag), format="pdf", bbox_inches="tight", pad_inches=0.04)
            plt.close(fig)

        CENTROID_SIZE = 220
        CENTROID_LINEWIDTH = 1.8

        TITLE_FONTSIZE = 20
        AXIS_LABEL_FONTSIZE = 20
        TICK_LABEL_FONTSIZE = 12
        LEGEND_FONTSIZE = 13

        POINT_ID_FONTSIZE = 8
        CENTROID_TEXT_FONTSIZE = 9
        AB_TEXT_FONTSIZE = 7

        Q_TITLE_FONTSIZE = 20
        Q_AXIS_LABEL_FONTSIZE = 20
        Q_TICK_LABEL_FONTSIZE = 11
        Q_CELL_FONTSIZE = 13
        Q_COLORBAR_TICK_FONTSIZE = 10
        def _save_cluster_stage_figure(data,
                                       centroids,
                                       assignments,
                                       stage_tag,
                                       title,
                                       suspect_idx=None,
                                       moved_idx=None,
                                       extra_centroids=None,
                                       draw_ab=False,
                                       A=None,
                                       B=None):
            data = np.asarray(data)
            centroids = np.asarray(centroids)
            assignments = np.asarray(assignments)

            if data.ndim != 2 or data.shape[1] < 2:
                return

            data2 = data[:, :2]
            cent2 = centroids[:, :2]
            k = centroids.shape[0]

            if CURRENT_PLOT_SHAPES is None or len(CURRENT_PLOT_SHAPES) != len(data2):
                shape_labels = np.zeros(len(data2), dtype=int)
            else:
                shape_labels = np.asarray(CURRENT_PLOT_SHAPES, dtype=int)

            if CURRENT_PLOT_POINT_IDS is None or len(CURRENT_PLOT_POINT_IDS) != len(data2):
                point_ids = np.arange(len(data2), dtype=int)
            else:
                point_ids = np.asarray(CURRENT_PLOT_POINT_IDS, dtype=int)

            colours = plt.get_cmap("tab10").colors

            fig, ax = plt.subplots(figsize=(7.0, 5.2))
            ax.grid(True, alpha=0.18)
            ax.set_axisbelow(True)

            # 1. draw data points: shape by Y, color by current assignment
            unique_shapes = np.unique(shape_labels)
            for sh in unique_shapes:
                marker = PLOT_SHAPE_BANK[int(sh) % len(PLOT_SHAPE_BANK)]
                mask_sh = (shape_labels == sh)

                for cid in range(k):
                    idx = np.where(mask_sh & (assignments == cid))[0]
                    if len(idx) == 0:
                        continue

                    ax.scatter(
                        data2[idx, 0],
                        data2[idx, 1],
                        s=190,
                        marker=marker,
                        c=[colours[cid % len(colours)]],
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=3
                    )

            # 2. red circles: suspected
            if suspect_idx is not None and len(np.asarray(suspect_idx)) > 0:
                suspect_idx = np.asarray(suspect_idx, dtype=int)
                ax.scatter(
                    data2[suspect_idx, 0],
                    data2[suspect_idx, 1],
                    s=430,
                    facecolors="none",
                    edgecolors="crimson",
                    linewidths=4.0,
                    zorder=5
                )

            # 3. green circles: moved
            if moved_idx is not None and len(np.asarray(moved_idx)) > 0:
                moved_idx = np.asarray(moved_idx, dtype=int)
                ax.scatter(
                    data2[moved_idx, 0],
                    data2[moved_idx, 1],
                    s=380,
                    facecolors="none",
                    edgecolors="limegreen",
                    linewidths=4.2,
                    zorder=6
                )

            # 4. point numbers
            for i, (x, y) in enumerate(data2):
                ax.annotate(
                    str(point_ids[i]),
                    (x, y),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=POINT_ID_FONTSIZE,
                    color="black",
                    zorder=7
                )
            # 5. HYQK centroids
            for cid in range(k):
                ax.scatter(
                    cent2[cid, 0], cent2[cid, 1],
                    marker="X",
                    s=CENTROID_SIZE,
                    c=[colours[cid % len(colours)]],
                    edgecolors="black",
                    linewidths=CENTROID_LINEWIDTH,
                    zorder=8,
                    label="HYQK centroid" if cid == 0 else None
                )

                ax.annotate(
                    rf'$\mathbf{{\mu_{{{cid}}}}}$',
                    (cent2[cid, 0], cent2[cid, 1]),
                    xytext=(-6, -5),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=CENTROID_TEXT_FONTSIZE,
                    color="black",
                    zorder=10
                )

            # 6. A/B lines
            if draw_ab and suspect_idx is not None and A is not None and B is not None:
                for i in np.asarray(suspect_idx, dtype=int):
                    p = data2[i]
                    a = cent2[A[i]]
                    b = cent2[B[i]]

                    ax.plot(
                        [p[0], a[0]], [p[1], a[1]],
                        linestyle="-",
                        color="gray",
                        linewidth=1.8,
                        zorder=1
                    )

                    ax.plot(
                        [p[0], b[0]], [p[1], b[1]],
                        linestyle=(0, (5, 3)),
                        color="gray",
                        linewidth=1.8,
                        zorder=1
                    )

                    ax.annotate(
                        fr'$A_{{{point_ids[i]}}}=\mu_{{{A[i]}}}$' + '\n' + fr'$B_{{{point_ids[i]}}}=\mu_{{{B[i]}}}$',
                        (p[0], p[1]),
                        xytext=(0, -15),
                        textcoords='offset points',
                        ha='center',
                        va='top',
                        fontsize=AB_TEXT_FONTSIZE,
                        zorder=9
                    )

            # 7. extra centroids
            if extra_centroids:
                for name, C in extra_centroids.items():
                    C = np.asarray(C)[:, :2]
                    name_up = str(name).upper()

                    if name_up == "KMEANS":
                        marker = "D"
                        edge = "red"
                        legend_name = "KMeans centroid"
                        text_prefix = "K"
                        size = CENTROID_SIZE
                    elif name_up in {"Y", "TRUE", "GROUND_TRUTH"}:
                        marker = "H"
                        edge = "darkviolet"
                        legend_name = "True centroid"
                        text_prefix = "Y"
                        size = CENTROID_SIZE
                    else:
                        marker = "D"
                        edge = "purple"
                        legend_name = f"{name} centroid"
                        text_prefix = str(name)
                        size = CENTROID_SIZE

                    for cid in range(len(C)):
                        ax.scatter(
                            C[cid, 0], C[cid, 1],
                            marker=marker,
                            s=size,
                            facecolors="none",
                            edgecolors=edge,
                            linewidths=CENTROID_LINEWIDTH,
                            zorder=9,
                            label=legend_name if cid == 0 else None
                        )

                        # find nearest HYQK centroid index
                        nearest_hyqk = int(np.argmin(np.sum((cent2 - C[cid]) ** 2, axis=1)))

                        ax.annotate(
                            rf'$\mathbf{{{text_prefix}_{{{nearest_hyqk}}}}}$',
                            (C[cid, 0], C[cid, 1]),
                            xytext=(6, -8),
                            textcoords="offset points",
                            ha="left",
                            va="top",
                            fontsize=CENTROID_TEXT_FONTSIZE,
                            color=edge,
                            zorder=10
                        )

            ax.set_title(title, fontsize=TITLE_FONTSIZE)
            ax.set_xlabel("d1", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
            ax.set_ylabel("d2", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            from matplotlib.lines import Line2D

            handles, labels = ax.get_legend_handles_labels()
            uniq = {}
            for h, l in zip(handles, labels):
                if l and l not in uniq:
                    uniq[l] = h

            custom_handles = []
            custom_labels = []

            if "HYQK centroid" in uniq:
                custom_handles.append(
                    Line2D(
                        [0], [0],
                        marker='X',
                        linestyle='None',
                        markersize=14,
                        markerfacecolor='none',
                        markeredgecolor='black',
                        markeredgewidth=2.2,
                        color='black'
                    )
                )
                custom_labels.append("HYQK centroid")

            if "KMeans centroid" in uniq:
                custom_handles.append(
                    Line2D(
                        [0], [0],
                        marker='D',
                        linestyle='None',
                        markersize=11,
                        markerfacecolor='none',
                        markeredgecolor='red',
                        markeredgewidth=2.2,
                        color='red'
                    )
                )
                custom_labels.append("KMeans centroid")

            if "True centroid" in uniq:
                custom_handles.append(
                    Line2D(
                        [0], [0],
                        marker='H',
                        linestyle='None',
                        markersize=12,
                        markerfacecolor='none',
                        markeredgecolor='darkviolet',
                        markeredgewidth=2.2,
                        color='darkviolet'
                    )
                )
                custom_labels.append("True centroid")

            if suspect_idx is not None and len(np.asarray(suspect_idx)) > 0:
                custom_handles.append(
                    Line2D(
                        [0], [0],
                        marker='o',
                        linestyle='None',
                        markersize=14,
                        markerfacecolor='none',
                        markeredgecolor='crimson',
                        markeredgewidth=3.0,
                        color='crimson'
                    )
                )
                custom_labels.append("Suspected")

            if moved_idx is not None and len(np.asarray(moved_idx)) > 0:
                custom_handles.append(
                    Line2D(
                        [0], [0],
                        marker='o',
                        linestyle='None',
                        markersize=12,
                        markerfacecolor='none',
                        markeredgecolor='limegreen',
                        markeredgewidth=3.0,
                        color='limegreen'
                    )
                )
                custom_labels.append("Moved")

            if draw_ab:
                custom_handles.append(
                    Line2D(
                        [0], [0],
                        linestyle='-',
                        linewidth=2.0,
                        color='gray'
                    )
                )
                custom_labels.append("A link")

                custom_handles.append(
                    Line2D(
                        [0], [0],
                        linestyle=(0, (5, 3)),
                        linewidth=2.0,
                        color='gray'
                    )
                )
                custom_labels.append("B link")

            if custom_handles:
                ax.legend(
                    custom_handles,
                    custom_labels,
                    labelspacing=0.9,
                    loc="best",
                    fontsize=LEGEND_FONTSIZE,
                    frameon=True
                )
            plt.tight_layout()
            fig.savefig(_article_plot_path(stage_tag), format="pdf", bbox_inches="tight", pad_inches=0.04)
            plt.close(fig)





        # ---------- CONFIG -----------------------------------------------------------------
        userToken = "ENTER_YOUR_TOKEN_HERE"
        RESULTS_TXT  = os.path.join(BASE_DIR, "results.txt")
        MAX_DATASETS = 123  # was 81
        # SEED         = 42
        VIEW_Y_ALSO = True
        viewing_in_the_run = True
        random.seed(SEED)
        np.random.seed(SEED)
        # rng = np.random.default_rng(SEED)

        # NEW: read cleaned datasets from this directory
        CLEAN_DIR = "ENTER_PATH_OF_DATASETS_IN_THE_FORMAT"  # AFTER AL FIXES FINAL!!!
        CLEAN_FILES = sorted([f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(".csv")])

        KMEANS_COST_TXT = os.path.join(BASE_DIR, "kmeans_costs.txt")

        KMEANS_COST_HEADER = (
            "SEED  DROP_CDIAG INIT_AFTER_KMEANS LAMBDAS                 DATASET          "
            "KMEANS_COST      HYQK_COST        RATIO(H/K)      DELTA(H-K)\n"
        )


        def _lambdas_str(lambdas):
            return "(" + ",".join([str(x) for x in lambdas]) + ")"


        def _format_float(x):
            try:
                return f"{float(x):14.6f}"
            except Exception:
                return f"{str(x):>14}"


        if not os.path.exists(KMEANS_COST_TXT) or os.path.getsize(KMEANS_COST_TXT) == 0:
            with open(KMEANS_COST_TXT, "w", encoding="utf-8") as fh:
                fh.write(KMEANS_COST_HEADER)
        else:
            with open(KMEANS_COST_TXT, encoding="utf-8") as fh:
                first = fh.readline()
            if "KMEANS_COST" not in first:
                bak = KMEANS_COST_TXT.replace(".txt", f".old_{int(time.time())}.txt")
                os.replace(KMEANS_COST_TXT, bak)
                with open(KMEANS_COST_TXT, "w", encoding="utf-8") as fh:
                    fh.write(KMEANS_COST_HEADER)

        # Resume set
        cost_done = set()
        if os.path.exists(KMEANS_COST_TXT):
            with open(KMEANS_COST_TXT, encoding="utf-8") as fh:
                for ln in fh:
                    parts = ln.strip().split()
                    if not parts:
                        continue
                    if parts[0] == "SEED":
                        continue
                    # columns:
                    # 0 SEED
                    # 1 DROP_CDIAG
                    # 2 INIT_AFTER_KMEANS
                    # 3 LAMBDAS
                    # 4 DATASET
                    if len(parts) >= 5:
                        key = (parts[0], parts[1], parts[2], parts[3], parts[4])
                        cost_done.add(key)


        def append_kmeans_cost_line(seed, drop_cdiag, init_after_kmeans, lambdas, dataset, k_cost, h_cost):
            lam_s = _lambdas_str(lambdas)
            key = (str(seed), str(bool(drop_cdiag)), str(bool(init_after_kmeans)), lam_s, dataset)
            if key in cost_done:
                return False

            ratio = (h_cost / k_cost) if (k_cost is not None and k_cost != 0) else float("nan")
            delta = (h_cost - k_cost) if (h_cost is not None and k_cost is not None) else float("nan")

            line = (
                f"{seed:<5d} "
                f"{str(bool(drop_cdiag)):<9} "
                f"{str(bool(init_after_kmeans)):<16} "
                f"{lam_s:<22} "
                f"{dataset:<15} "
                f"{_format_float(k_cost)} "
                f"{_format_float(h_cost)} "
                f"{_format_float(ratio)} "
                f"{_format_float(delta)}\n"
            )

            with open(KMEANS_COST_TXT, "a", encoding="utf-8", newline="") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())

            cost_done.add(key)
            return True


        # ---- RATIO helpers for BK metrics (used with class means) ----
        def _ctrd_ptcd_ratio(centres: np.ndarray, X: np.ndarray, y_idx: np.ndarray):
            k = centres.shape[0]
            pair_sum = 0.0
            for i in range(k):
                for j in range(i+1, k):
                    pair_sum += np.linalg.norm(centres[i] - centres[j])
            ctrd  = pair_sum / k if k > 0 else 0.0
            ptcd  = float(np.mean(np.linalg.norm(X - centres[y_idx], axis=1)))
            ratio = (ctrd / ptcd) if ptcd != 0 else float('nan')
            return float(ctrd), float(ptcd), float(ratio)

        ## --- BK file (single file, with CTRD/PTCD/RATIO) ---
        BK_RESULTS_TXT = os.path.join(BASE_DIR, "with_biasedkmeans_fixed_results.txt")

        BK_HEADER = ("ALGO        DATASET  NUMPT CLST FEAT "
                     "  CTRD     PTCD     RATIO    SILH      DAVB     CHAR       HOMO     meanΔ       WS\n")

        # create BK header if needed
        if not os.path.exists(BK_RESULTS_TXT) or os.path.getsize(BK_RESULTS_TXT) == 0:
            with open(BK_RESULTS_TXT, "w", encoding="utf-8") as fh:
                fh.write(BK_HEADER)
        else:
            with open(BK_RESULTS_TXT, encoding="utf-8") as fh:
                first = fh.readline()
            if "RATIO" not in first:
                bak = BK_RESULTS_TXT.replace(".txt", f".old_{int(time.time())}.txt")
                os.replace(BK_RESULTS_TXT, bak)
                with open(BK_RESULTS_TXT, "w", encoding="utf-8") as fh:
                    fh.write(BK_HEADER)

        # resume set for BK
        bk_done = set()
        if os.path.exists(BK_RESULTS_TXT):
            with open(BK_RESULTS_TXT, encoding="utf-8") as fh:
                for ln in fh:
                    parts = ln.strip().split()
                    if len(parts) >= 2 and parts[0] != "ALGO":
                        bk_done.add((parts[0], parts[1]))  # (ALGO, DATASET)

        # params identical to your code
        BK_P_SWAP = 0.001
        BK_MAX_ITER = 300

        # def mean_centroids_deltas(X: np.ndarray, Y: np.ndarray, pred: np.ndarray) -> float:
        #     deltas = []
        #     for c in np.unique(Y):
        #         a = X[Y == c].mean(0)
        #         b = X[pred == c].mean(0)
        #         deltas.append(np.abs(a - b).sum())
        #     return float(np.mean(deltas))

        def _bk_rng(ds_idx: int):
            return np.random.default_rng(np.random.SeedSequence([SEED, ds_idx, 7777]))

        def kmeans_pp_init_bk(X: np.ndarray, k: int, rng_local) -> np.ndarray:
            n = X.shape[0]
            C = np.empty((k, X.shape[1]), dtype=X.dtype)
            C[0] = X[rng_local.integers(n)]
            closest = ((X - C[0]) ** 2).sum(axis=1)
            for c in range(1, k):
                probs = closest / closest.sum()
                idx   = rng_local.choice(n, p=probs)
                C[c]  = X[idx]
                closest = np.minimum(closest, ((X - C[c]) ** 2).sum(axis=1))
            return C

        def biased_kmeans(X: np.ndarray, k: int, ds_idx: int,
                          max_iter: int = BK_MAX_ITER,
                          p_swap: float = BK_P_SWAP) -> np.ndarray:
            rng_local = _bk_rng(ds_idx)
            C      = kmeans_pp_init_bk(X, k, rng_local)
            d2     = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(d2, axis=1)

            for _ in range(max_iter):
                for cls in range(k):
                    pts = X[labels == cls]
                    if len(pts):
                        C[cls] = pts.mean(0)

                d2_new = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
                new_lab = np.argmin(d2_new, axis=1)

                swap = rng_local.random(len(X)) < p_swap
                if swap.any():
                    secs = []
                    for i in np.where(swap)[0]:
                        order = np.argsort(d2_new[i])
                        secs.append(order[1] if order[0] == new_lab[i] else order[0])
                    new_lab[swap] = np.array(secs, dtype=new_lab.dtype)

                if np.array_equal(new_lab, labels):
                    break
                labels = new_lab
            return labels

        def algo_format_bk(algo: str) -> str:
            if algo == "KMEANS":
                return f"{algo}{' '*5}"
            if algo == "HYQK":
                return f"{algo}{' '*8}"
            return f"{algo} "

        def write_line_bk(fh, algo: str, ds: str, n: int, k: int, d: int,
                          ctrd, ptcd, ratio, sil, db, ch, hom, mcd, w):
            pad = "     " if algo == "KMEANS" else ("        " if algo == "HYQK" else " ")
            fh.write(
                f"{algo}{pad} {ds:<12} {n:5d}     {k:3d}     {d:3d} "
                f"{format_num(ctrd)} {format_num(ptcd)} {format_num(ratio)} "
                f"{format_num(sil)} {format_num(db)}  {format_num(ch)} "
                f"{format_num(hom)} {format_num(mcd)} {format_num(w)}\n"
            )

        # ---------- helpers -----------------------------------------------------------------
        def format_num(val):
            """exactly 5 digits before dot, 4 after (padded)."""
            return None if val is None else f"{val:9.4f}"[-10:]

        def write_result_line(fh, algo, name, npts, k, d,
                              sil, db, ch, hom, mse, w):
            if algo == "HYQK":
                line = (f"{algo:<7}    {name:<12} "
                        f"{npts:5d}     {k:3d}     {d:3d} "
                        f"{format_num(sil)} {format_num(db)}  {format_num(ch)} "
                        f"{format_num(hom)} {format_num(mse)} {format_num(w)}\n")
            elif algo == "KMEANS":
                line = (f"{algo:<7}  {name:<12} "
                        f"{npts:5d}     {k:3d}     {d:3d} "
                        f"{format_num(sil)} {format_num(db)}  {format_num(ch)} "
                        f"{format_num(hom)} {format_num(mse)} {format_num(w)}\n")
            else:
                line = (f"{algo:<7} {name:<12} "
                        f"{npts:5d}     {k:3d}     {d:3d} "
                        f"{format_num(sil)} {format_num(db)} {format_num(ch)} "
                        f"{format_num(hom)} {format_num(mse)} {format_num(w)}\n")
            key = (algo, name)
            if key not in done_keys:
                fh.write(line)
                done_keys.add(key)

        def dataset_filename(npts, with_y=False):
            return f"{npts}_points{'_with_Y' if with_y else ''}.csv"

        def run_kmeans(data, k):
            global max_iter
            print("run_kmeans")

            km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED)
            labels = km.fit_predict(data)
            print(f"KMeans converged in {km.n_iter_} iterations.")
            max_iter = km.n_iter_
            return labels, km

        # ---------- resume header files -----------------------------------------------------
        HEADER = ("ALGO         DATASET  NUMPT CLST FEAT "
                  " SILH      DAVB     CHAR       HOMO     MSE        WS\n")
        # max_iter = 7

        pathlib.Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(RESULTS_TXT) or os.path.getsize(RESULTS_TXT) == 0:
            with open(RESULTS_TXT, "w", encoding="utf-8") as fh:
                fh.write(HEADER)

        done_keys = set()
        if os.path.exists(RESULTS_TXT):
            with open(RESULTS_TXT, encoding="utf-8") as fh:
                for ln in fh:
                    parts = ln.strip().split()
                    if len(parts) >= 2:
                        if (parts[0], parts[1]) != ('ALGO', 'DATASET'):
                            done_keys.add((parts[0], parts[1]))
        print(done_keys)

        # ---------- deterministic dataset index for BK RNG ---------------------------------
        def _stable_dataset_index():
            per = {}
            for algo, name in done_keys:
                per.setdefault(name, set()).add(algo)
            completed = sum(1 for algos in per.values() if {'KMEANS', 'HYQK'}.issubset(algos))
            return completed + 1

        # ---------- main loop ----------------------------------------------------------------
        results_fh = open(RESULTS_TXT, "a", encoding="utf-8", newline='')

        generated = 0
        while generated < MAX_DATASETS:

            print(f"SEED={SEED}")
            # timeout = 0.1
            # print(f"timeout = {timeout}")
            curr_ds_idx = _stable_dataset_index()

            # build / load dataset ----------------------
            picked = None
            for fname in CLEAN_FILES:
                base_name_candidate = os.path.splitext(fname)[0]
                # print(f"base name candidate = {base_name_candidate}")
                if ("HYQK", base_name_candidate) in done_keys and ("KMEANS", base_name_candidate) in done_keys:
                    continue
                #  relevant for time measuring only!!!!!
                # if "r15" not in base_name_candidate:
                #     continue
                #  relevant for time measuring only!!!!!
                # if base_name_candidate not in datasets: # or "german_credit_cleaned" in base_name_candidate or "libras_cleaned" in base_name_candidate or "voting_cleaned" in base_name_candidate:
                #     continue
                picked = fname
                base_name = base_name_candidate
                break

            if picked is None:
                print("All cleaned datasets are already processed")
                break

            full_path = os.path.join(CLEAN_DIR, picked)
            df_loaded = pd.read_csv(full_path)

            if "s-sets_preprocessed" not in base_name and "mafat_10k_260107_B_svd50_dense_l2norm" not in base_name:
                if "target" not in df_loaded.columns:
                    raise ValueError(f'"target" column missing in {picked}')

                feature_cols = [c for c in df_loaded.columns if c != "target"]
                X = df_loaded[feature_cols].to_numpy()
                Y = df_loaded["target"].to_numpy()

                # shuffle rows while keeping X,Y aligned
                if base_name != "hyqk_points_d1_d2_Y":
                    perm = np.random.default_rng(SEED).permutation(len(df_loaded))
                    X = X[perm]
                    Y = Y[perm]
                n = X.shape[0]
                d = X.shape[1]
                classes, y_idx = np.unique(Y, return_inverse=True)
                k = len(classes)


                CURRENT_PLOT_DATASET_NAME = base_name
                CURRENT_PLOT_SHAPES = y_idx.copy()
                CURRENT_PLOT_POINT_IDS = np.arange(n)
                CURRENT_PLOT_TRUE_CENTROIDS = None

                # construct class-mean centres for BK metrics
                true_centroids = np.vstack([X[y_idx == c].mean(axis=0) for c in range(k)]).astype(float)

                CURRENT_PLOT_TRUE_CENTROIDS = true_centroids.copy()

                print(f"now ctrd, ptcd, ratio in {picked}")
                ctrd, ptcd, ratio = _ctrd_ptcd_ratio(true_centroids, X, y_idx)
                print(f"ctrd, ptcd, ratio = {ctrd, ptcd, ratio}")
                print(X.shape, d, k, "dataset:", base_name)
                print(f"K is {k}")

                # keep df for downstream usage
                df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(d)])
                print(df)
                # ---------- run K-Means ----------
                k_labels, km = run_kmeans(X, k)
                print("f")
                CURRENT_PLOT_KMEANS_CENTROIDS = km.cluster_centers_.copy()
                #     timeout = 10
                #     print(f"changed to timeout = {timeout}")


                # ---------- run HYQK -------------
                try:
                    # print(f"max_iter={max_iter}")
                    # max_iter = max_iter if max_iter < 3 else 3
                    # print(f"HYQK max_iter={max_iter if max_iter < 3 else 3}")
                    max_iter = len(lambda_factors)
                    if viewing_in_the_run and d == 2:
                        q_labels, q_centroids = HYQK_full(X, k, viewing=True, max_iter=max_iter)
                    else:
                        q_labels, q_centroids = HYQK_full(X, k, viewing=False, max_iter=max_iter)

                    # try:
                    #     if viewing_in_the_run and d == 2 and CURRENT_PLOT_TRUE_CENTROIDS is not None:
                    #         if "ILLUSTRATION...":
                    #             _save_cluster_stage_figure(
                    #                 data=X,
                    #                 centroids=q_centroids,
                    #                 assignments=q_labels,
                    #                 stage_tag="05_final_before_ground_truth",
                    #                 title="Final state before GT",
                    #                 extra_centroids={
                    #                     "KMEANS": CURRENT_PLOT_KMEANS_CENTROIDS,
                    #                     "Y": CURRENT_PLOT_TRUE_CENTROIDS
                    #                 }
                    #             )
                    # except Exception as e:
                    #     print("save 05_final_before_ground_truth failed:", e)

                    if VIEW_Y_ALSO and viewing_in_the_run and d == 2:
                        _plot_clusters(X, true_centroids, y_idx, title="Ground truth clusters", max_dim=2)
                        extra = {
                            "KMEANS": km.cluster_centers_,
                            "HYQK": q_centroids
                        }
                        _plot_clusters(X, true_centroids, y_idx,
                                       title="GT vs KMEANS vs HYQK centroids",
                                       extra_centroids=extra, max_dim=2)

                except Exception as e:
                    print("Error HYQK crashed on", base_name, ":", e)
                    break
            else:  # dataset that has no true or Y
                feature_cols = [c for c in df_loaded.columns if c != "target"]  # columns to ignore? 'target' is an example
                X = df_loaded[feature_cols].to_numpy()
                # Y = df_loaded["target"].to_numpy()

                # shuffle rows while keeping X,Y aligned
                perm = np.random.default_rng(SEED).permutation(len(df_loaded))
                X = X[perm]
                # Y = Y[perm]

                n = X.shape[0]
                d = X.shape[1]
                # classes, y_idx = np.unique(Y, return_inverse=True)
                # k = len(classes)

                # construct class-mean centres for BK metrics
                # true_centroids = np.vstack([X[y_idx == c].mean(axis=0) for c in range(k)]).astype(float)
                # print(f"now ctrd, ptcd, ratio in {picked}")
                # ctrd, ptcd, ratio = _ctrd_ptcd_ratio(true_centroids, X, y_idx)
                # print(f"ctrd, ptcd, ratio = {ctrd, ptcd, ratio}")
                ctrd, ptcd, ratio = None, None, None
                k = k_glob
                print(X.shape, d, k, "dataset:", base_name)
                print(f"K is {k}")

                # keep df for downstream usage
                df = pd.DataFrame(X, columns=[f"feature_{i + 1}" for i in range(d)])

                # ---------- run K-Means ----------
                k_labels, km = run_kmeans(X, k)

                # if n > 1000 and timeout == 0.1:
                #     timeout = 10
                #     print(f"changed to timeout = {timeout}")

                # ---------- run HYQK -------------
                try:
                    # print(f"max_iter={max_iter}")
                    # max_iter = max_iter if max_iter < 3 else 3
                    # print(f"HYQK max_iter={max_iter if max_iter < 3 else 3}")
                    max_iter = len(lambda_factors)
                    if viewing_in_the_run and d == 2:
                        q_labels, q_centroids = HYQK_full(X, k, viewing=True, max_iter=max_iter)
                    else:
                        q_labels, q_centroids = HYQK_full(X, k, viewing=False, max_iter=max_iter)

                    # if VIEW_Y_ALSO and viewing_in_the_run and d == 2:
                    #     _plot_clusters(X, true_centroids, y_idx, title="Ground truth clusters", max_dim=2)
                    #     extra = {
                    #         "KMEANS": km.cluster_centers_,
                    #         "HYQK": q_centroids
                    #     }
                    #     _plot_clusters(X, true_centroids, y_idx,
                    #                    title="GT vs KMEANS vs HYQK centroids",
                    #                    extra_centroids=extra, max_dim=2)

                except Exception as e:
                    print("Error HYQK crashed on", base_name, ":", e)
                    break


            # ---------- align labels to true Y ----------
            # def align(pred):
            #     conf = pd.crosstab(pred, Y).to_numpy()
            #     row, col = linear_sum_assignment(-conf)
            #     mapping = {r: c for r, c in zip(row, col)}
            #     return np.array([mapping[l] for l in pred])

            # --- KMEANS COST (SSE) ---
            kmeans_cost = float(km.inertia_)

            # --- HYQK COST (SSE) ---
            # HYQK returns centroids + labels. compute SSE the same way:
            hyqk_cost = float(np.sum((X - q_centroids[q_labels]) ** 2))

            append_kmeans_cost_line(
                seed=SEED,
                drop_cdiag=drop_cdiag,
                init_after_kmeans=init_after_kmeans,
                lambdas=lambda_factors,
                dataset=base_name,
                k_cost=kmeans_cost,
                h_cost=hyqk_cost
            )

            if "s-sets_preprocessed" not in base_name and "mafat_10k_260107_B_svd50_dense_l2norm" not in base_name:
                # # # # # #
                def align_to_Y(pred, Y):
                    tab = pd.crosstab(pred, Y)  # keep real labels
                    r, c = linear_sum_assignment(-tab.to_numpy())
                    rlab = tab.index.to_numpy()  # predicted labels present
                    clab = tab.columns.to_numpy()  # true labels present
                    mapping = {rlab[i]: clab[j] for i, j in zip(r, c)}
                    return np.array([mapping[l] for l in pred])

                # k_aligned = align(k_labels)
                # q_aligned = align(q_labels)
                k_aligned = align_to_Y(k_labels, Y)
                q_aligned = align_to_Y(q_labels, Y)
                # BKMEANS
                bk_raw = biased_kmeans(X, k, ds_idx=curr_ds_idx)
                # bk_aligned = align(bk_raw)
                bk_aligned = align_to_Y(bk_raw, Y)


                # ---------- metrics ----------
                def mse_score(X_, labels_, centroids_):
                    return np.mean(np.sum((X_ - centroids_[labels_]) ** 2, axis=1))

                def metrics(pred, cents):
                    sil = silhouette_score(X, pred)
                    db  = davies_bouldin_score(X, pred)
                    ch  = calinski_harabasz_score(X, pred)
                    hom = homogeneity_score(Y, pred)
                    mse = mean_squared_error(Y, pred)

                    sil_norm = max(0.0, sil)
                    db_norm = 1.0 / (1.0 + db)
                    ch_norm = ch / (ch + 1.0)
                    mse_norm = 1.0 / (1.0 + mse)

                    w = ((0.5/3.0) * sil_norm +
                         0.25 * hom +
                         0.25 * mse_norm +
                         (0.5/3.0) * db_norm +
                         (0.5/3.0) * ch_norm)
                    return sil, db, ch, hom, mse, w

                k_sil, k_db, k_ch, k_hom, k_mse, k_w = metrics(k_aligned, Y)
                q_sil, q_db, q_ch, q_hom, q_mse, q_w = metrics(q_aligned, Y)

                # ---------- save RUN csv ----------
                run_df = df.copy()
                run_df["KMEANS"] = k_aligned
                run_df["HYQK"] = q_aligned

                if "Y" in run_df.columns:
                    y_pos = run_df.columns.get_loc("Y")
                    if "BKMEANS" in run_df.columns:
                        run_df.drop(columns=["BKMEANS"], inplace=True)
                    run_df.insert(y_pos, "BKMEANS", bk_aligned)
                else:
                    run_df["BKMEANS"] = bk_aligned

                run_df["Y"] = Y
                run_df.to_csv(os.path.join(BASE_DIR, f"{base_name}_RUN.csv"), index=False)

                # ---------- append results.txt ----
                write_result_line(results_fh, "KMEANS", base_name, n, k, d,
                                  k_sil, k_db, k_ch, k_hom, k_mse, k_w)
                write_result_line(results_fh, "HYQK",  base_name, n, k, d,
                                  q_sil, q_db, q_ch, q_hom, q_mse, q_w)
                results_fh.flush()


                def mean_centroids_deltas(X, Y, pred):
                    vals = []
                    for c in np.unique(Y):
                        a = X[Y == c].mean(0)
                        mask = (pred == c)
                        if not np.any(mask):  # no points predicted as c
                            continue
                        b = X[mask].mean(0)
                        vals.append(np.abs(a - b).sum())
                    return float(np.mean(vals)) if vals else float("nan")


                def metrics_bk(pred_):
                    sil = silhouette_score(X, pred_)
                    db = davies_bouldin_score(X, pred_)
                    ch = calinski_harabasz_score(X, pred_)
                    hom = homogeneity_score(Y, pred_)
                    mcd = mean_centroids_deltas(X, Y, pred_)
                    sil_n = max(0.0, sil)
                    db_n = 1.0 / (1.0 + db)
                    ch_n = ch / (ch + 1.0)
                    mcd_n = 1.0 / (1.0 + mcd)
                    w = ((0.5 / 3.0) * sil_n +
                         0.25 * hom +
                         0.25 * mcd_n +
                         (0.5 / 3.0) * db_n +
                         (0.5 / 3.0) * ch_n)
                    return sil, db, ch, hom, mcd, w

                with open(BK_RESULTS_TXT, "a", encoding="utf-8") as bfh:
                    for algo, pred in (("KMEANS", k_aligned), ("HYQK", q_aligned), ("BKMEANS", bk_aligned)):
                        if (algo, base_name) in bk_done:
                            continue
                        sil, db, ch, hom, mcd, w = metrics_bk(pred)
                        write_line_bk(bfh, algo, base_name, n, k, d,
                                      ctrd, ptcd, ratio, sil, db, ch, hom, mcd, w)
                        bk_done.add((algo, base_name))

            else:

                bk_raw = biased_kmeans(X, k, ds_idx=curr_ds_idx)
                def metrics(pred, cents):
                    sil = silhouette_score(X, pred)
                    db = davies_bouldin_score(X, pred)
                    ch = calinski_harabasz_score(X, pred)
                    hom = None
                    mse = None

                    sil_norm = max(0.0, sil)
                    db_norm = 1.0 / (1.0 + db)
                    ch_norm = ch / (ch + 1.0)
                    mse_norm = None

                    w = None
                    return sil, db, ch, hom, mse, w


                k_sil, k_db, k_ch, k_hom, k_mse, k_w = metrics(k_labels, None)
                q_sil, q_db, q_ch, q_hom, q_mse, q_w = metrics(q_labels, None)

                # ---------- save RUN csv ----------
                run_df = df.copy()
                run_df["KMEANS"] = k_labels
                run_df["HYQK"] = q_labels

                if "Y" in run_df.columns:
                    y_pos = run_df.columns.get_loc("Y")
                    if "BKMEANS" in run_df.columns:
                        run_df.drop(columns=["BKMEANS"], inplace=True)
                    run_df.insert(y_pos, "BKMEANS", bk_raw)
                else:
                    run_df["BKMEANS"] = bk_raw

                # run_df["Y"] = Y
                run_df.to_csv(os.path.join(BASE_DIR, f"{base_name}_RUN.csv"), index=False)

                # ---------- append results.txt ----
                write_result_line(results_fh, "KMEANS", base_name, n, k, d,
                                  k_sil, k_db, k_ch, k_hom, k_mse, k_w)
                write_result_line(results_fh, "HYQK", base_name, n, k, d,
                                  q_sil, q_db, q_ch, q_hom, q_mse, q_w)
                results_fh.flush()


                # BK live metrics (with meanΔ instead of MSE)
                # def mean_centroids_deltas_local(X_, Y_, pred_):
                #     deltas = []
                #     for c in np.unique(Y_):
                #         a = X_[Y_ == c].mean(0)
                #         b = X_[pred_ == c].mean(0)
                #         deltas.append(np.abs(a - b).sum())
                #     return float(np.mean(deltas))

                def mean_centroids_deltas(X, Y, pred):
                    vals = []
                    for c in np.unique(Y):
                        a = X[Y == c].mean(0)
                        mask = (pred == c)
                        if not np.any(mask):  # no points predicted as c
                            continue
                        b = X[mask].mean(0)
                        vals.append(np.abs(a - b).sum())
                    return float(np.mean(vals)) if vals else float("nan")


                def metrics_bk(pred_):
                    sil = silhouette_score(X, pred)
                    db = davies_bouldin_score(X, pred)
                    ch = calinski_harabasz_score(X, pred)
                    hom = None
                    mse = None

                    sil_norm = max(0.0, sil)
                    db_norm = 1.0 / (1.0 + db)
                    ch_norm = ch / (ch + 1.0)
                    mse_norm = None

                    w = None
                    return sil, db, ch, hom, mse, w


                with open(BK_RESULTS_TXT, "a", encoding="utf-8") as bfh:
                    for algo, pred in (("KMEANS", k_labels), ("HYQK", q_labels), ("BKMEANS", bk_raw)):
                        if (algo, base_name) in bk_done:
                            continue
                        sil, db, ch, hom, mcd, w = metrics_bk(pred)
                        write_line_bk(bfh, algo, base_name, n, k, d,
                                      ctrd, ptcd, ratio, sil, db, ch, hom, mcd, w)
                        bk_done.add((algo, base_name))




            done_keys.add(("KMEANS", base_name))
            done_keys.add(("HYQK",  base_name))
            generated += 1
            print(f"Dataset {base_name} done (#{generated}/{MAX_DATASETS})")

        results_fh.close()
    # break
