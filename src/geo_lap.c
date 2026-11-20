/*
 * geo_lap.c - exact min-cost assignment of points to grid sites (L1 cost).
 *
 * Successive shortest augmenting paths (Hungarian / Jonker-Volgenant style)
 * with Dijkstra on reduced costs.  The bipartite graph is complete (every
 * terminal may take every site), but it is never materialised: sites are kept
 * in a uniform cell grid and each terminal's out-edges are generated lazily,
 * one Chebyshev ring of cells at a time.  A ring event carries a lower bound
 * on the reduced cost of every site in it, so Dijkstra stays exact.
 *
 * Invariants (r_ij = c_ij - u_i - v_j):
 *   r_ij >= 0 for all pairs, r = 0 on matched pairs, v_j <= 0, free sites v = 0.
 * Hence r_ij >= c_ij - u_i, which is the ring lower bound used below.
 *
 * Build: gcc -O3 -shared -fPIC -o libgeolap.so geo_lap.c
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double key;
    int a;      /* row or site index */
    int b;      /* ring index for RING events */
    int type;   /* 0 = ROW, 1 = SITE, 2 = RING */
} Ev;

typedef struct {
    Ev *e;
    long n, cap;
} Heap;

static int heap_push(Heap *h, double key, int a, int b, int type) {
    if (h->n == h->cap) {
        long nc = h->cap ? h->cap * 2 : 1024;
        Ev *ne = (Ev *)realloc(h->e, (size_t)nc * sizeof(Ev));
        if (!ne) return -1;
        h->e = ne;
        h->cap = nc;
    }
    long i = h->n++;
    Ev x = {key, a, b, type};
    while (i > 0) {
        long p = (i - 1) >> 1;
        if (h->e[p].key <= key) break;
        h->e[i] = h->e[p];
        i = p;
    }
    h->e[i] = x;
    return 0;
}

static Ev heap_pop(Heap *h) {
    Ev top = h->e[0];
    Ev last = h->e[--h->n];
    long i = 0, n = h->n;
    for (;;) {
        long l = 2 * i + 1;
        if (l >= n) break;
        long c = l;
        if (l + 1 < n && h->e[l + 1].key < h->e[l].key) c = l + 1;
        if (h->e[c].key >= last.key) break;
        h->e[i] = h->e[c];
        i = c;
    }
    if (n > 0) h->e[i] = last;
    return top;
}

/*
 * n rows at (px,py); m sites at (sx,sy); cell = grid cell size (use pitch).
 * order: processing order of rows (length n), or NULL for 0..n-1.
 * out_assign[n]: site index per row.
 * stats[8]: 0 heap pops, 1 site relaxations, 2 rows scanned, 3 ring events,
 *           4 max heap size, 5 augment path edges total.
 * Returns 0 on success, negative on error.
 */
int geo_lap(int n, const double *px, const double *py,
            int m, const double *sx, const double *sy,
            double cell, const int *order, int *out_assign, long long *stats) {
    if (n > m) return -1;
    if (cell <= 0) return -2;
    double ox = sx[0], oy = sy[0], mx = sx[0], my = sy[0];
    for (int j = 1; j < m; j++) {
        if (sx[j] < ox) ox = sx[j];
        if (sy[j] < oy) oy = sy[j];
        if (sx[j] > mx) mx = sx[j];
        if (sy[j] > my) my = sy[j];
    }
    ox -= 0.5 * cell;
    oy -= 0.5 * cell;
    int GX = (int)floor((mx - ox) / cell) + 1;
    int GY = (int)floor((my - oy) / cell) + 1;
    long NC = (long)GX * GY;

    int *cstart = (int *)calloc((size_t)NC + 1, sizeof(int));
    int *csites = (int *)malloc((size_t)m * sizeof(int));
    int *fill = (int *)calloc((size_t)NC, sizeof(int));
    double *u = (double *)calloc((size_t)n, sizeof(double));
    double *v = (double *)calloc((size_t)m, sizeof(double));
    int *mrow = (int *)malloc((size_t)n * sizeof(int));
    int *msite = (int *)malloc((size_t)m * sizeof(int));
    double *drow = (double *)malloc((size_t)n * sizeof(double));
    double *dsite = (double *)malloc((size_t)m * sizeof(double));
    int *srow = (int *)calloc((size_t)n, sizeof(int));   /* stamp: reached */
    int *ssite = (int *)calloc((size_t)m, sizeof(int));
    int *frow = (int *)calloc((size_t)n, sizeof(int));   /* stamp: final */
    int *fsite = (int *)calloc((size_t)m, sizeof(int));
    int *pred = (int *)malloc((size_t)m * sizeof(int));
    int *rcx = (int *)malloc((size_t)n * sizeof(int));
    int *rcy = (int *)malloc((size_t)n * sizeof(int));
    int *rmax = (int *)malloc((size_t)n * sizeof(int));
    int *scan_r = (int *)malloc((size_t)n * sizeof(int));
    int *scan_s = (int *)malloc((size_t)m * sizeof(int));
    Heap h = {0, 0, 0};
    int rc = 0;
    if (!cstart || !csites || !fill || !u || !v || !mrow || !msite || !drow || !dsite ||
        !srow || !ssite || !frow || !fsite || !pred || !rcx || !rcy || !rmax || !scan_r || !scan_s) {
        rc = -3;
        goto done;
    }

    /* bucket sites (CSR, ascending site id inside each cell) */
    for (int j = 0; j < m; j++) {
        int cx = (int)floor((sx[j] - ox) / cell), cy = (int)floor((sy[j] - oy) / cell);
        if (cx < 0) cx = 0;
        if (cx >= GX) cx = GX - 1;
        if (cy < 0) cy = 0;
        if (cy >= GY) cy = GY - 1;
        cstart[(long)cy * GX + cx + 1]++;
    }
    for (long c = 0; c < NC; c++) cstart[c + 1] += cstart[c];
    for (int j = 0; j < m; j++) {
        int cx = (int)floor((sx[j] - ox) / cell), cy = (int)floor((sy[j] - oy) / cell);
        if (cx < 0) cx = 0;
        if (cx >= GX) cx = GX - 1;
        if (cy < 0) cy = 0;
        if (cy >= GY) cy = GY - 1;
        long c = (long)cy * GX + cx;
        csites[cstart[c] + fill[c]++] = j;
    }
    for (int i = 0; i < n; i++) {
        int cx = (int)floor((px[i] - ox) / cell), cy = (int)floor((py[i] - oy) / cell);
        rcx[i] = cx;
        rcy[i] = cy;
        int r = abs(cx), t;
        t = abs(cx - (GX - 1)); if (t > r) r = t;
        t = abs(cy);            if (t > r) r = t;
        t = abs(cy - (GY - 1)); if (t > r) r = t;
        rmax[i] = r;
        mrow[i] = -1;
    }
    for (int j = 0; j < m; j++) msite[j] = -1;

    long long pops = 0, relax = 0, rows_scanned = 0, rings = 0, maxheap = 0, pathlen = 0;
    int stamp = 0;

    for (int it = 0; it < n; it++) {
        int i0 = order ? order[it] : it;
        stamp++;
        h.n = 0;
        int nsr = 0, nss = 0;
        drow[i0] = 0.0;
        srow[i0] = stamp;
        if (heap_push(&h, 0.0, i0, 0, 0)) { rc = -3; goto done; }
        int target = -1;
        double D = 0.0;
        while (h.n > 0) {
            if (h.n > maxheap) maxheap = h.n;
            Ev ev = heap_pop(&h);
            pops++;
            if (ev.type == 1) { /* SITE */
                int j = ev.a;
                if (fsite[j] == stamp || ev.key > dsite[j]) continue;
                fsite[j] = stamp;
                scan_s[nss++] = j;
                if (msite[j] < 0) { target = j; D = ev.key; break; }
                int ip = msite[j];
                /* matched edge is tight: reduced cost 0 */
                if (srow[ip] != stamp || ev.key < drow[ip]) {
                    srow[ip] = stamp;
                    drow[ip] = ev.key;
                    if (heap_push(&h, ev.key, ip, 0, 0)) { rc = -3; goto done; }
                }
            } else if (ev.type == 0) { /* ROW */
                int i = ev.a;
                if (frow[i] == stamp || ev.key > drow[i]) continue;
                frow[i] = stamp;
                scan_r[nsr++] = i;
                rows_scanned++;
                double lb = -u[i];
                if (lb < 0) lb = 0;
                if (heap_push(&h, drow[i] + lb, i, 0, 2)) { rc = -3; goto done; }
            } else { /* RING */
                int i = ev.a, r = ev.b;
                rings++;
                double di = drow[i], xi = px[i], yi = py[i], ui = u[i];
                int cx = rcx[i], cy = rcy[i];
                for (int dy = -r; dy <= r; dy++) {
                    int yy = cy + dy;
                    if (yy < 0 || yy >= GY) continue;
                    int step = (dy == -r || dy == r) ? 1 : 2 * r;
                    for (int dx = -r; dx <= r; dx += (step ? step : 1)) {
                        int xx = cx + dx;
                        if (xx >= 0 && xx < GX) {
                            long c = (long)yy * GX + xx;
                            for (int q = cstart[c]; q < cstart[c + 1]; q++) {
                                int j = csites[q];
                                if (fsite[j] == stamp) continue;
                                if (mrow[i] == j) continue; /* own matched edge is reverse-only */
                                double cij = fabs(xi - sx[j]) + fabs(yi - sy[j]);
                                double red = cij - ui - v[j];
                                if (red < 0) red = 0; /* fp guard */
                                double nd = di + red;
                                relax++;
                                if (ssite[j] != stamp || nd < dsite[j]) {
                                    ssite[j] = stamp;
                                    dsite[j] = nd;
                                    pred[j] = i;
                                    if (heap_push(&h, nd, j, 0, 1)) { rc = -3; goto done; }
                                }
                            }
                        }
                        if (r == 0) break;
                    }
                }
                if (r + 1 <= rmax[i]) {
                    double lb = (double)r * cell - ui; /* min L1 to ring r+1 is >= r*cell */
                    if (lb < 0) lb = 0;
                    if (heap_push(&h, di + lb, i, r + 1, 2)) { rc = -3; goto done; }
                }
            }
        }
        if (target < 0) { rc = -4; goto done; }
        /* augment */
        int j = target;
        for (;;) {
            int i = pred[j];
            int nj = mrow[i];
            mrow[i] = j;
            msite[j] = i;
            pathlen++;
            if (i == i0) break;
            j = nj;
        }
        /* dual update for finalized nodes */
        for (int q = 0; q < nsr; q++) {
            int i = scan_r[q];
            u[i] += D - drow[i];
        }
        for (int q = 0; q < nss; q++) {
            int s = scan_s[q];
            if (dsite[s] < D) v[s] -= D - dsite[s];
        }
    }
    for (int i = 0; i < n; i++) out_assign[i] = mrow[i];
    if (stats) {
        stats[0] = pops; stats[1] = relax; stats[2] = rows_scanned;
        stats[3] = rings; stats[4] = maxheap; stats[5] = pathlen;
    }
done:
    free(cstart); free(csites); free(fill); free(u); free(v); free(mrow); free(msite);
    free(drow); free(dsite); free(srow); free(ssite); free(frow); free(fsite); free(pred);
    free(rcx); free(rcy); free(rmax); free(scan_r); free(scan_s); free(h.e);
    return rc;
}

/*
 * geo_lap2: same algorithm, with a two-level bound.  Cells are grouped into
 * bs x bs blocks.  For each block we keep vmax[b] = max v_j over its sites
 * (v only ever decreases, so vmax is refreshed for touched blocks after each
 * dual update).  A row's out-edges are generated block by block:
 *   BRING(i, R): push a BLOCK event for every block in Chebyshev block-ring R,
 *                keyed by d_i + max(0, mindist(p_i, block) - u_i - vmax[b]),
 *                then push BRING(i, R+1) keyed by d_i + max(0, R*bs*cell - u_i).
 *   BLOCK(i, b): relax every site in block b.
 * Both keys are valid lower bounds on d_i + r_ij for the sites they cover.
 */
int geo_lap2(int n, const double *px, const double *py,
             int m, const double *sx, const double *sy,
             double cell, int bs, const int *order, int *out_assign, long long *stats) {
    if (n > m) return -1;
    if (cell <= 0 || bs <= 0) return -2;
    double ox = sx[0], oy = sy[0], mx = sx[0], my = sy[0];
    for (int j = 1; j < m; j++) {
        if (sx[j] < ox) ox = sx[j];
        if (sy[j] < oy) oy = sy[j];
        if (sx[j] > mx) mx = sx[j];
        if (sy[j] > my) my = sy[j];
    }
    ox -= 0.5 * cell;
    oy -= 0.5 * cell;
    int GX = (int)floor((mx - ox) / cell) + 1;
    int GY = (int)floor((my - oy) / cell) + 1;
    int BX = (GX + bs - 1) / bs, BY = (GY + bs - 1) / bs;
    long NB = (long)BX * BY;
    double bw = bs * cell;

    /* sites bucketed directly by block (CSR) */
    int *bstart = (int *)calloc((size_t)NB + 1, sizeof(int));
    int *bsites = (int *)malloc((size_t)m * sizeof(int));
    int *fill = (int *)calloc((size_t)NB, sizeof(int));
    int *sblock = (int *)malloc((size_t)m * sizeof(int));
    double *vmax = (double *)calloc((size_t)NB, sizeof(double));
    int *bdirty = (int *)calloc((size_t)NB, sizeof(int));
    int *dirty_list = (int *)malloc((size_t)m * sizeof(int));
    double *u = (double *)calloc((size_t)n, sizeof(double));
    double *v = (double *)calloc((size_t)m, sizeof(double));
    int *mrow = (int *)malloc((size_t)n * sizeof(int));
    int *msite = (int *)malloc((size_t)m * sizeof(int));
    double *drow = (double *)malloc((size_t)n * sizeof(double));
    double *dsite = (double *)malloc((size_t)m * sizeof(double));
    int *srow = (int *)calloc((size_t)n, sizeof(int));
    int *ssite = (int *)calloc((size_t)m, sizeof(int));
    int *frow = (int *)calloc((size_t)n, sizeof(int));
    int *fsite = (int *)calloc((size_t)m, sizeof(int));
    int *pred = (int *)malloc((size_t)m * sizeof(int));
    int *rbx = (int *)malloc((size_t)n * sizeof(int));
    int *rby = (int *)malloc((size_t)n * sizeof(int));
    int *rmax = (int *)malloc((size_t)n * sizeof(int));
    int *scan_r = (int *)malloc((size_t)n * sizeof(int));
    int *scan_s = (int *)malloc((size_t)m * sizeof(int));
    Heap h = {0, 0, 0};
    int rc = 0;
    if (!bstart || !bsites || !fill || !sblock || !vmax || !bdirty || !dirty_list || !u || !v ||
        !mrow || !msite || !drow || !dsite || !srow || !ssite || !frow || !fsite || !pred ||
        !rbx || !rby || !rmax || !scan_r || !scan_s) {
        rc = -3;
        goto done2;
    }
    for (int j = 0; j < m; j++) {
        int cx = (int)floor((sx[j] - ox) / cell), cy = (int)floor((sy[j] - oy) / cell);
        if (cx < 0) cx = 0;
        if (cx >= GX) cx = GX - 1;
        if (cy < 0) cy = 0;
        if (cy >= GY) cy = GY - 1;
        int b = (cy / bs) * BX + (cx / bs);
        sblock[j] = b;
        bstart[b + 1]++;
    }
    for (long b = 0; b < NB; b++) bstart[b + 1] += bstart[b];
    for (int j = 0; j < m; j++) {
        int b = sblock[j];
        bsites[bstart[b] + fill[b]++] = j;
    }
    for (int i = 0; i < n; i++) {
        int bx = (int)floor((px[i] - ox) / bw), by = (int)floor((py[i] - oy) / bw);
        rbx[i] = bx;
        rby[i] = by;
        int r = abs(bx), t;
        t = abs(bx - (BX - 1)); if (t > r) r = t;
        t = abs(by);            if (t > r) r = t;
        t = abs(by - (BY - 1)); if (t > r) r = t;
        rmax[i] = r;
        mrow[i] = -1;
    }
    for (int j = 0; j < m; j++) msite[j] = -1;

    long long pops = 0, relax = 0, rows_scanned = 0, rings = 0, maxheap = 0, pathlen = 0;
    int stamp = 0;

    for (int it = 0; it < n; it++) {
        int i0 = order ? order[it] : it;
        stamp++;
        h.n = 0;
        int nsr = 0, nss = 0;
        drow[i0] = 0.0;
        srow[i0] = stamp;
        if (heap_push(&h, 0.0, i0, 0, 0)) { rc = -3; goto done2; }
        int target = -1;
        double D = 0.0;
        double UB = INFINITY; /* best tentative distance to any free site */
        while (h.n > 0) {
            if (h.n > maxheap) maxheap = h.n;
            Ev ev = heap_pop(&h);
            pops++;
            if (ev.type == 1) { /* SITE */
                int j = ev.a;
                if (fsite[j] == stamp || ev.key > dsite[j]) continue;
                fsite[j] = stamp;
                scan_s[nss++] = j;
                if (msite[j] < 0) { target = j; D = ev.key; break; }
                int ip = msite[j];
                if (ev.key < UB && (srow[ip] != stamp || ev.key < drow[ip])) {
                    srow[ip] = stamp;
                    drow[ip] = ev.key;
                    if (heap_push(&h, ev.key, ip, 0, 0)) { rc = -3; goto done2; }
                }
            } else if (ev.type == 0) { /* ROW */
                int i = ev.a;
                if (frow[i] == stamp || ev.key > drow[i]) continue;
                frow[i] = stamp;
                scan_r[nsr++] = i;
                rows_scanned++;
                if (heap_push(&h, drow[i], i, 0, 2)) { rc = -3; goto done2; }
            } else if (ev.type == 2) { /* BRING */
                int i = ev.a, R = ev.b;
                rings++;
                double di = drow[i], xi = px[i], yi = py[i], ui = u[i];
                int cbx = rbx[i], cby = rby[i];
                for (int dy = -R; dy <= R; dy++) {
                    int yy = cby + dy;
                    if (yy < 0 || yy >= BY) continue;
                    int step = (dy == -R || dy == R) ? 1 : 2 * R;
                    for (int dx = -R; dx <= R; dx += (step ? step : 1)) {
                        int xx = cbx + dx;
                        if (xx >= 0 && xx < BX) {
                            int b = yy * BX + xx;
                            if (bstart[b] < bstart[b + 1]) {
                                double x0 = ox + xx * bw, x1 = x0 + bw;
                                double y0 = oy + yy * bw, y1 = y0 + bw;
                                double gx = xi < x0 ? x0 - xi : (xi > x1 ? xi - x1 : 0.0);
                                double gy = yi < y0 ? y0 - yi : (yi > y1 ? yi - y1 : 0.0);
                                double lb = gx + gy - ui - vmax[b];
                                if (lb < 0) lb = 0;
                                if (di + lb < UB && heap_push(&h, di + lb, i, b, 3)) { rc = -3; goto done2; }
                            }
                        }
                        if (R == 0) break;
                    }
                }
                if (R + 1 <= rmax[i]) {
                    double lb = (double)R * bw - ui;
                    if (lb < 0) lb = 0;
                    if (di + lb < UB && heap_push(&h, di + lb, i, R + 1, 2)) { rc = -3; goto done2; }
                }
            } else { /* BLOCK */
                int i = ev.a, b = ev.b;
                double di = drow[i], xi = px[i], yi = py[i], ui = u[i];
                int own = mrow[i];
                for (int q = bstart[b]; q < bstart[b + 1]; q++) {
                    int j = bsites[q];
                    if (fsite[j] == stamp || j == own) continue;
                    double red = fabs(xi - sx[j]) + fabs(yi - sy[j]) - ui - v[j];
                    if (red < 0) red = 0;
                    double nd = di + red;
                    relax++;
                    if (nd >= UB) continue;
                    if (ssite[j] != stamp || nd < dsite[j]) {
                        ssite[j] = stamp;
                        dsite[j] = nd;
                        pred[j] = i;
                        if (msite[j] < 0) UB = nd;
                        if (heap_push(&h, nd, j, 0, 1)) { rc = -3; goto done2; }
                    }
                }
            }
        }
        if (target < 0) { rc = -4; goto done2; }
        int j = target;
        for (;;) {
            int i = pred[j];
            int nj = mrow[i];
            mrow[i] = j;
            msite[j] = i;
            pathlen++;
            if (i == i0) break;
            j = nj;
        }
        for (int q = 0; q < nsr; q++) {
            int i = scan_r[q];
            u[i] += D - drow[i];
        }
        int nd_ = 0;
        for (int q = 0; q < nss; q++) {
            int s = scan_s[q];
            if (dsite[s] < D) {
                v[s] -= D - dsite[s];
                int b = sblock[s];
                if (bdirty[b] != stamp) { bdirty[b] = stamp; dirty_list[nd_++] = b; }
            }
        }
        for (int q = 0; q < nd_; q++) {
            int b = dirty_list[q];
            double best = -INFINITY;
            for (int k = bstart[b]; k < bstart[b + 1]; k++)
                if (v[bsites[k]] > best) best = v[bsites[k]];
            vmax[b] = best;
        }
    }
    for (int i = 0; i < n; i++) out_assign[i] = mrow[i];
    if (stats) {
        stats[0] = pops; stats[1] = relax; stats[2] = rows_scanned;
        stats[3] = rings; stats[4] = maxheap; stats[5] = pathlen;
    }
done2:
    free(bstart); free(bsites); free(fill); free(sblock); free(vmax); free(bdirty); free(dirty_list);
    free(u); free(v); free(mrow); free(msite); free(drow); free(dsite); free(srow); free(ssite);
    free(frow); free(fsite); free(pred); free(rbx); free(rby); free(rmax); free(scan_r); free(scan_s);
    free(h.e);
    return rc;
}
