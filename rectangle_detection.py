import math
import cv2
import numpy as np
from collections import defaultdict

# If skeleton path finds fewer than this many rectangles, use contour-based fallback
FALLBACK_MIN_RECTANGLES = 3
# When skeleton succeeds, merge contour rects that don't overlap skeleton (restore thick-line detections)
MERGE_CONTOUR_OVERLAP_THRESHOLD = 0.6  # skip contour rect if overlap with any skeleton rect > this (0.5=original; 0.55 reduces duplicate boxes)
# Segment merge: same logical line = same orientation, row/col within tol, overlapping extent (reduces duplicate segments)
MERGE_POSITION_TOL = 2  # px; horizontal: merge skeleton+morph same line
MERGE_POSITION_TOL_VERTICAL = 0  # vertical: only merge segments at same x (do not merge adjacent column boundaries); preserves vertical divisions
MERGE_EXTENT_GAP = 2    # px; baseline max gap between segment extents to merge
# Path simplification; lower = segment endpoints follow path more closely (reduces misfitting)
APPROX_POLYGON_TOLERANCE = 2.5  # 3.0=original; 2.5=tighter, less endpoint drift
# Adaptive extent gap: scale with polygon tolerance so merge budget covers endpoint drift (Issue 2)
ADAPTIVE_EXTENT_GAP = max(MERGE_EXTENT_GAP, int(math.ceil(APPROX_POLYGON_TOLERANCE * 1.5)))  # currently 4


def _detect_rectangles_contour_based(
    roi: np.ndarray, fx: int, fy: int, page_id, adaptive_thresh: bool,
    img_w: int, img_h: int,
) -> list:
    """
    Original contour-based rectangle detection (identical params/optimisations to pre-skeleton version).
    Uses full image dimensions (img_w, img_h) for block_size, min_area, and oversized rejection.
    Returns list of rect dicts (same format as skeleton path).
    """
    roi_h, roi_w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if adaptive_thresh:
        block_size = max(15, min(img_w, img_h) // 15) | 1
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 1,
        )
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Minimum contour area as a fraction of page area.
    min_area = (img_w * img_h) * 0.0001
    rectangles = []

    for cnt in contours:
        epsilon = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        accepted = False
        if len(approx) == 4:
            rect = cv2.boundingRect(approx)
            accepted = True
        elif len(approx) in (5, 6):
            epsilon2 = 0.02 * cv2.arcLength(cnt, True)
            approx2 = cv2.approxPolyDP(cnt, epsilon2, True)
            if len(approx2) == 4:
                rect = cv2.boundingRect(approx2)
                accepted = True
        if not accepted:
            rot_rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rot_rect)
            box = np.intp(box)
            area_box = cv2.contourArea(box)
            rectangularity = area / (area_box + 1e-6)
            if area_box > min_area and rectangularity > 0.86:
                side_lengths = np.sqrt(((box - np.roll(box, 1, axis=0)) ** 2).sum(axis=1))
                aspect_ratio = max(side_lengths) / (side_lengths.min() + 1e-6)
                if side_lengths.min() >= 8 and aspect_ratio < 12:
                    rect = cv2.boundingRect(box)
                    accepted = True
        if not accepted:
            continue

        x, y, rw, rh = rect
        if rw * rh > 0.5 * (img_w * img_h):
            continue
        crop = roi[y : y + rh, x : x + rw].copy()
        bbox_global = (x + fx, y + fy, rw, rh)
        # Match original: store approx when 4-vertex, else raw contour (for drawing / downstream)
        contour_pts = approx if len(approx) == 4 else cnt
        rectangles.append({
            "contour": contour_pts,
            "bbox": (x, y, rw, rh),
            "area": area,
            "crop": crop,
            "bbox_global": bbox_global,
            "page_id": page_id,
            "poly_id": f"POLY_{len(rectangles)}",
        })

    for idx, r in enumerate(rectangles):
        r["poly_id"] = f"POLY_{idx}"
    return rectangles


def _bbox_intersection_area(bbox_a: tuple, bbox_b: tuple) -> float:
    """Intersection area of two axis-aligned boxes (x, y, w, h)."""
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return float(ix * iy)


def _merge_contour_rectangles_into_skeleton(
    skeleton_rectangles: list, contour_rectangles: list, overlap_threshold: float
) -> list:
    """
    Add contour-derived rectangles that don't heavily overlap skeleton-derived ones.
    Restores thick-line/component detections (valve outlines, flanges) while keeping
    skeleton detections (thin grid lines, table cells). Avoids duplicate table cells.
    """
    out = list(skeleton_rectangles)
    for c in contour_rectangles:
        cb = c["bbox"]
        ca = cb[2] * cb[3]
        if ca <= 0:
            continue
        max_overlap_ratio = 0.0
        for s in skeleton_rectangles:
            inter = _bbox_intersection_area(cb, s["bbox"])
            max_overlap_ratio = max(max_overlap_ratio, inter / ca)
        if max_overlap_ratio > overlap_threshold:
            continue
        out.append(c)
    for idx, r in enumerate(out):
        r["poly_id"] = f"POLY_{idx}"
    return out


def _estimate_line_thickness(roi_gray: np.ndarray) -> float:
    """
    Roughly estimate dominant line thickness (in pixels) for this ROI.
    Uses an Otsu binary + distance transform; returns median stroke width.
    """
    # Invert so lines are foreground (white) for distanceTransform
    _, main = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg = (main > 0).astype("uint8")
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    if vals.size == 0:
        return 2.0
    # Stroke width ~ 2 * distance from center to edge
    thickness = float(np.median(vals) * 2.0)
    # Clamp to a reasonable range
    return max(1.5, min(thickness, 12.0))


def _binarize_robust(roi_gray: np.ndarray, adaptive_thresh: bool) -> np.ndarray:
    """
    Binarize so both thick and thin lines are picked up consistently across the image.
    ORs a main binary with a lower-threshold binary so faint lines in low-contrast regions
    (e.g. right side of BOM) and inconsistently thresholded thin lines still appear.
    """
    h, w = roi_gray.shape[:2]
    line_thickness = _estimate_line_thickness(roi_gray)
    aspect = max(w / max(h, 1), h / max(w, 1))
    if adaptive_thresh:
        # Base block size on line thickness and aspect ratio: larger for big pages,
        # but scaled so thin lines still get contrast.
        base_block = int(max(9, line_thickness * 4.0)) | 1
        max_block = max(15, min(w, h) // 5) | 1
        block_size = min(base_block, max_block)
        block_size = max(15, block_size) | 1
        main = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 1,
        )
    else:
        _, main = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Second pass: lower threshold so faint lines (same thickness but different contrast) survive
    otsu_val, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Scale low threshold based on line thickness and aspect ratio:
    # thinner lines + extreme aspect ratio → slightly more permissive.
    scale = 1.15
    if line_thickness < 3.0:
        scale += 0.05
    if aspect > 2.0:
        scale += 0.05
    low_thresh = min(254, max(1, int(otsu_val * scale)))
    _, low_binary = cv2.threshold(roi_gray, low_thresh, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_or(main, low_binary)
    # Only dilate if lines are genuinely thin (< 1.5px); else dilation offsets skeleton inward (Issue 3a).
    if line_thickness < 1.5:
        k = int(round(max(2.0, min(line_thickness, 4.0))))
        kernel = np.ones((k, k), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def _skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    """Skeleton (medial axis) from already binarized image."""
    binary_bool = binary.astype(bool)
    from skimage.morphology import skeletonize
    skeleton = skeletonize(binary_bool)
    return np.asarray(skeleton, dtype=bool)


def _generate_skeleton(roi_gray: np.ndarray, adaptive_thresh: bool = False) -> np.ndarray:
    """Skeleton from ROI gray; uses _binarize_robust then _skeletonize_binary."""
    binary = _binarize_robust(roi_gray, adaptive_thresh)
    return _skeletonize_binary(binary)


def _median_vertical_spacing_horizontal_segments(segments: list):
    """
    Median vertical spacing between horizontal segment y-centers (typical cell height).
    Used to set klen_v adaptively so short vertical boundaries are not suppressed (Issue 3b).
    Returns None if fewer than 2 horizontal segments.
    """
    h_segs = [s for s in segments if s.get("orientation") == "horizontal"]
    if len(h_segs) < 2:
        return None
    y_centers = sorted((s["y1"] + s["y2"]) / 2 for s in h_segs)
    gaps = [y_centers[i + 1] - y_centers[i] for i in range(len(y_centers) - 1) if y_centers[i + 1] > y_centers[i]]
    if not gaps:
        return None
    return float(np.median(gaps))


def _morphological_line_segments(
    binary: np.ndarray, roi_h: int, roi_w: int, median_vertical_spacing=None
) -> list:
    """
    Extract line segments from binary using morphological open (horizontal + vertical kernels).
    Catches lines skeleton may miss (e.g. right side of BOM, low-contrast regions).
    Returns segment dicts in same format as _paths_to_hv_segments for merging.
    If median_vertical_spacing is provided, klen_v is derived from it (30–40% of cell height, floor 6px) for Issue 3b.
    """
    w, h = roi_w, roi_h
    min_len = 15
    aspect = 2.5
    segments = []
    # Horizontal lines; clip each segment to actual foreground run (reduces misfitting/overlong lines)
    klen_h = max(20, w // 25)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (klen_h, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    for i in range(1, num_labels):
        x, y, kw, kh, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
        if kw < min_len or kh < 1:
            continue
        if kw < aspect * kh:
            continue
        # Clip to actual foreground run so segment doesn't extend beyond the line
        rows, cols = np.where(labels == i)
        if rows.size and cols.size:
            x_min, x_max = int(cols.min()), int(cols.max())
            y_center = int(round(rows.mean()))
            seg_len = max(x_max - x_min, 1)
            segments.append({
                "x1": x_min, "y1": y_center,
                "x2": x_max, "y2": y_center,
                "orientation": "horizontal",
                "length": float(seg_len),
            })
    # Vertical lines; klen_v adaptive from median cell height only when valid (Issue 3b). Degenerate input
    # (median_vertical_spacing < 3px from within-line skeleton fragmentation) falls back to height-based.
    if median_vertical_spacing is not None and median_vertical_spacing >= 3.0:
        klen_v = max(6, min(int(math.ceil(median_vertical_spacing * 0.35)), h // 5))
    else:
        klen_v = max(20, h // 25)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen_v))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    for i in range(1, num_labels):
        x, y, kw, kh, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
        if kh < min_len or kw < 1:
            continue
        if kh < aspect * kw:
            continue
        rows, cols = np.where(labels == i)
        if rows.size and cols.size:
            y_min, y_max = int(rows.min()), int(rows.max())
            x_center = int(round(cols.mean()))
            seg_len = max(y_max - y_min, 1)
            segments.append({
                "x1": x_center, "y1": y_min,
                "x2": x_center, "y2": y_max,
                "orientation": "vertical",
                "length": float(seg_len),
            })
    return segments


def _trace_skeleton_paths(skeleton: np.ndarray) -> list:
    """
    Trace continuous paths through the skeleton. Traces all branches at junctions
    (no dropped stubs) so thin lines that meet at T-junctions are consistently extracted.
    """
    h, w = skeleton.shape
    visited = np.zeros((h, w), dtype=bool)
    paths = []
    to_trace = []

    def count_neighbors(x, y):
        c = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx]:
                    c += 1
        return c

    endpoints = []
    junctions = []
    for y in range(h):
        for x in range(w):
            if not skeleton[y, x]:
                continue
            n = count_neighbors(x, y)
            if n == 1:
                endpoints.append((x, y))
            elif n >= 3:
                junctions.append((x, y))

    def trace_from(start_x, start_y):
        path = [(start_x, start_y)]
        visited[start_y, start_x] = True
        current = (start_x, start_y)
        while True:
            x, y = current
            unvisited = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] and not visited[ny, nx]:
                        unvisited.append((nx, ny))
            if not unvisited:
                break
            # Trace all branches: queue the rest for later, continue with one
            for i in range(1, len(unvisited)):
                to_trace.append(unvisited[i])
            next_pt = unvisited[0]
            path.append(next_pt)
            visited[next_pt[1], next_pt[0]] = True
            current = next_pt
        return path if len(path) >= 2 else None

    to_trace = list(endpoints)
    for (jx, jy) in junctions:
        if not visited[jy, jx]:
            visited[jy, jx] = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = jx + dx, jy + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] and not visited[ny, nx]:
                    to_trace.append((nx, ny))

    while to_trace:
        (ex, ey) = to_trace.pop()
        if visited[ey, ex]:
            continue
        path = trace_from(ex, ey)
        if path and len(path) >= 2:
            paths.append(path)

    return paths


def _paths_to_hv_segments(paths: list) -> list:
    """Convert traced paths to segments; keep only horizontal or vertical (reject diagonal). No length filter."""
    from skimage.measure import approximate_polygon

    segments = []
    for path in paths:
        path_array = np.array(path)
        simplified = approximate_polygon(path_array, tolerance=APPROX_POLYGON_TOLERANCE)
        for i in range(len(simplified) - 1):
            x1, y1 = simplified[i][0], simplified[i][1]
            x2, y2 = simplified[i + 1][0], simplified[i + 1][1]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > dy * 1.5:
                orientation = "horizontal"
            elif dy > dx * 1.5:
                orientation = "vertical"
            else:
                continue
            length = math.hypot(x2 - x1, y2 - y1)
            segments.append({
                "x1": int(round(x1)), "y1": int(round(y1)),
                "x2": int(round(x2)), "y2": int(round(y2)),
                "orientation": orientation,
                "length": length,
            })
    return segments


def _merge_coincident_segments(
    segments: list,
    position_tol: int = MERGE_POSITION_TOL,
    extent_gap=None,
) -> list:
    """
    Merge segments that represent the same logical line (same orientation, same row/col within
    position_tol px, overlapping or adjacent extent). Prevents duplicate segments (skeleton+morph)
    so we don't get nested white regions and duplicate rectangles.
    extent_gap: if None, computed per-call from segment geometry so dense grids (rows a few px apart)
    do not get cross-row merges; otherwise uses the provided value (e.g. 2 for tests).
    """
    if not segments:
        return []
    out = []
    # Horizontal: merge by (y, x range)
    h_segs = [s for s in segments if s["orientation"] == "horizontal"]
    h_segs.sort(key=lambda s: (round((s["y1"] + s["y2"]) / 2), min(s["x1"], s["x2"])))

    # Per-call extent_gap from row spacing so we don't merge across distinct rows (Cause 1 regression)
    if extent_gap is None:
        y_buckets = sorted(set(round((s["y1"] + s["y2"]) / 2 / position_tol) * position_tol for s in h_segs))
        if len(y_buckets) >= 2:
            min_row_spacing = min(y_buckets[i + 1] - y_buckets[i] for i in range(len(y_buckets) - 1))
        else:
            min_row_spacing = 999
        if min_row_spacing < 6:
            extent_gap = MERGE_EXTENT_GAP  # dense grid: do not risk cross-row merge
        else:
            extent_gap = min(ADAPTIVE_EXTENT_GAP, max(MERGE_EXTENT_GAP, min_row_spacing // 2 - 1))
    extent_gap = int(extent_gap)
    i = 0
    while i < len(h_segs):
        s = h_segs[i]
        y_center = (s["y1"] + s["y2"]) / 2
        x_min, x_max = min(s["x1"], s["x2"]), max(s["x1"], s["x2"])
        j = i + 1
        while j < len(h_segs):
            t = h_segs[j]
            y_center_t = (t["y1"] + t["y2"]) / 2
            if abs(y_center_t - y_center) > position_tol:
                break
            tx_min, tx_max = min(t["x1"], t["x2"]), max(t["x1"], t["x2"])
            if tx_min <= x_max + extent_gap and tx_max >= x_min - extent_gap:
                x_min = min(x_min, tx_min)
                x_max = max(x_max, tx_max)
                j += 1
            else:
                break
        out.append({
            "x1": int(round(x_min)), "y1": int(round(y_center)),
            "x2": int(round(x_max)), "y2": int(round(y_center)),
            "orientation": "horizontal",
            "length": max(x_max - x_min, 1),
        })
        i = j

    v_segs = [s for s in segments if s["orientation"] == "vertical"]
    v_segs.sort(key=lambda s: (round((s["x1"] + s["x2"]) / 2), min(s["y1"], s["y2"])))
    v_tol = MERGE_POSITION_TOL_VERTICAL  # strict so we don't merge adjacent column boundaries (vertical lines removed / oversimplification)
    i = 0
    while i < len(v_segs):
        s = v_segs[i]
        x_center = (s["x1"] + s["x2"]) / 2
        y_min, y_max = min(s["y1"], s["y2"]), max(s["y1"], s["y2"])
        j = i + 1
        while j < len(v_segs):
            t = v_segs[j]
            x_center_t = (t["x1"] + t["x2"]) / 2
            if abs(x_center_t - x_center) > v_tol:
                break
            ty_min, ty_max = min(t["y1"], t["y2"]), max(t["y1"], t["y2"])
            if ty_min <= y_max + extent_gap and ty_max >= y_min - extent_gap:
                y_min = min(y_min, ty_min)
                y_max = max(y_max, ty_max)
                j += 1
            else:
                break
        out.append({
            "x1": int(round(x_center)), "y1": int(round(y_min)),
            "x2": int(round(x_center)), "y2": int(round(y_max)),
            "orientation": "vertical",
            "length": max(y_max - y_min, 1),
        })
        i = j

    return out


# Vertical segments drawn thicker so they don't vanish at T-junctions (preserves column boundaries on 01_original).
# Horizontals stay 1 so we don't over-thicken and break detection on images like 1759857039776.
LINE_MASK_THICKNESS_HORIZONTAL = 1
LINE_MASK_THICKNESS_VERTICAL = 2


def _segments_to_line_mask(segments: list, h: int, w: int, line_thickness: int = 1) -> np.ndarray:
    """Draw segments as black lines on white background for finding enclosed regions.
    Orientation is derived from geometry (dy >= dx => vertical) so verticals always get thickness 2
    and don't disappear at T-junctions; horizontals stay 1 to avoid over-thick mask on other images."""
    line_image = np.ones((h, w), dtype=np.uint8) * 255
    for seg in segments:
        x1, y1 = seg["x1"], seg["y1"]
        x2, y2 = seg["x2"], seg["y2"]
        pt1 = (max(0, min(x1, w - 1)), max(0, min(y1, h - 1)))
        pt2 = (max(0, min(x2, w - 1)), max(0, min(y2, h - 1)))
        # Use geometry so verticals always get thickness 2 even if orientation was missing/wrong
        dy, dx = abs(y2 - y1), abs(x2 - x1)
        is_vertical = dy >= dx
        th = LINE_MASK_THICKNESS_VERTICAL if is_vertical else LINE_MASK_THICKNESS_HORIZONTAL
        cv2.line(line_image, pt1, pt2, 0, thickness=max(1, th))
    return line_image


def _identify_rectangles_from_segments(segments: list, roi_h: int, roi_w: int) -> list:
    """
    Identify rectangles as enclosed regions: draw segments, find white regions, then apply
    the same geometric classification as the contour path so we don't accept non-rectangular
    shapes (L-shapes, slits, etc.). Only accept regions that pass 4-vertex approx or
    rectangularity + aspect_ratio + min-side checks.
    """
    if not segments:
        return []
    line_mask = _segments_to_line_mask(segments, roi_h, roi_w)
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Minimum contour area as a fraction of ROI area.
    min_area = (roi_w * roi_h) * 0.0001
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # Same classification as contour path: only accept if shape is rectangular enough
        epsilon = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        accepted = False
        rect = None
        if len(approx) == 4:
            rect = cv2.boundingRect(approx)
            accepted = True
        elif len(approx) in (5, 6):
            epsilon2 = 0.02 * cv2.arcLength(cnt, True)
            approx2 = cv2.approxPolyDP(cnt, epsilon2, True)
            if len(approx2) == 4:
                rect = cv2.boundingRect(approx2)
                accepted = True
        if not accepted:
            rot_rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rot_rect)
            box = np.intp(box)
            area_box = cv2.contourArea(box)
            rectangularity = area / (area_box + 1e-6)
            if area_box > min_area and rectangularity > 0.86:
                side_lengths = np.sqrt(((box - np.roll(box, 1, axis=0)) ** 2).sum(axis=1))
                aspect_ratio = max(side_lengths) / (side_lengths.min() + 1e-6)
                if side_lengths.min() >= 8 and aspect_ratio < 12:
                    rect = cv2.boundingRect(box)
                    accepted = True
        if not accepted or rect is None:
            continue
        x, y, rw, rh = rect
        if rw <= 0 or rh <= 0:
            continue
        if rw * rh > 0.5 * (roi_w * roi_h):
            continue
        rects.append((x, y, rw, rh))
    return rects


# --- Structural frames from line topology (table-like lattices), not containment ---
STRUCTURAL_FRAME_MIN_VERT = 4
STRUCTURAL_FRAME_MIN_HORIZ = 4
STRUCTURAL_FRAME_MIN_JUNCTIONS = 10
STRUCTURAL_FRAME_ENDPOINT_TOL = 8
STRUCTURAL_FRAME_RECT_OVERLAP_MIN = 0.6


def _segment_endpoints(seg: dict):
    """Return ((x1,y1), (x2,y2)) in consistent order for connectivity."""
    return ((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))


def _endpoints_near(a: tuple, b: tuple, tol: float) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _segments_connected(s1: dict, s2: dict, tol: float) -> bool:
    """True if any endpoint of s1 is within tol of any endpoint of s2."""
    e1a, e1b = _segment_endpoints(s1)
    e2a, e2b = _segment_endpoints(s2)
    return (
        _endpoints_near(e1a, e2a, tol)
        or _endpoints_near(e1a, e2b, tol)
        or _endpoints_near(e1b, e2a, tol)
        or _endpoints_near(e1b, e2b, tol)
    )


def _build_structural_frames_from_segments(
    segments: list, roi_h: int, roi_w: int, fx: int, fy: int
) -> list:
    """
    Build structural frames = connected systems of lines forming a rectangular lattice.
    Uses segment connectivity (shared endpoints); each connected component is a candidate frame.
    Filter: >= 4 vertical and >= 4 horizontal lines, or >= 10 junctions (shared endpoints).
    Returns list of frame dicts: poly_id, bbox (ROI), bbox_global.
    """
    if not segments:
        return []
    tol = STRUCTURAL_FRAME_ENDPOINT_TOL

    # Union-find: segment index -> parent index
    n = len(segments)
    parent = list(range(n))

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if _segments_connected(segments[i], segments[j], tol):
                union(i, j)

    comp_seg_ids = defaultdict(list)
    for i in range(n):
        comp_seg_ids[find(i)].append(i)

    frames = []
    for fi, (root, seg_ids) in enumerate(comp_seg_ids.items()):
        segs = [segments[i] for i in seg_ids]
        n_horiz = sum(1 for s in segs if s.get("orientation") == "horizontal")
        n_vert = sum(1 for s in segs if s.get("orientation") == "vertical")
        # Junctions: endpoints that appear in more than one segment (within tol)
        endpoints = []
        for s in segs:
            endpoints.append((s["x1"], s["y1"]))
            endpoints.append((s["x2"], s["y2"]))
        # Junctions = distinct points where >= 2 segment endpoints meet (within tol)
        point_count = defaultdict(int)
        for s in segs:
            for (x, y) in _segment_endpoints(s):
                key = (round(x / tol) * tol, round(y / tol) * tol)
                point_count[key] += 1
        n_junctions = sum(1 for c in point_count.values() if c >= 2)

        if not (
            (n_vert >= STRUCTURAL_FRAME_MIN_VERT and n_horiz >= STRUCTURAL_FRAME_MIN_HORIZ)
            or n_junctions >= STRUCTURAL_FRAME_MIN_JUNCTIONS
        ):
            continue

        xs, ys = [], []
        for s in segs:
            xs.extend([s["x1"], s["x2"]])
            ys.extend([s["y1"], s["y2"]])
        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(roi_w, max(xs))
        y2 = min(roi_h, max(ys))
        if x2 <= x1 or y2 <= y1:
            continue
        frames.append({
            "poly_id": f"FRAME_{fi}",
            "bbox": (x1, y1, x2, y2),
            "bbox_global": (x1 + fx, y1 + fy, x2 + fx, y2 + fy),
        })
    return frames


def _rect_frame_overlap_ratio(rect_bbox: tuple, frame_bbox_roi: tuple) -> float:
    """Rect bbox (x, y, w, h), frame bbox ROI (x1, y1, x2, y2). Return intersection area / rect area."""
    x, y, w, h = rect_bbox
    x1, y1, x2, y2 = frame_bbox_roi
    if w <= 0 or h <= 0:
        return 0.0
    ix = max(0, min(x + w, x2) - max(x, x1))
    iy = max(0, min(y + h, y2) - max(y, y1))
    return (ix * iy) / (w * h)


def _assign_rectangles_to_frames(
    rectangles: list, structural_frames: list
) -> dict:
    """rect index -> frame index for rects with overlap >= OVERLAP_MIN. Ties: max overlap."""
    out = {}
    for i, r in enumerate(rectangles):
        bbox = r.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        best_fi, best_ratio = None, 0.0
        for fi, frame in enumerate(structural_frames):
            fbbox = frame.get("bbox")
            if not fbbox or len(fbbox) != 4:
                continue
            ratio = _rect_frame_overlap_ratio(bbox, fbbox)
            if ratio >= STRUCTURAL_FRAME_RECT_OVERLAP_MIN and ratio > best_ratio:
                best_ratio = ratio
                best_fi = fi
        if best_fi is not None:
            out[i] = best_fi
    return out


def _frame_based_hierarchy(n_rects: int, rect_to_frame: dict) -> dict:
    """Build hierarchy dict: parent_idx (n_rects + frame_idx) -> list of rect indices in that frame."""
    hierarchy = defaultdict(list)
    for rect_idx, frame_idx in rect_to_frame.items():
        parent_idx = n_rects + frame_idx
        hierarchy[parent_idx].append(rect_idx)
    return dict(hierarchy)


def detect_rectangles(image, frame_rect=None, page_id=None, adaptive_thresh=True, debug=False, basename=None):
    """
    Detect rectangles via skeleton tracing (replaces contour-based detection).
    Same signature and return shape: rect_data (rectangles, hierarchy, clusters, page_id), vis.
    """
    h, w = image.shape[:2]
    if frame_rect:
        fx, fy, fw, fh = frame_rect
        roi = image[fy:fy+fh, fx:fx+fw]
    else:
        roi = image
        fx, fy = 0, 0

    roi_h, roi_w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = _binarize_robust(gray, adaptive_thresh)
    skeleton = _skeletonize_binary(binary)
    paths = _trace_skeleton_paths(skeleton)
    segments_skeleton = _paths_to_hv_segments(paths)
    median_spacing = _median_vertical_spacing_horizontal_segments(segments_skeleton)
    segments = list(segments_skeleton)
    segments.extend(_morphological_line_segments(binary, roi_h, roi_w, median_vertical_spacing=median_spacing))
    segments = _merge_coincident_segments(segments)  # one logical line per segment → avoids duplicate rectangles
    raw_rects = _identify_rectangles_from_segments(segments, roi_h, roi_w)
    rectangles = []
    for idx, (rx, ry, rw, rh) in enumerate(raw_rects):
        rx = max(0, min(rx, roi_w - 1))
        ry = max(0, min(ry, roi_h - 1))
        rw = min(rw, roi_w - rx)
        rh = min(rh, roi_h - ry)
        if rw <= 0 or rh <= 0:
            continue
        # Skip very thin rectangles (artifacts)
        if rw < 2 or rh < 2:
            continue
        area = rw * rh
        crop = roi[ry:ry+rh, rx:rx+rw].copy()
        bbox_global = (rx + fx, ry + fy, rw, rh)
        contour = np.array([[rx, ry], [rx+rw, ry], [rx+rw, ry+rh], [rx, ry+rh]], dtype=np.int32)
        rectangles.append({
            "contour": contour,
            "bbox": (rx, ry, rw, rh),
            "area": area,
            "crop": crop,
            "bbox_global": bbox_global,
            "page_id": page_id,
            "poly_id": f"POLY_{idx}",
        })

    # Fallback: if skeleton path found no or very few rectangles, use contour-based only
    if len(rectangles) < FALLBACK_MIN_RECTANGLES:
        rectangles = _detect_rectangles_contour_based(roi, fx, fy, page_id, adaptive_thresh, img_w=w, img_h=h)
    else:
        # Hybrid: keep skeleton rects (thin lines, cells) and add contour rects that don't overlap them
        # (restores thick-line detections: component outlines, flanges)
        contour_rects = _detect_rectangles_contour_based(roi, fx, fy, page_id, adaptive_thresh, img_w=w, img_h=h)
        rectangles = _merge_contour_rectangles_into_skeleton(
            rectangles, contour_rects, MERGE_CONTOUR_OVERLAP_THRESHOLD
        )

    # Structural frames from line topology (connected segment components), not containment.
    # Grid formation uses same_frame → allow merge; different_frame + large distance → veto.
    structural_frames = _build_structural_frames_from_segments(segments, roi_h, roi_w, fx, fy)
    rect_to_frame = _assign_rectangles_to_frames(rectangles, structural_frames)
    hierarchy = _frame_based_hierarchy(len(rectangles), rect_to_frame)

    clusters = []
    visited = set()
    prox_thresh = max(roi_w, roi_h) * 0.02

    for i, rect in enumerate(rectangles):
        if i in visited:
            continue
        cluster = [i]
        queue = [i]
        visited.add(i)
        rx, ry, rw, rh = rect["bbox"]
        while queue:
            current = queue.pop()
            x, y, cw, ch = rectangles[current]["bbox"]
            for j, other in enumerate(rectangles):
                if j in visited:
                    continue
                ox, oy, ow, oh = other["bbox"]
                if (
                    abs(x - ox) < prox_thresh or abs(y - oy) < prox_thresh
                    or abs(x + cw - ox - ow) < prox_thresh or abs(y + ch - oy - oh) < prox_thresh
                ):
                    cluster.append(j)
                    queue.append(j)
                    visited.add(j)
        clusters.append(cluster)

    classified = []
    for cluster in clusters:
        if not cluster:
            continue
        bboxes = [rectangles[i]["bbox"] for i in cluster]
        classified.append({
            "indices": cluster,
            "type": "cluster",
            "bboxes": bboxes,
            "page_id": page_id,
        })

    vis = roi.copy()
    for rect in rectangles:
        rx, ry, rw, rh = rect["bbox"]
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
    for group in classified:
        if not group["bboxes"]:
            continue
        xs = [x for x, _, _, _ in group["bboxes"]]
        ys = [y for _, y, _, _ in group["bboxes"]]
        ws = [w for _, _, w, _ in group["bboxes"]]
        hs = [h for _, _, _, h in group["bboxes"]]
        min_x = min(xs)
        max_x = max(x + w for x, w in zip(xs, ws))
        min_y = min(ys)
        max_y = max(y + h for y, h in zip(ys, hs))
        cv2.rectangle(vis, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    if basename:
        cv2.imwrite(f"{basename}_rect_overlay.png", vis)
    if debug:
        cv2.imshow("Rectangle Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "rectangles": rectangles,
        "hierarchy": hierarchy,
        "clusters": classified,
        "structural_frames": structural_frames,
        "page_id": page_id,
    }, vis