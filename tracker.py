import cv2
import numpy as np


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _quad_area(corners):
    return abs(cv2.contourArea(corners.reshape(-1, 2).astype(np.float32)))


class FeatureTracker:
    """Трекинг матчингом дескрипторов с эталоном (baseline).

    SIFT/ORB -> BFMatcher + ratio test -> findHomography(RANSAC) -> валидация.
    Эталон строится в нескольких масштабах для устойчивости к отдалению.
    """

    def __init__(self, detector="SIFT", nfeatures=4000, contrast=0.03,
                 scales=(1.0, 0.5, 0.25, 0.15), ratio=0.8,
                 min_matches=8, min_inliers=6, ransac_thresh=5.0,
                 min_area_ratio=0.0005, max_area_ratio=8.0):
        if detector == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast)
            norm = cv2.NORM_L2
        else:
            self.detector = cv2.ORB_create(nfeatures=nfeatures)
            norm = cv2.NORM_HAMMING
        self.matcher = cv2.BFMatcher(norm)
        self.scales = scales
        self.ratio = ratio
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.refs = []

    def set_reference(self, ref_image):
        gray = _gray(ref_image)
        self.refs = []
        for s in self.scales:
            r = cv2.resize(gray, None, fx=s, fy=s) if s != 1.0 else gray
            kp, des = self.detector.detectAndCompute(r, None)
            h, w = r.shape
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2) / s
            self.refs.append({"scale": s, "kp": kp, "des": des, "corners": corners})
        return len(self.refs[0]["kp"])

    def detect(self, frame):
        """Возвращает (corners, inliers, H) лучшей гомографии или (None, 0, None)."""
        gray = _gray(frame)
        kp, des = self.detector.detectAndCompute(gray, None)
        if des is None or len(kp) < 4:
            return None, 0, None
        best = (None, 0, None)
        for ref in self.refs:
            matches = self.matcher.knnMatch(ref["des"], des, k=2)
            good = [m for mn in matches if len(mn) == 2
                    for m in [mn[0]] if m.distance < self.ratio * mn[1].distance]
            if len(good) < self.min_matches:
                continue
            src = np.float32([ref["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
            if H is None:
                continue
            inliers = int(mask.sum())
            if inliers < self.min_inliers:
                continue
            corners = cv2.perspectiveTransform(ref["corners"], H)
            pts = corners.reshape(-1, 2)
            if not np.isfinite(pts).all() or not cv2.isContourConvex(pts.astype(np.float32)):
                continue
            area = _quad_area(corners)
            if area < self.min_area_ratio * gray.size or area > self.max_area_ratio * gray.size:
                continue
            if inliers > best[1]:
                best = (corners, inliers, H)
        return best

    def track(self, frame):
        corners, inliers, _ = self.detect(frame)
        return corners, {"method": "sift", "inliers": inliers}


class LKTracker:
    """Трекинг ключевых точек оптическим потоком Lucas-Kanade + гомография.

    Точки Shi-Tomasi трекаются LK между кадрами (с forward-backward проверкой),
    гомография эталон->кадр оценивается по RANSAC. Пере-якоризация (перевыбор
    точек внутри текущей рамки) устраняет дрейф. При потере объекта выполняется
    восстановление через FeatureTracker (SIFT-матчинг с эталоном).
    """

    def __init__(self, max_corners=800, quality=0.01, min_dist=5, min_pts=10,
                 reanchor_every=8, fb_thresh=1.0, win_size=25, max_level=4,
                 recover_with_sift=True):
        self.mc = max_corners
        self.q = quality
        self.md = min_dist
        self.min_pts = min_pts
        self.reanchor_every = reanchor_every
        self.fb_thresh = fb_thresh
        self.win = (win_size, win_size)
        self.max_level = max_level
        self.recover_with_sift = recover_with_sift
        self.sift = FeatureTracker() if recover_with_sift else None

    def set_reference(self, ref_image, first_frame=None, offset=(0, 0)):
        ref_gray = _gray(ref_image)
        self.first_full = ref_image if first_frame is None else first_frame
        self.prev_gray = _gray(self.first_full)
        pts = cv2.goodFeaturesToTrack(ref_gray, self.mc, self.q, self.md)
        self.ref_pts = pts.reshape(-1, 1, 2).astype(np.float32)
        self.cur_pts = (pts + np.array(offset, np.float32)).astype(np.float32)
        h, w = ref_gray.shape
        self.ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        self.corners = self.ref_corners + np.array(offset, np.float32)
        self.H = np.eye(3)
        self.frame_idx = 0
        self.lost_streak = 0
        if self.sift is not None:
            self.sift.set_reference(ref_image)
        return len(self.ref_pts)

    def _reanchor(self, gray):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(mask, [self.corners.astype(np.int32).reshape(-1, 2)], 255)
        new = cv2.goodFeaturesToTrack(gray, self.mc, self.q, self.md, mask=mask)
        if new is not None and len(new) >= self.min_pts:
            ref = cv2.perspectiveTransform(new, np.linalg.inv(self.H))
            self.ref_pts = ref.astype(np.float32)
            self.cur_pts = new.astype(np.float32)

    def _lk_step(self, gray):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.cur_pts,
                                              None, winSize=self.win, maxLevel=self.max_level)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, nxt,
                                                None, winSize=self.win, maxLevel=self.max_level)
        fb = np.linalg.norm(back - self.cur_pts, axis=2).ravel()
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_thresh)
        return nxt, good

    def _valid_update(self, corners, gray):
        pts = corners.reshape(-1, 2)
        if not np.isfinite(pts).all() or not cv2.isContourConvex(pts.astype(np.float32)):
            return False
        area = _quad_area(corners)
        prev_area = max(_quad_area(self.corners), 1.0)
        if not (0.4 < area / prev_area < 2.5):
            return False
        if area < 0.0005 * gray.size:
            return False
        return True

    def track(self, frame):
        gray = _gray(frame)
        self.frame_idx += 1
        nxt, good = self._lk_step(gray)
        corners = None
        method = "lk"
        inliers = 0
        if good.sum() >= self.min_pts:
            ref = self.ref_pts[good]
            cur = nxt[good]
            H, mask = cv2.findHomography(ref, cur, cv2.RANSAC, 4.0)
            if H is not None and mask is not None and mask.sum() >= self.min_pts:
                c = cv2.perspectiveTransform(self.ref_corners, H)
                if self._valid_update(c, gray):
                    corners = c
                    self.H = H
                    self.corners = c
                    inliers = int(mask.sum())
                    inl = mask.ravel() == 1
                    self.ref_pts = ref[inl]
                    self.cur_pts = cur[inl]
        if corners is not None:
            self.lost_streak = 0
            self.prev_gray = gray
            if self.frame_idx % self.reanchor_every == 0:
                self._reanchor(gray)
        else:
            # LK не дал валидной рамки — продолжаем трекать точки и пробуем SIFT
            self.lost_streak += 1
            if good.sum() >= self.min_pts:
                self.cur_pts = nxt[good]
                self.ref_pts = self.ref_pts[good]
            if self.sift is not None:
                c, inl, H_sift = self.sift.detect(frame)
                if c is not None:
                    self.H = H_sift
                    self.corners = c
                    self._reanchor(gray)
                    corners = c
                    method = "sift"
                    inliers = inl
                    self.lost_streak = 0
            self.prev_gray = gray
        stats = {"method": method if corners is not None else "none",
                 "inliers": inliers, "tracked_pts": int(good.sum())}
        return corners, stats


def draw_tracking(frame, corners, label="Object", color=(0, 255, 0), thickness=2):
    out = frame.copy()
    pts = corners.astype(np.int32).reshape(-1, 2)
    cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA)
    x, y = pts[0]
    cv2.putText(out, label, (int(x), max(int(y) - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out


def process_video(path, tracker, out_path=None, max_frames=None, label="Object",
                  ref_crop=None, sample_frames=()):
    cap = cv2.VideoCapture(path)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read {path}")
    if ref_crop is None:
        ref = first
        offset = (0, 0)
    else:
        ref = first[ref_crop[1]:ref_crop[3], ref_crop[0]:ref_crop[2]]
        offset = (ref_crop[0], ref_crop[1])
    if isinstance(tracker, LKTracker):
        n_ref = tracker.set_reference(ref, first, offset)
    else:
        n_ref = tracker.set_reference(ref)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    if out_path:
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    stats_list = []
    samples = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if max_frames and idx > max_frames:
            break
        corners, stats = tracker.track(frame)
        found = corners is not None
        vis = draw_tracking(frame, corners, label) if found else frame.copy()
        if not found:
            cv2.putText(vis, "not found", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2, cv2.LINE_AA)
        stats["frame"] = idx
        stats["found"] = found
        stats_list.append(stats)
        if writer:
            writer.write(vis)
        if idx in sample_frames:
            samples[idx] = vis
    cap.release()
    if writer:
        writer.release()
    return {"ref_keypoints": n_ref, "stats": stats_list, "samples": samples, "fps": fps}
