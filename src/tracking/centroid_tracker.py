import math


class CentroidTracker:
    """
    Assigns a stable integer ID to each detected person by matching
    frame centroids to the closest tracked centroid from previous frames.
    Coordinates are normalized to [0, 1].
    Includes debouncing for brand-new candidate IDs to avoid false positives.

    Also tracks cumulative centroid displacement per person to distinguish
    mobile (walking) persons like examiners from stationary seated students.
    """

    def __init__(self, max_distance=0.15, max_missed_frames=20, min_confirmation_frames=3,
                 mobility_window=30, mobility_threshold=0.20):
        self.next_id = 0
        self.tracked = {}          # id -> (x, y) centroid, normalized
        self.missed_frames = {}    # id -> consecutive frames not matched
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

        self.min_confirmation_frames = min_confirmation_frames
        self.pending = {}          # pending_key -> {"centroid": (x,y), "count": int}
        self.pending_next_key = 0

        # Mobility tracking
        self.mobility_window = mobility_window          # number of recent frames to track
        self.mobility_threshold = mobility_threshold    # cumulative normalized displacement to flag as mobile
        self.centroid_history = {}                      # id -> deque-like list of (x, y)

    def _promote_pending(self, pending_key):
        new_id = self.next_id
        self.next_id += 1
        self.tracked[new_id] = self.pending[pending_key]["centroid"]
        self.missed_frames[new_id] = 0
        self.centroid_history[new_id] = []
        del self.pending[pending_key]
        return new_id

    def is_mobile(self, person_id):
        """
        Returns True if this person's cumulative centroid displacement over the
        last `mobility_window` frames exceeds `mobility_threshold`.
        Used to identify the mobile examiner walking around the room.
        """
        history = self.centroid_history.get(person_id, [])
        if len(history) < 2:
            return False
        total_disp = sum(
            math.sqrt((history[i][0] - history[i-1][0])**2 + (history[i][1] - history[i-1][1])**2)
            for i in range(1, len(history))
        )
        return total_disp >= self.mobility_threshold

    def update(self, detected_centroids):
        """
        detected_centroids: list of (x, y) tuples for this frame.
        Returns: dict mapping detected_centroid_index -> stable student ID.
        """
        assignments = {}
        used_ids = set()

        unmatched = []
        for i, (dx, dy) in enumerate(detected_centroids):
            best_id, best_dist = None, self.max_distance
            for pid, (px, py) in self.tracked.items():
                if pid in used_ids:
                    continue
                dist = math.sqrt((dx - px) ** 2 + (dy - py) ** 2)
                if dist < best_dist:
                    best_id, best_dist = pid, dist

            if best_id is not None:
                assignments[i] = best_id
                self.tracked[best_id] = (dx, dy)
                self.missed_frames[best_id] = 0
                used_ids.add(best_id)

                # Update mobility history
                h = self.centroid_history.setdefault(best_id, [])
                h.append((dx, dy))
                if len(h) > self.mobility_window:
                    h.pop(0)
            else:
                unmatched.append(i)

        still_unmatched = []
        for i in unmatched:
            dx, dy = detected_centroids[i]
            best_key, best_dist = None, self.max_distance
            for key, info in self.pending.items():
                px, py = info["centroid"]
                dist = math.sqrt((dx - px) ** 2 + (dy - py) ** 2)
                if dist < best_dist:
                    best_key, best_dist = key, dist

            if best_key is not None:
                self.pending[best_key]["centroid"] = (dx, dy)
                self.pending[best_key]["count"] += 1
                if self.pending[best_key]["count"] >= self.min_confirmation_frames:
                    new_id = self._promote_pending(best_key)
                    assignments[i] = new_id
                    used_ids.add(new_id)
            else:
                still_unmatched.append(i)

        for i in still_unmatched:
            key = self.pending_next_key
            self.pending_next_key += 1
            self.pending[key] = {"centroid": detected_centroids[i], "count": 1}

        matched_pending_keys = set()
        for i in unmatched:
            dx, dy = detected_centroids[i]
            for key, info in self.pending.items():
                px, py = info["centroid"]
                if math.sqrt((dx - px) ** 2 + (dy - py) ** 2) < self.max_distance:
                    matched_pending_keys.add(key)
                    break
        for key in list(self.pending.keys()):
            if key not in matched_pending_keys and self.pending[key]["count"] < self.min_confirmation_frames:
                del self.pending[key]

        for pid in list(self.tracked.keys()):
            if pid not in used_ids:
                self.missed_frames[pid] = self.missed_frames.get(pid, 0) + 1
                if self.missed_frames[pid] > self.max_missed_frames:
                    del self.tracked[pid]
                    del self.missed_frames[pid]
                    self.centroid_history.pop(pid, None)

        return assignments
