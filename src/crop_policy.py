"""Single-pass scheduling for distant-target crops and yellow candidates."""


def should_zoom_lock(box, frame_count, refresh_every=5):
    """Use a crop for a recent real small body; periodically refresh full view."""
    if box is None or refresh_every < 2 or frame_count % refresh_every == 0:
        return False
    _, _, width, height = map(float, box)
    return width >= 5 and 16 <= height <= 55


def choose_color_candidate(proposals, probe_index, lock_center=None):
    """Use one crop per frame; stick to a lock or rotate among top candidates."""
    if not proposals:
        return []
    if lock_center is not None:
        cx, cy = lock_center
        choice = min(proposals, key=lambda p: (p[0] + p[2] / 2 - cx) ** 2
                     + (p[1] + p[3] / 2 - cy) ** 2)
    else:
        choice = proposals[probe_index % len(proposals)]
    return [choice]
