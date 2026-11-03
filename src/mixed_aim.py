"""Stable per-target head/body selection for the optional mixed aim mode."""


class MixedAimPolicy:
    """Choose head for one in three new track IDs; never alternate per frame."""

    def head_for(self, track_id):
        return int(track_id) % 3 == 0
