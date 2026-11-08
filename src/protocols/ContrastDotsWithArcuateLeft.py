# -*- coding: utf-8 -*-
"""
Contrast Dots with a gaze-contingent left-eye arcuate defect.

The arc starts at 15° temporal on the horizontal meridian, arches through
the upper field, and ends at 22° nasal. It is narrow at the temporal end
and wider nasally. Blocked Hemifield chooses superior ('upper') or inferior
('lower').
"""
from protocols._fieldDefectMask import ArcuateMask


class ContrastDotsWithArcuateLeft(ArcuateMask):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithArcuateLeft'
        self.eye = 'left'  # blind spot is left of fixation
