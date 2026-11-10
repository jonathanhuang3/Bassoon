# -*- coding: utf-8 -*-
"""
Contrast Dots with a gaze-contingent right-eye arcuate defect.

The arc starts at 15° temporal on the horizontal meridian, peaks about 10°
above fixation, and ends at 20° nasal. It is narrow at the temporal end
and wider nasally. Blocked Hemifield chooses superior ('upper') or inferior
('lower').
"""
from protocols._fieldDefectMask import ArcuateMask


class ContrastDotsWithArcuateRight(ArcuateMask):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithArcuateRight'
        self.eye = 'right'  # blind spot is right of fixation
