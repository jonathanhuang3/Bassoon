# -*- coding: utf-8 -*-
"""
Contrast Dots with a gaze-contingent left-eye nasal step.

The step is nasal to 10° (right of fixation) and sits on the horizontal
meridian. Its upper edge rises smoothly toward the nasal periphery.
Blocked Hemifield chooses superior ('upper') or inferior ('lower').
"""
from protocols._fieldDefectMask import NasalStepMask


class ContrastDotsWithNasalStepLeft(NasalStepMask):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithNasalStepLeft'
        self.eye = 'left'  # nasal field is right of fixation; temporal field is left
