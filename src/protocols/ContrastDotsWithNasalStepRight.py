# -*- coding: utf-8 -*-
"""
Contrast Dots with a gaze-contingent right-eye nasal step.

The step is nasal to 10° (left of fixation) and sits on the horizontal
meridian. Its upper edge rises smoothly toward the nasal periphery.
Blocked Hemifield chooses superior ('upper') or inferior ('lower').
"""
from protocols._fieldDefectMask import NasalStepMask


class ContrastDotsWithNasalStepRight(NasalStepMask):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithNasalStepRight'
        self.eye = 'right'  # nasal field is left of fixation; temporal field is right
