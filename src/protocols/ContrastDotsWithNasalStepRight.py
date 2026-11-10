# -*- coding: utf-8 -*-
"""
Contrast Dots with a gaze-contingent right-eye nasal step.

The step runs from 10° to 30° nasal of fixation (left of gaze) and sits
on the horizontal meridian. Its upper edge rises from 3° to 9°. It keeps
that retinal size as the eye moves.
Blocked Hemifield chooses superior ('upper') or inferior ('lower').
"""
from protocols._fieldDefectMask import NasalStepMask


class ContrastDotsWithNasalStepRight(NasalStepMask):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithNasalStepRight'
        self.eye = 'right'  # nasal field is left of fixation; temporal field is right
