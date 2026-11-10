# -*- coding: utf-8 -*-
"""
Contrast Dots with a gaze-contingent left-eye nasal step.

The step runs from 10° to 30° nasal of fixation (right of gaze) and sits
on the horizontal meridian. Its upper edge rises from 3° to 9°. It keeps
that retinal size as the eye moves.
Blocked Hemifield chooses superior ('upper') or inferior ('lower').
"""
from protocols._fieldDefectMask import NasalStepMask


class ContrastDotsWithNasalStepLeft(NasalStepMask):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithNasalStepLeft'
        self.eye = 'left'  # nasal field is right of fixation; temporal field is left
