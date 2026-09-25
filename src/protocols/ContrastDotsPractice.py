# -*- coding: utf-8 -*-
"""
Contrast Dots Practice is a simplified Contrast Dots run for training subjects
on the attention-probe task (spacebar) before the real multi-contrast stimulus.

White dots at 100% contrast move up and down with the same size, speed, and
probe settings as Contrast Dots.
"""
from protocols.ContrastDots import ContrastDots


class ContrastDotsPractice(ContrastDots):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsPractice'
        # Practice: full white only, same up/down motion and probe task as Contrast Dots.
        self.contrasts = [1.0]
        self.directions = [90.0, 270.0]
        self.attentionProbe = True


    def _stimulusTitle(self):
        return 'Contrast Dots Practice'
