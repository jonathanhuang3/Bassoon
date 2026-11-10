# -*- coding: utf-8 -*-
"""
Contrast Dots Persistent is Contrast Dots with dots that stay on screen.

Each dot is placed once per trial and keeps moving. Dots that leave one edge
wrap to the opposite edge. They do not expire and respawn, so the field does
not flicker. Attention-probe flashes still last one dot lifetime.
"""
from protocols.ContrastDots import ContrastDots


class ContrastDotsPersistent(ContrastDots):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsPersistent'

    def _stimulusTitle(self):
        return 'Contrast Dots Persistent'

    def _usePersistentDots(self):
        '''Dots stay in the field and wrap at the screen edge.'''
        return True
