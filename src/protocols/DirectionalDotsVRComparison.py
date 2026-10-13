# -*- coding: utf-8 -*-
"""
Directional Dots VR Comparison: Directional Dots tuned to match a Unity/VR
reference field (dark background, large sparse dots, spawn to screen edges).

Dot lifetime matches Contrast Dots: fixed lifetime + spawnStagger (not the
classic Directional Dots lifetimeOffset shortening).
"""
import random

from protocols.DirectionalDots import DirectionalDots


class DirectionalDotsVRComparison(DirectionalDots):
    def __init__(self):
        super().__init__()
        self.protocolName = 'DirectionalDotsVRComparison'
        # Match VR comparison field: dark linearized gray, sparse large dots.
        self.backgroundColor = [-0.91, -0.91, -0.91]
        self.numberOfDots = 25
        self.dotRadius_degrees = 2.5  # degrees - radius (diameter = 5 deg)
        self.scotomaOn = False
        self.interStimulusInterval = 0.0 #seconds - no wait between epochs
        # Contrast Dots-style lifetime (fixed duration + staggered first expiry).
        self.lifetimeMode = 'stagger' # 'stagger' = fixed dotLifetime for all dots; first expiry spread by spawnStagger (Contrast Dots style). Do not use 'offset' here.
        self.dotLifetime = 0.15 # seconds - how long each dot moves before respawning
        self.spawnStagger = 0.15 # seconds - max random delay before each dot's first lifetime expiry
        # Not used in stagger mode; hide from the Edit Stimulus panel.
        if hasattr(self, 'lifetimeOffset'):
            del self.lifetimeOffset


    def lifetime(self, win, dot=None):
        '''
        Restart lifetime timer and place the dot anywhere on screen
        (no edge inset; dots may be partially clipped at borders).
        '''
        halfW = win.size[0] / 2.0
        halfH = win.size[1] / 2.0
        xPos = random.uniform(-halfW, halfW)
        yPos = random.uniform(-halfH, halfH)
        if dot is not None:
            self.currentFrames[dot] = 0
            self.dotCoords[dot] = [xPos, yPos]
        return xPos, yPos
