# -*- coding: utf-8 -*-
"""
Contrast Dots with Hemifield Mask is Contrast Dots with a gaze-centered upper
or lower hemifield scotoma.

One half of the field relative to gaze is occluded; the other half stays
open. The border is a single raised-cosine transition across the horizontal
meridian (no radial tunnel).
"""
import math

import numpy as np
from psychopy import visual

from protocols.ContrastDots import ContrastDots
from protocols.ContrastDotsWithTunnelMask import ContrastDotsWithTunnelMask


class ContrastDotsWithHemifieldMask(ContrastDotsWithTunnelMask):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithHemifieldMask'
        self._stripLegacyTunnelParams()

        # Superior hemifield loss is a common starting clinical pattern
        self.blockedHemifield = 'upper'
        # Full width of the raised-cosine fade across the horizontal meridian (degrees)
        self.transitionWidthDegrees = 1.0
        self.maskColor = [0.0, 0.0, 0.0]
        self.persistentDots = False
        self.contrasts = [1.0, 0.1, 0.05, -0.05, -0.1, -1.0]

    def _stripLegacyTunnelParams(self):
        '''Remove tunnel fields so they do not appear in Edit / saved sketches.'''
        for name in ('tunnelVisibleDiameterDegrees', 'tunnelEdgeSigmaDegrees'):
            if hasattr(self, name):
                delattr(self, name)

    def _stimulusTitle(self):
        return 'Contrast Dots with Hemifield Mask'

    def internalValidation(self):
        self._stripLegacyTunnelParams()
        # Skip tunnel checks; validate ContrastDots fields + hemifield params.
        tf, errorMessage = ContrastDots.internalValidation(self)
        blocked = str(self.blockedHemifield).strip().lower()
        if blocked not in ('upper', 'lower'):
            tf = False
            errorMessage.append("Blocked Hemifield must be 'upper' or 'lower'.")
        else:
            self.blockedHemifield = blocked
        if self.transitionWidthDegrees < 0:
            tf = False
            errorMessage.append('Transition Width must be 0 or greater degrees.')
        return tf, errorMessage

    def _buildTunnelOpacityMap(self, tex_size, overlay_size_pix, ppd_xy):
        '''Gaze-centered hemifield mask with one raised-cosine meridian edge.'''
        ppd_h, ppd_v = ppd_xy
        width_deg = float(self.transitionWidthDegrees)
        center = (tex_size - 1) / 2.0
        yy, xx = np.mgrid[0:tex_size, 0:tex_size]
        scale = overlay_size_pix / float(tex_size)
        # Texture +y is down in array rows; convert so +y is up (matches PsychoPy)
        y_deg = (center - yy) * scale / ppd_v
        # Positive t = blocked side of the meridian.
        if self.blockedHemifield == 'upper':
            t = y_deg
        else:
            t = -y_deg
        return self._raisedCosineOpacity(t, width_deg)

    def _gazeMissingMessage(self):
        return (
            '*** Contrast Dots with Hemifield Mask: No valid gaze sample yet; '
            'holding mask at last position.'
        )

    def _trackerInactiveMessage(self):
        return (
            '*** Contrast Dots with Hemifield Mask: EyeLink is not active. '
            'The hemifield mask will stay at screen center.'
        )

    def _initPerRunStimulus(self, win, ppd_xy):
        self._stripLegacyTunnelParams()
        self.blockedHemifield = str(self.blockedHemifield).strip().lower()
        ppd_h, ppd_v = ppd_xy
        transition_pix = max(
            0.0,
            float(self.transitionWidthDegrees) * max(ppd_h, ppd_v),
        )
        # Same coverage rule as the tunnel mask: full diagonal from extreme gaze.
        full_diagonal = math.hypot(float(win.size[0]), float(win.size[1]))
        gaze_margin_pix = 200.0
        overlay_half_size = int(math.ceil(
            full_diagonal + gaze_margin_pix + transition_pix + 1.0
        ))
        overlay_size_pix = max(overlay_half_size * 2, 2)
        tex_size = 1024
        opacity = self._buildTunnelOpacityMap(tex_size, overlay_size_pix, ppd_xy)
        self._gazeMask = visual.ImageStim(
            win,
            image=np.ones((tex_size, tex_size), dtype=np.float32),
            mask=opacity,
            color=self.maskColor,
            size=(overlay_size_pix, overlay_size_pix),
            units='pix',
            pos=(0.0, 0.0),
            interpolate=True,
        )
        self._lastGaze = (0.0, 0.0)
        self._gazeMaskWarningShown = False
        self._initGazeReadCache(win)
        print(
            '--> Contrast Dots with Hemifield Mask: blocking {side} hemifield, '
            'raised-cosine transition = {w:g}° '
            '(mask overlay {s} px, ppd_h={h:.2f} ppd_v={v:.2f}).'.format(
                side=self.blockedHemifield,
                w=float(self.transitionWidthDegrees),
                s=overlay_size_pix,
                h=ppd_h,
                v=ppd_v,
            )
        )
        if self._tracker is None:
            print(self._trackerInactiveMessage())
            self._gazeMaskWarningShown = True
