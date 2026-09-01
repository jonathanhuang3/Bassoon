# -*- coding: utf-8 -*-
"""
Mask Dots is Contrast Dots with a gaze-contingent tunnel mask: a circular
aperture follows EyeLink gaze while the periphery is occluded with a soft
Gaussian falloff using the gray background color.
"""
import math

import numpy as np
from psychopy import visual

from protocols.ContrastDots import ContrastDots


class MaskDots(ContrastDots):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'MaskDots'
        self.tunnelVisibleDiameterDegrees = 10.0  # clear aperture diameter in degrees
        self.tunnelEdgeSigmaDegrees = 1.5  # Gaussian edge softness in degrees
        self.maskColor = [0.0, 0.0, 0.0]  # defaults to the gray background color
        self.persistentDots = True  # True = dots never respawn (infinite lifetime)
        self.contrasts = [1.0]  # single full-contrast block by default

    def _usePersistentDots(self):
        return bool(self.persistentDots)

    def _stimulusTitle(self):
        return 'Mask Dots'

    def internalValidation(self):
        tf, errorMessage = super().internalValidation()
        if self.tunnelVisibleDiameterDegrees <= 0:
            tf = False
            errorMessage.append('Tunnel Visible Diameter must be greater than 0 degrees.')
        if self.tunnelEdgeSigmaDegrees <= 0:
            tf = False
            errorMessage.append('Tunnel Edge Sigma must be greater than 0 degrees.')
        return tf, errorMessage

    def _readGazePix(self, win):
        '''Return the latest gaze position in PsychoPy pixel coordinates, or None.'''
        tracker = getattr(self, '_elTracker', None)
        if tracker is None:
            return None
        try:
            import pylink
            sample = tracker.getNewestSample()
            if sample is None:
                return None
            if sample.isLeftSample():
                gaze = sample.getLeftEye().getGaze()
            elif sample.isRightSample():
                gaze = sample.getRightEye().getGaze()
            else:
                return None
            if gaze[0] == pylink.MISSING_DATA or gaze[1] == pylink.MISSING_DATA:
                return None
            x = float(gaze[0]) - win.size[0] / 2.0
            y = win.size[1] / 2.0 - float(gaze[1])
            return (x, y)
        except Exception:
            return None

    def _buildTunnelOpacityMap(self, tex_size, overlay_size_pix, pix_per_deg):
        '''Radial Gaussian edge: transparent center, opaque periphery.'''
        radius_pix = (self.tunnelVisibleDiameterDegrees / 2.0) * pix_per_deg
        sigma_pix = self.tunnelEdgeSigmaDegrees * pix_per_deg
        center = (tex_size - 1) / 2.0
        yy, xx = np.mgrid[0:tex_size, 0:tex_size]
        r_pix = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        r_pix = r_pix * (overlay_size_pix / tex_size)
        scaled = (r_pix - radius_pix) / (sigma_pix * math.sqrt(2.0))
        opacity = 0.5 * (1.0 + np.vectorize(math.erf)(scaled))
        return np.clip(opacity, 0.0, 1.0).astype(np.float32)

    def _initPerRunStimulus(self, win, pix_per_deg):
        radius_pix = (self.tunnelVisibleDiameterDegrees / 2.0) * pix_per_deg
        sigma_pix = self.tunnelEdgeSigmaDegrees * pix_per_deg
        half_diagonal = math.hypot(win.size[0] / 2.0, win.size[1] / 2.0)
        overlay_half_size = int(
            math.ceil(half_diagonal + radius_pix + 4.0 * sigma_pix)
        )
        overlay_size_pix = overlay_half_size * 2
        tex_size = 512
        opacity = self._buildTunnelOpacityMap(tex_size, overlay_size_pix, pix_per_deg)
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
        self._lastGaze = [0.0, 0.0]
        self._gazeMaskWarningShown = False
        if getattr(self, '_elTracker', None) is None:
            print('*** Mask Dots: EyeLink is not active. The tunnel will stay at screen center.')
            self._gazeMaskWarningShown = True

    def _renderDotsFrame(self, win, dots):
        dots.draw()
        gaze = self._readGazePix(win)
        if gaze is not None:
            self._lastGaze = list(gaze)
        elif not self._gazeMaskWarningShown and getattr(self, '_elTracker', None) is not None:
            print('*** Mask Dots: No valid gaze sample yet; holding tunnel at last position.')
            self._gazeMaskWarningShown = True
        self._gazeMask.pos = self._lastGaze
        self._gazeMask.draw()

    def _teardownPerRunStimulus(self):
        self._gazeMask = None
