# -*- coding: utf-8 -*-
"""
Hemifield Mask Dots is Mask Dots with an upper or lower hemifield scotoma.

Defaults approximate a *severe* glaucomatous hemifield defect: one half of the
field relative to gaze is densely occluded, while the opposite hemifield stays
largely open (large aperture, not a small tunnel island). Soft Gaussian edges
are used at the horizontal meridian and far periphery.

This is a simplified altitudinal / hemifield model, not a patient-specific
arcuate scotoma from a Humphrey visual field.
"""
import math

import numpy as np
from psychopy import visual

from protocols.MaskDots import MaskDots


class HemifieldMaskDots(MaskDots):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'HemifieldMaskDots'
        # Severe hemifield: large enough that the clear half fills most of a
        # typical stimulus screen (~full hemifield), not a small 10° island.
        self.tunnelVisibleDiameterDegrees = 60.0
        # Relatively steep border for a dense defect (still soft, not razor-cut)
        self.tunnelEdgeSigmaDegrees = 1.0
        # Superior hemifield loss is a common starting clinical pattern
        self.blockedHemifield = 'upper'
        self.maskColor = [0.0, 0.0, 0.0]
        self.persistentDots = True
        self.contrasts = [1.0]

    def _stimulusTitle(self):
        return 'Hemifield Mask Dots'

    def internalValidation(self):
        tf, errorMessage = super().internalValidation()
        blocked = str(self.blockedHemifield).strip().lower()
        if blocked not in ('upper', 'lower'):
            tf = False
            errorMessage.append("Blocked Hemifield must be 'upper' or 'lower'.")
        else:
            self.blockedHemifield = blocked
        return tf, errorMessage

    def _buildTunnelOpacityMap(self, tex_size, overlay_size_pix, pix_per_deg):
        '''
        Semicircular gaze-centered aperture with soft radial and meridian edges.

        Clear only inside the tunnel radius AND in the visible hemifield.
        '''
        radius_pix = (self.tunnelVisibleDiameterDegrees / 2.0) * pix_per_deg
        sigma_pix = self.tunnelEdgeSigmaDegrees * pix_per_deg
        center = (tex_size - 1) / 2.0
        yy, xx = np.mgrid[0:tex_size, 0:tex_size]
        # Texture +y is down in array rows; convert so +y is up (matches PsychoPy)
        x_pix = (xx - center) * (overlay_size_pix / tex_size)
        y_pix = (center - yy) * (overlay_size_pix / tex_size)

        r_pix = np.sqrt(x_pix ** 2 + y_pix ** 2)
        radial_scaled = (r_pix - radius_pix) / (sigma_pix * math.sqrt(2.0))
        radial_opacity = 0.5 * (1.0 + np.vectorize(math.erf)(radial_scaled))

        # Soft step across the horizontal meridian through gaze.
        # blocked upper → mask when y > 0; blocked lower → mask when y < 0.
        meridian_scaled = y_pix / (sigma_pix * math.sqrt(2.0))
        if self.blockedHemifield == 'upper':
            hemifield_opacity = 0.5 * (1.0 + np.vectorize(math.erf)(meridian_scaled))
        else:
            hemifield_opacity = 0.5 * (1.0 + np.vectorize(math.erf)(-meridian_scaled))

        # Opaque if outside the tunnel OR in the blocked hemifield.
        tunnel_clear = 1.0 - np.clip(radial_opacity, 0.0, 1.0)
        visible_hemifield = 1.0 - np.clip(hemifield_opacity, 0.0, 1.0)
        opacity = 1.0 - tunnel_clear * visible_hemifield
        return np.clip(opacity, 0.0, 1.0).astype(np.float32)

    def _initPerRunStimulus(self, win, pix_per_deg):
        self.blockedHemifield = str(self.blockedHemifield).strip().lower()
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
        print(
            '--> Hemifield Mask Dots (severe hemifield defaults): blocking {side} '
            'hemifield, visible aperture diameter = {d:g}°, edge σ = {s:g}°.'.format(
                side=self.blockedHemifield,
                d=self.tunnelVisibleDiameterDegrees,
                s=self.tunnelEdgeSigmaDegrees,
            )
        )
        if getattr(self, '_elTracker', None) is None:
            print(
                '*** Hemifield Mask Dots: EyeLink is not active. '
                'The semicircle will stay at screen center.'
            )
            self._gazeMaskWarningShown = True

    def _renderDotsFrame(self, win, dots):
        dots.draw()
        gaze = self._readGazePix(win)
        if gaze is not None:
            self._lastGaze = list(gaze)
        elif not self._gazeMaskWarningShown and getattr(self, '_elTracker', None) is not None:
            print(
                '*** Hemifield Mask Dots: No valid gaze sample yet; '
                'holding mask at last position.'
            )
            self._gazeMaskWarningShown = True
        self._gazeMask.pos = self._lastGaze
        self._gazeMask.draw()
