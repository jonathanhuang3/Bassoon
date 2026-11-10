# -*- coding: utf-8 -*-
"""
Contrast Dots with Central Scotoma Threshold starts with a gaze-centered
scotoma large enough to cover the screen, then shrinks it in fixed steps
(default −1° of diameter every 2 s) until it is gone.

The periphery is uncovered first. The center stays covered until the scotoma
has shrunk past it. Set scotomaStartDiameterDegrees to 0 to begin at full-screen
coverage. One shrinking-scotoma trial is shown for each direction. The default
is upward motion only (90°). Trials are separated by the black fixation cross
(tailTime). Dots stay at full white contrast.
"""
import math

import numpy as np

from protocols.ContrastDots import ContrastDots
from protocols.ContrastDotsWithTunnelMask import ContrastDotsWithTunnelMask


class ContrastDotsWithCentralScotomaThreshold(ContrastDotsWithTunnelMask):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithCentralScotomaThreshold'
        self.dotSizeDegrees = 1.0
        self.speed = 10.0
        self.directions = [90.0]  # degrees - upward motion only
        self.contrasts = [1.0]
        self.attentionProbe = False
        self.postStimTime = 0.0
        self.tailTime = 4.0  # seconds - fixation cross between scotoma ramp trials
        # 0 = cover the whole screen at the start, then shrink.
        self.scotomaStartDiameterDegrees = 0.0  # degrees; 0 = full-screen coverage
        self.scotomaStepDegrees = 1.0  # degrees of diameter removed at each step
        self.scotomaStepDuration = 2.0  # seconds spent at each diameter
        self._autoScotomaStartDegrees = 60.0  # fallback before the window size is known
        self._scotomaDiameterDegrees = 0.0
        self._currentScotomaDiameter = None
        self._lastPrintedScotomaDegree = None
        # The live diameter is scotoma-specific. Hide the tunnel aperture field.
        if hasattr(self, 'tunnelVisibleDiameterDegrees'):
            delattr(self, 'tunnelVisibleDiameterDegrees')
        self._syncStimTimeFromRamp()


    def _stimulusTitle(self):
        return 'Contrast Dots with Central Scotoma Threshold'


    def _gazeMissingMessage(self):
        return (
            '*** Contrast Dots with Central Scotoma Threshold: No valid gaze sample yet; '
            'holding the scotoma at the last position.'
        )


    def _trackerInactiveMessage(self):
        return (
            '*** Contrast Dots with Central Scotoma Threshold: EyeLink is not active. '
            'The scotoma will stay at screen center.'
        )


    def _screenCoverageDiameter(self, win, ppd_xy):
        '''Diameter that still occludes every screen pixel, including the soft edge.'''
        ppd_h, ppd_v = ppd_xy
        width_deg = float(win.size[0]) / float(ppd_h)
        height_deg = float(win.size[1]) / float(ppd_v)
        return math.hypot(width_deg, height_deg) + float(self.transitionWidthDegrees)


    def _updateAutoStartDiameter(self, win, ppd_xy):
        self._autoScotomaStartDegrees = self._screenCoverageDiameter(win, ppd_xy)


    def _effectiveStartDiameter(self):
        start = float(self.scotomaStartDiameterDegrees)
        if start > 0:
            return start
        return float(getattr(self, '_autoScotomaStartDegrees', 60.0))


    def _scotomaDiameterLevels(self):
        '''Descending scotoma diameters from full coverage down to 0.'''
        step = float(self.scotomaStepDegrees)
        start = self._effectiveStartDiameter()
        if step <= 0 or start <= 0:
            return [max(start, 0.0)]
        levels = []
        value = start
        while value > 1e-9:
            levels.append(round(value, 10))
            value -= step
        levels.append(0.0)
        return levels


    def _syncStimTimeFromRamp(self):
        levels = self._scotomaDiameterLevels()
        self.stimTime = max(1, len(levels)) * float(self.scotomaStepDuration)


    def internalValidation(self):
        tf = True
        errorMessage = []
        if float(self.scotomaStepDegrees) <= 0:
            tf = False
            errorMessage.append('Scotoma Step must be greater than 0 degrees.')
        if float(self.scotomaStepDuration) <= 0:
            tf = False
            errorMessage.append('Scotoma Step Duration must be greater than 0 seconds.')
        if float(self.scotomaStartDiameterDegrees) < 0:
            tf = False
            errorMessage.append(
                'Scotoma Start Diameter must be 0 (full screen) or greater degrees.'
            )
        if float(getattr(self, 'transitionWidthDegrees', 0)) < 0:
            tf = False
            errorMessage.append('Transition Width must be 0 or greater degrees.')
        start = self._effectiveStartDiameter()
        if float(self.scotomaStepDegrees) > start + 1e-12 and start > 0:
            tf = False
            errorMessage.append('Scotoma Step must be less than or equal to the starting diameter.')
        self._syncStimTimeFromRamp()
        parentTf, parentErrors = ContrastDots.internalValidation(self)
        tf = tf and parentTf
        errorMessage += parentErrors
        return tf, errorMessage


    def estimateTime(self):
        self._syncStimTimeFromRamp()
        postStimTime = getattr(self, 'postStimTime', 0.0)
        timePerEpoch = (
            self.preTime
            + self.stimTime
            + postStimTime
            + self.tailTime
            + self.interStimulusInterval
        )
        numberOfEpochs = self.stimulusReps * len(self._directionPool())
        # One lead-in fixation (tailTime) before the first dots trial.
        self._estimatedTime = timePerEpoch * numberOfEpochs + float(self.tailTime)
        return self._estimatedTime


    def createEpochLog(self):
        '''One shrinking-scotoma trial per direction, repeated stimulusReps times.'''
        self._syncStimTimeFromRamp()
        epochs = []
        for direction in self._directionPool():
            epochs.append({
                'contrast': 1.0,
                'direction': direction,
                'ramp': True,
                'scotomaStart': self._effectiveStartDiameter(),
            })
        self._epochLog = []
        for _ in range(int(self.stimulusReps)):
            self._epochLog.extend([dict(epoch) for epoch in epochs])
        self._contrastLog = [epoch['contrast'] for epoch in self._epochLog]


    def _scotomaDiameterForFrame(self, frameIndex):
        levels = self._scotomaDiameterLevels()
        stepFrames = max(1, int(round(self._FR * float(self.scotomaStepDuration))))
        idx = min(int(frameIndex) // stepFrames, len(levels) - 1)
        return levels[idx]


    def _buildTunnelOpacityMap(self, tex_size, overlay_size_pix, ppd_xy):
        '''Opaque disk around gaze. Outside the disk the dots stay visible.'''
        ppd_h, ppd_v = ppd_xy
        radius_deg = float(self._scotomaDiameterDegrees) / 2.0
        width_deg = float(self.transitionWidthDegrees)
        center = (tex_size - 1) / 2.0
        yy, xx = np.mgrid[0:tex_size, 0:tex_size]
        scale = overlay_size_pix / float(tex_size)
        x_deg = (xx - center) * scale / ppd_h
        y_deg = (yy - center) * scale / ppd_v
        r_deg = np.sqrt(x_deg ** 2 + y_deg ** 2)
        # Positive inside the scotoma -> opaque. Outside is clear.
        return self._raisedCosineOpacity(radius_deg - r_deg, width_deg)


    def _onMotionFrame(self, frameIndex, epoch):
        diameter = self._scotomaDiameterForFrame(frameIndex)
        if diameter == getattr(self, '_currentScotomaDiameter', None):
            return
        self._currentScotomaDiameter = diameter
        self._scotomaDiameterDegrees = diameter
        self._refreshTunnelMask()
        shown = int(round(diameter))
        if frameIndex == 0 or shown != getattr(self, '_lastPrintedScotomaDegree', None):
            self._lastPrintedScotomaDegree = shown
            print('--> Central scotoma diameter now {d:g}°'.format(d=diameter))


    def _epochContrastDisplay(self, epoch):
        start = float(epoch.get('scotomaStart', self._effectiveStartDiameter()))
        step = float(self.scotomaStepDegrees)
        dur = float(self.scotomaStepDuration)
        auto = float(self.scotomaStartDiameterDegrees) <= 0
        startLabel = '{start:g}\u00b0 full-screen'.format(start=start) if auto else '{start:g}\u00b0'.format(start=start)
        return 'Scotoma {start} \u2192 0 (\u2212{step:g}\u00b0 / {dur:g}s)'.format(
            start=startLabel, step=step, dur=dur,
        )


    def _epochDirectionExtra(self, epoch):
        return '\nCentral scotoma ramp: periphery uncovers as the disk shrinks'


    def _initPerRunStimulus(self, win, ppd_xy):
        self._updateAutoStartDiameter(win, ppd_xy)
        self._syncStimTimeFromRamp()
        self._scotomaDiameterDegrees = self._effectiveStartDiameter()
        self._currentScotomaDiameter = None
        self._lastPrintedScotomaDegree = None
        # Parent setup reads this for the mask log line, then it stays hidden.
        self.tunnelVisibleDiameterDegrees = self._scotomaDiameterDegrees
        super()._initPerRunStimulus(win, ppd_xy)
        delattr(self, 'tunnelVisibleDiameterDegrees')


    def run(self, win, informationWin):
        ppd_h, ppd_v = self.getPixPerDegXY(win.monitor, win=win)
        self._updateAutoStartDiameter(win, (ppd_h, ppd_v))
        self._syncStimTimeFromRamp()
        print(
            '--> Central scotoma starts at {p:g}° ({src}) and shrinks to 0°.'.format(
                p=self._effectiveStartDiameter(),
                src='full-screen coverage' if float(self.scotomaStartDiameterDegrees) <= 0 else 'user setting',
            )
        )
        return super().run(win, informationWin)
