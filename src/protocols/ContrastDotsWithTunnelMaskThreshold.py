# -*- coding: utf-8 -*-
"""
Contrast Dots with Tunnel Mask Threshold ramps a gaze-contingent tunnel aperture
from 0° diameter up in fixed steps (default +1° every 2 s) until a peak size.

Peak defaults to 31° (≈ vertical FOV on a 27″ monitor at 60 cm).
Set tunnelPeakDiameterDegrees to 0 for full-screen diagonal at run time.

For each direction in directions, one expanding-tunnel trial is shown, separated
by the black fixation cross (tailTime). Dots stay at full white contrast.
"""
import math

from protocols.ContrastDotsWithTunnelMask import ContrastDotsWithTunnelMask


class ContrastDotsWithTunnelMaskThreshold(ContrastDotsWithTunnelMask):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithTunnelMaskThreshold'
        self.dotSizeDegrees = 1.0
        self.speed = 10.0
        self.directions = [90.0, 270.0]
        self.contrasts = [1.0]
        self.attentionProbe = False
        self.postStimTime = 0.0
        self.tailTime = 4.0 #seconds - fixation cross between tunnel ramp trials
        # Tunnel ramp: start fully occluded, grow clear aperture around gaze.
        self.tunnelVisibleDiameterDegrees = 0.0
        self.tunnelStepDegrees = 1.0
        self.tunnelStepDuration = 2.0 #seconds spent at each diameter
        self.tunnelPeakDiameterDegrees = 31.0  # ≈ VFOV @ 27″ / 60 cm; 0 = full-screen diagonal
        self._autoTunnelPeakDegrees = 31.0  # fallback for time estimates before a window exists
        self._currentTunnelDiameter = None
        self._syncStimTimeFromRamp()


    def _stimulusTitle(self):
        return 'Contrast Dots with Tunnel Mask Threshold'


    def _effectivePeakDiameter(self):
        peak = float(self.tunnelPeakDiameterDegrees)
        if peak > 0:
            return peak
        return float(getattr(self, '_autoTunnelPeakDegrees', 50.0))


    def _tunnelDiameterLevels(self):
        '''Ascending clear-aperture diameters from 0 to peak inclusive.'''
        step = float(self.tunnelStepDegrees)
        peak = self._effectivePeakDiameter()
        if step <= 0 or peak < 0:
            return [0.0]
        levels = []
        value = 0.0
        while value < peak - 1e-12:
            levels.append(round(value, 10))
            value += step
        levels.append(round(peak, 10))
        return levels


    def _syncStimTimeFromRamp(self):
        levels = self._tunnelDiameterLevels()
        self.stimTime = max(1, len(levels)) * float(self.tunnelStepDuration)
        # Keep the static property in sync for logging / edit UI display.
        self.tunnelVisibleDiameterDegrees = 0.0


    def internalValidation(self):
        tf = True
        errorMessage = []
        if float(self.tunnelStepDegrees) <= 0:
            tf = False
            errorMessage.append('Tunnel Step must be greater than 0 degrees.')
        if float(self.tunnelStepDuration) <= 0:
            tf = False
            errorMessage.append('Tunnel Step Duration must be greater than 0 seconds.')
        if float(self.tunnelPeakDiameterDegrees) < 0:
            tf = False
            errorMessage.append(
                'Tunnel Peak Diameter must be 0 (full screen) or greater degrees.'
            )
        peak = self._effectivePeakDiameter()
        if float(self.tunnelStepDegrees) > peak + 1e-12 and peak > 0:
            tf = False
            errorMessage.append('Tunnel Step must be less than or equal to Tunnel Peak Diameter.')
        self._syncStimTimeFromRamp()
        parentTf, parentErrors = super().internalValidation()
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
        '''One expanding-tunnel trial per direction, repeated stimulusReps times.'''
        self._syncStimTimeFromRamp()
        epochs = []
        for direction in self._directionPool():
            epochs.append({
                'contrast': 1.0,
                'direction': direction,
                'ramp': True,
                'tunnelPeak': self._effectivePeakDiameter(),
            })
        self._epochLog = []
        for _ in range(int(self.stimulusReps)):
            self._epochLog.extend([dict(epoch) for epoch in epochs])
        self._contrastLog = [epoch['contrast'] for epoch in self._epochLog]


    def _tunnelDiameterForFrame(self, frameIndex):
        levels = self._tunnelDiameterLevels()
        stepFrames = max(1, int(round(self._FR * float(self.tunnelStepDuration))))
        idx = min(int(frameIndex) // stepFrames, len(levels) - 1)
        return levels[idx]


    def _onMotionFrame(self, frameIndex, epoch):
        diameter = self._tunnelDiameterForFrame(frameIndex)
        if diameter == getattr(self, '_currentTunnelDiameter', None):
            return
        self._currentTunnelDiameter = diameter
        self.tunnelVisibleDiameterDegrees = diameter
        self._refreshTunnelMask()
        if frameIndex == 0 or abs(diameter - round(diameter)) < 1e-9:
            # Sparse console update at integer-ish steps (including start).
            if abs(diameter - round(diameter)) < 1e-9:
                print('--> Tunnel diameter now {d:g}°'.format(d=diameter))


    def _epochContrastDisplay(self, epoch):
        peak = float(epoch.get('tunnelPeak', self._effectivePeakDiameter()))
        step = float(self.tunnelStepDegrees)
        dur = float(self.tunnelStepDuration)
        auto = float(self.tunnelPeakDiameterDegrees) <= 0
        peakLabel = '{peak:g}\u00b0 full-screen'.format(peak=peak) if auto else '{peak:g}\u00b0'.format(peak=peak)
        return 'Tunnel 0 \u2192 {peak} (+{step:g}\u00b0 / {dur:g}s)'.format(
            peak=peakLabel, step=step, dur=dur,
        )


    def _epochDirectionExtra(self, epoch):
        return '\nTunnel ramp: expanding gaze aperture'


    def _initPerRunStimulus(self, win, ppd_xy):
        # Start each run fully occluded before the first motion frame updates.
        self.tunnelVisibleDiameterDegrees = 0.0
        self._currentTunnelDiameter = None
        super()._initPerRunStimulus(win, ppd_xy)


    def run(self, win, informationWin):
        ppd_h, ppd_v = self.getPixPerDegXY(win.monitor, win=win)
        widthDeg = float(win.size[0]) / float(ppd_h)
        heightDeg = float(win.size[1]) / float(ppd_v)
        self._autoTunnelPeakDegrees = math.hypot(widthDeg, heightDeg)
        self._syncStimTimeFromRamp()
        print(
            '--> Tunnel threshold peak = {p:g}° ({src}).'.format(
                p=self._effectivePeakDiameter(),
                src='full-screen diagonal' if float(self.tunnelPeakDiameterDegrees) <= 0 else 'user setting',
            )
        )
        return super().run(win, informationWin)
