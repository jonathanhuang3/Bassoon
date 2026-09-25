# -*- coding: utf-8 -*-
"""
Contrast Dots Threshold finds approximate contrast thresholds with a continuous
ramp: dots start at mid-gray (contrast 0) and step toward full white (+peak) or
full black (-peak) by contrastStep every contrastStepDuration seconds.

For each direction in directions, one positive ramp trial and one negative ramp
trial are shown. Trials are separated by the red fixation cross (tailTime).
"""
from protocols.ContrastDots import ContrastDots


class ContrastDotsThreshold(ContrastDots):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsThreshold'
        self.dotSizeDegrees = 1.0
        self.speed = 10.0
        self.directions = [90.0, 270.0]
        # Ramp: 0, step, 2*step, ... peak (same magnitudes for negative polarity).
        self.contrastStep = 0.01
        self.contrastStepDuration = 2.0 #seconds spent at each contrast level
        self.contrastPeak = 0.3
        self.postStimTime = 0.0 #no afternystagmus blank; fixation separates trials
        self.tailTime = 4.0 #seconds - fixation cross between threshold trials
        self.attentionProbe = False
        # contrasts is unused for epoch building; kept valid for shared validation helpers.
        self.contrasts = [0.0]
        self._syncStimTimeFromRamp()


    def _stimulusTitle(self):
        return 'Contrast Dots Threshold'


    def _syncStimTimeFromRamp(self):
        levels = self._unsignedRampLevels()
        self.stimTime = max(1, len(levels)) * float(self.contrastStepDuration)


    def _unsignedRampLevels(self):
        '''Ascending magnitudes from 0 to contrastPeak inclusive.'''
        step = float(self.contrastStep)
        peak = float(self.contrastPeak)
        if step <= 0 or peak < 0:
            return [0.0]
        levels = []
        value = 0.0
        # Guard against float drift when stepping to peak.
        while value < peak - 1e-12:
            levels.append(round(value, 10))
            value += step
        levels.append(round(peak, 10))
        return levels


    def internalValidation(self):
        tf = True
        errorMessage = []
        if float(self.contrastStep) <= 0:
            tf = False
            errorMessage.append('Contrast Step must be greater than 0.')
        if float(self.contrastStepDuration) <= 0:
            tf = False
            errorMessage.append('Contrast Step Duration must be greater than 0 seconds.')
        if float(self.contrastPeak) <= 0 or float(self.contrastPeak) > 1:
            tf = False
            errorMessage.append('Contrast Peak must be greater than 0 and at most 1.')
        if float(self.contrastStep) > float(self.contrastPeak) + 1e-12:
            tf = False
            errorMessage.append('Contrast Step must be less than or equal to Contrast Peak.')
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
        # Each direction → positive ramp + negative ramp.
        numberOfEpochs = self.stimulusReps * len(self._directionPool()) * 2
        self._estimatedTime = timePerEpoch * numberOfEpochs
        return self._estimatedTime


    def createEpochLog(self):
        '''
        Ordered trials: for each direction, positive ramp then negative ramp.
        Repeats the full sequence stimulusReps times. Not shuffled.
        '''
        self._syncStimTimeFromRamp()
        peak = float(self.contrastPeak)
        epochs = []
        for direction in self._directionPool():
            for polarity in (1.0, -1.0):
                epochs.append({
                    'contrast': polarity * peak,
                    'direction': direction,
                    'polarity': polarity,
                    'ramp': True,
                })
        self._epochLog = []
        for _ in range(int(self.stimulusReps)):
            self._epochLog.extend([dict(epoch) for epoch in epochs])
        self._contrastLog = [epoch['contrast'] for epoch in self._epochLog]


    def _motionContrast(self, epoch, frameIndex):
        levels = self._unsignedRampLevels()
        stepFrames = max(1, int(round(self._FR * float(self.contrastStepDuration))))
        idx = min(int(frameIndex) // stepFrames, len(levels) - 1)
        return float(epoch.get('polarity', 1.0)) * levels[idx]


    def _epochContrastDisplay(self, epoch):
        peak = abs(float(epoch.get('contrast', self.contrastPeak)))
        step = float(self.contrastStep)
        if float(epoch.get('polarity', 1.0)) >= 0:
            return '0 \u2192 +{peak:g} (+{step:g}/s)'.format(peak=peak, step=step)
        return '0 \u2192 -{peak:g} (-{step:g}/s)'.format(peak=peak, step=step)


    def _epochDirectionExtra(self, epoch):
        polarity = float(epoch.get('polarity', 1.0))
        label = 'positive (white)' if polarity >= 0 else 'negative (black)'
        return '\nThreshold ramp: {label}'.format(label=label)


    def run(self, win, informationWin):
        self._syncStimTimeFromRamp()
        return super().run(win, informationWin)
