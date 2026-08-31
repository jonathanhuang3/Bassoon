# -*- coding: utf-8 -*-
"""
Full-field uniform gray levels for monitor luminance / gamma calibration.
Steps through RGB levels from levelStart to levelEnd and holds each steady.
"""
from protocols.protocol import protocol
from psychopy import core, visual, event


class CalibrationMonitor(protocol):
    def __init__(self):
        super().__init__()
        self.protocolName = 'CalibrationMonitor'
        self.levelStart = -1.0 #minimum RGB level (PsychoPy -1 to 1)
        self.levelEnd = 1.0 #maximum RGB level (PsychoPy -1 to 1)
        self.levelStep = 0.1 #RGB increment between calibration levels
        self.holdTime = 20.0 #seconds to hold each level steady for photometer readings
        self.preTime = 0.0 #seconds - unused; kept for protocol base class compatibility
        self.stimTime = 20.0 #seconds - mirrored to holdTime for timing helpers
        self.tailTime = 0.0 #seconds - unused; kept for protocol base class compatibility
        self.interStimulusInterval = 0.0 #seconds - pause between levels


    def internalValidation(self):
        tf = True
        errorMessage = []

        if self.levelStep <= 0:
            tf = False
            errorMessage.append('Level Step must be greater than 0.')
        if self.levelEnd < self.levelStart:
            tf = False
            errorMessage.append('Level End must be greater than or equal to Level Start.')
        if self.holdTime <= 0:
            tf = False
            errorMessage.append('Hold Time must be greater than 0 seconds.')
        if self.levelStart < -1 or self.levelEnd > 1:
            tf = False
            errorMessage.append('Level Start and Level End must be between -1 and 1.')

        return tf, errorMessage


    def buildLevelSequence(self):
        levels = []
        level = self.levelStart
        while level <= self.levelEnd + (self.levelStep * 0.5):
            levels.append(round(level, 4))
            level += self.levelStep
        return levels


    def estimateTime(self):
        levels = self.buildLevelSequence()
        timePerLevel = self.holdTime + self.interStimulusInterval
        self._estimatedTime = timePerLevel * len(levels)
        return self._estimatedTime


    def run(self, win, informationWin):
        self._completed = 0
        self._informationWin = informationWin
        self.stimTime = self.holdTime
        self.getFR(win)

        self._interStimulusIntervalNumFrames = round(self._FR * self.interStimulusInterval)
        self._stimTimeNumFrames = round(self._FR * self.holdTime)
        self._actualStimTime = self._stimTimeNumFrames * (1 / self._FR)

        levels = self.buildLevelSequence()
        totalLevels = len(levels)

        if self.userInitiated:
            self.showInformationText(
                win,
                'Stimulus Information: Calibration Monitor\n'
                'Full-field levels from {start} to {end}\n'
                'Press any key to begin'.format(
                    start=self.levelStart, end=self.levelEnd,
                ),
            )
            event.waitKeys()

        trialClock = core.Clock()

        for levelNum, level in enumerate(levels, start=1):
            if self._informationWin[0]:
                self.showInformationText(
                    win,
                    'Calibration Monitor\n'
                    'Level = {level}\n'
                    '{n} of {total}'.format(
                        level=level, n=levelNum, total=totalLevels,
                    ),
                )

            print(
                '--> CalibrationMonitor level {level} ({n}/{total}), '
                'hold {t:.1f}s'.format(
                    level=level, n=levelNum, total=totalLevels, t=self.holdTime,
                ),
            )

            for f in range(self._interStimulusIntervalNumFrames):
                win.flip()
                if self.checkQuitOrPause():
                    return

            win.color = (level, level, level)
            self._stimulusStartLog.append(trialClock.getTime())
            self.sendTTL()
            self._numberOfEpochsStarted += 1

            for f in range(self._stimTimeNumFrames):
                win.flip()
                if self.checkQuitOrPause():
                    return

            self._stimulusEndLog.append(trialClock.getTime())
            self.sendTTL()
            self._numberOfEpochsCompleted += 1

        self._completed = 1
