# -*- coding: utf-8 -*-
"""
Monitor Alignment Cross shows a fixation cross at the top-center of the
stimulus monitor for a fixed hold time so the EyeLink cameras can be aimed
at the display midpoint / top edge landmark.
"""
from protocols.protocol import protocol
from psychopy import core, visual, event


class MonitorAlignmentCross(protocol):
    def __init__(self):
        super().__init__()
        self.protocolName = 'MonitorAlignmentCross'
        self.backgroundColor = [0.0, 0.0, 0.0]  # mid-gray (PsychoPy RGB -1 to 1)
        self.preTime = 0.0
        self.stimTime = 20.0  # seconds the top-center cross is shown
        self.tailTime = 0.0
        self.fixationCrossColor = [1.0, -1.0, -1.0]  # red
        self.fixationCrossSizeDegrees = 1.0  # tip-to-tip span
        # Extra inset below the top edge (0 = top tip of the cross touches the screen edge).
        self.topMarginDegrees = 0.0

    def internalValidation(self):
        tf = True
        errorMessage = []
        if self.stimTime <= 0:
            tf = False
            errorMessage.append('Stim Time must be greater than 0 seconds.')
        if self.fixationCrossSizeDegrees <= 0:
            tf = False
            errorMessage.append('Fixation Cross Size must be greater than 0 degrees.')
        if self.topMarginDegrees < 0:
            tf = False
            errorMessage.append('Top Margin must be 0 or greater degrees.')
        tfColors, colorErrorMessages = self.validateColorInput()
        tf = tf and tfColors
        errorMessage += colorErrorMessages
        return tf, errorMessage

    def estimateTime(self):
        self._estimatedTime = self.preTime + self.stimTime + self.tailTime
        return self._estimatedTime

    def run(self, win, informationWin):
        self._completed = 0
        self._informationWin = informationWin
        self.getFR(win)

        if self.userInitiated:
            self.showInformationText(
                win,
                'Stimulus Information: Monitor Alignment Cross\n'
                'Top-center cross for EyeLink camera aiming\n'
                'Press any key to begin',
            )
            event.waitKeys()

        pix_per_deg_h, pix_per_deg_v = self.getPixPerDegXY(win.monitor)
        crossHalfX = 0.5 * float(self.fixationCrossSizeDegrees) * pix_per_deg_h
        crossHalfY = 0.5 * float(self.fixationCrossSizeDegrees) * pix_per_deg_v
        crossLineWidth = max(2.0, min(crossHalfX, crossHalfY) * 2.0 * 0.15)
        topMarginPix = float(self.topMarginDegrees) * pix_per_deg_v
        # PsychoPy pix: origin center, +y up. Place cross just below the top edge.
        crossY = win.size[1] / 2.0 - topMarginPix - crossHalfY
        crossPos = (0.0, crossY)

        fixationCrossArms = (
            visual.ShapeStim(
                win,
                units='pix',
                vertices=((-crossHalfX, 0.0), (crossHalfX, 0.0)),
                lineWidth=crossLineWidth,
                closeShape=False,
                lineColor=self.fixationCrossColor,
                pos=crossPos,
            ),
            visual.ShapeStim(
                win,
                units='pix',
                vertices=((0.0, -crossHalfY), (0.0, crossHalfY)),
                lineWidth=crossLineWidth,
                closeShape=False,
                lineColor=self.fixationCrossColor,
                pos=crossPos,
            ),
        )

        win.color = self.backgroundColor
        win.flip()
        win.flip()

        if self._informationWin[0]:
            self.showInformationText(
                win,
                'Monitor Alignment Cross\nHolding for {t:g}s'.format(t=self.stimTime),
            )

        trialClock = core.Clock()
        self._stimulusStartLog.append(trialClock.getTime())
        self.sendTTL()
        self._numberOfEpochsStarted += 1

        for f in range(self._preTimeNumFrames):
            win.flip()
            if self.checkQuitOrPause():
                return

        for f in range(self._stimTimeNumFrames):
            for arm in fixationCrossArms:
                arm.draw()
            win.flip()
            if self.checkQuitOrPause():
                return

        for f in range(self._tailTimeNumFrames):
            win.flip()
            if self.checkQuitOrPause():
                return

        self._stimulusEndLog.append(trialClock.getTime())
        self.sendTTL()
        self._numberOfEpochsCompleted += 1
        self._completed = 1
        print(
            '--> Monitor Alignment Cross: top-center cross held for {t:.1f}s '
            '(size {s:g}°, top margin {m:g}°).'.format(
                t=self._actualStimTime,
                s=float(self.fixationCrossSizeDegrees),
                m=float(self.topMarginDegrees),
            )
        )
