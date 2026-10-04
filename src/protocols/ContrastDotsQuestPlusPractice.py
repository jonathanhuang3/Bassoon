# -*- coding: utf-8 -*-
"""
Contrast Dots Quest+ Practice: short easy 2AFC block matching Quest+ timing.

Use before Contrast Dots Quest+ so subjects learn ↑/↓ mapping and pacing
without contaminating Quest+ reaction times / threshold trials.

Trial UX matches Quest+: 1 s black-cross ITI, 0.5 s high-contrast motion,
blank gray until answer, marimba click. Ends with the same first-level jar
celebration as Contrast Dots Practice (jar fill = accuracy). No adaptive
Quest+, no OKR. Replay with R.
"""
from __future__ import annotations

import random

from psychopy import event

from protocols.ContrastDotsPractice import ContrastDotsPractice
from protocols.ContrastDotsQuestPlus import ContrastDotsQuestPlus


class ContrastDotsQuestPlusPractice(ContrastDotsQuestPlus):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsQuestPlusPractice'
        # Easy full-contrast practice; both polarities and up/down.
        self.questPlusPracticeTrials = 8
        self.questPlusPracticeContrast = 1.0
        self.attentionProbe = False
        self.staircaseCastleCelebration = False
        self.staircaseTestCastleCelebration = False
        self._practiceReplayRequested = False


    def _stimulusTitle(self):
        return 'Contrast Dots Quest+ Practice'


    def estimateTime(self):
        n = max(1, int(self.questPlusPracticeTrials))
        per = (
            float(self.staircaseITI)
            + float(self.staircaseStimDurationSec)
            + 0.8  # typical answer
        )
        self._estimatedTime = n * per + 8.0
        return self._estimatedTime


    def internalValidation(self):
        tf = True
        errorMessage = []
        if int(self.questPlusPracticeTrials) < 1:
            tf = False
            errorMessage.append('Quest+ Practice Trials must be at least 1.')
        mag = abs(float(self.questPlusPracticeContrast))
        if not (0.0 < mag <= 1.0):
            tf = False
            errorMessage.append(
                'Quest+ Practice Contrast magnitude must be > 0 and ≤ 1.'
            )
        dirs = getattr(self, 'staircaseDirections', None) or []
        if not dirs:
            tf = False
            errorMessage.append('Staircase Directions must contain at least one direction.')
        from protocols.ContrastDots import ContrastDots
        cdTf, cdErrors = ContrastDots.internalValidation(self)
        return tf and cdTf, errorMessage + cdErrors


    def _showPracticeEndCelebration(self, win, nCorrect, nTrials):
        '''
        Same first-level jar celebration as Contrast Dots Practice.
        Jar fill = 2AFC accuracy. Returns True to continue, False if quit.
        Sets replay flag on R.
        '''
        self._practiceReplayRequested = False
        fill = (nCorrect / float(nTrials)) if nTrials else 0.0
        helper = ContrastDotsPractice()
        try:
            action = helper._showPracticeJarCelebration(
                win, nCorrect, nTrials, 0, fill,
            )
        except Exception as err:
            print('*** Quest+ Practice celebration failed:', err)
            action = 'continue'
        if getattr(helper, '_stoppedEarly', 0):
            self._stoppedEarly = 1
        if action == 'quit':
            self._stoppedEarly = 1
            return False
        if action == 'replay':
            self._practiceReplayRequested = True
        return True


    def _runPracticeBlock(self, win, informationWin):
        '''Easy fixed-contrast 2AFC practice. Returns False if aborted.'''
        self._informationWin = informationWin
        self._initStaircaseSounds()
        ppd_h, ppd_v, dotDiameterPix, dotRadiusPix, dots, fixationCrossArms = (
            self._setupDotsAndCross(win)
        )
        self._staircaseFixationCrossArms = fixationCrossArms
        self._initPerRunStimulus(win, (ppd_h, ppd_v))

        nTrials = max(1, int(self.questPlusPracticeTrials))
        mag = abs(float(self.questPlusPracticeContrast))
        directionPool = [self.deg0to360(d) for d in self.staircaseDirections]

        if self.userInitiated:
            self.showInformationText(
                win,
                'Stimulus Information: {title}\n'
                'Practice: brief moving dots, then press UP or DOWN\n'
                'High contrast — learn the keys before the real test\n'
                'Press any key to begin'.format(title=self._stimulusTitle()),
            )
            event.waitKeys()

        trialClock = self._startTrialClock()
        nCorrect = 0
        rts = []
        print(
            '--> Quest+ Practice: {n} easy 2AFC trials '
            '(|contrast|={c:g}, stim={stim:g}s, ITI={iti:g}s)'.format(
                n=nTrials,
                c=mag,
                stim=float(self.staircaseStimDurationSec),
                iti=float(self.staircaseITI),
            )
        )

        for i in range(1, nTrials + 1):
            polarity = random.choice([1.0, -1.0])
            contrast = polarity * mag
            directionDeg = random.choice(directionPool)

            if not self._showFixation(win, fixationCrossArms, trialClock, self.staircaseITI):
                return False

            responseDir, correct, rt, _reminder, quitEarly = self._runStaircaseTrial(
                win, dots, dotDiameterPix, dotRadiusPix, ppd_h, ppd_v,
                contrast, directionDeg, trialClock,
            )
            if quitEarly:
                return False

            if correct:
                nCorrect += 1
            rts.append(float(rt))
            print(
                '--> Practice {i}/{n}: {stim} → {ok} RT={rt:.3f}s'.format(
                    i=i,
                    n=nTrials,
                    stim='up' if abs(directionDeg - 90.0) < 1e-6 else 'down',
                    ok='correct' if correct else 'WRONG',
                    rt=rt,
                )
            )

        meanRt = (sum(rts) / float(len(rts))) if rts else None
        print(
            '--> Quest+ Practice done: {ok}/{n} correct, mean RT={rt}'.format(
                ok=nCorrect,
                n=nTrials,
                rt=('NA' if meanRt is None else '{:.3f}s'.format(meanRt)),
            )
        )
        return self._showPracticeEndCelebration(win, nCorrect, nTrials)


    def run(self, win, informationWin):
        self._completed = 0
        self.attentionProbe = False
        while True:
            self._practiceReplayRequested = False
            print('--> Contrast Dots Quest+ Practice: starting')
            if not self._runPracticeBlock(win, informationWin):
                print('*** Quest+ Practice aborted')
                return
            if getattr(self, '_stoppedEarly', 0):
                return
            if not self._practiceReplayRequested:
                self._completed = 1
                return
            print('--> Practice again!')
