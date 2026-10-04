# -*- coding: utf-8 -*-
"""
Contrast Dots Quest+: dual-polarity 2AFC Quest+Weibull threshold then OKR.

Phase 1 — Two interleaved Quest+Weibull engines (+/− polarity magnitudes):
  Independent posteriors over log-spaced linear contrast thresholds
  (stim_scale='linear': grids are |contrast| in 0.01–1, not log10 units).
  Fixed trial budget per polarity; catch trials do not update Quest+.
  Same PsychoPy trial UX as Contrast Dots Staircase: 1 s black-cross ITI,
  0.5 s motion, blank gray until they answer. Optional test flag jumps to
  castle after one correct.

Phase 2 — Stacked-jar castle checkpoint, then OKR at +2/4/8×θ+ and −2/4/8×θ−
  (inherited from ContrastDotsStaircase).
"""
from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from protocols.ContrastDotsStaircase import ContrastDotsStaircase


class ContrastDotsQuestPlus(ContrastDotsStaircase):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsQuestPlus'

        # Quest+ domain / stopping (GUI-editable via comments).
        self.questPlusTrialsPerPolarity = 30  # adaptive trials per +/− polarity
        self.questPlusMinContrast = 0.01  # intensity / threshold floor
        self.questPlusMaxContrast = 1.0  # intensity / threshold ceiling
        self.questPlusNIntensities = 40  # log-spaced intensity & threshold grid size
        self.questPlusNSlopes = 20  # log-spaced slope grid size (0.5–8)
        self.questPlusSlopeMin = 0.5
        self.questPlusSlopeMax = 8.0
        self.questPlusLapseRate = 0.02  # fixed 2AFC lapse; absorbs key errors
        self.questPlusLowerAsymptote = 0.5  # 2AFC guess rate
        # Grids are linear |contrast|; must be 'linear' (not default log10).
        self.questPlusStimScale = 'linear'
        self.questPlusCatchEvery = 12  # catch every N trials (0 = none)
        # Keep staircase catch setting in sync for inherited compliance checks.
        self.staircaseCatchEvery = self.questPlusCatchEvery
        # Align floor used by inherited helpers / OKR clamp with Quest+ min.
        self.staircaseMinContrast = self.questPlusMinContrast

        # Match staircase 2AFC motion duration (hardened vs 1 s pilot).
        self.staircaseStimDurationSec = 0.5
        # Fixation-cross ITI between Quest+ trials (longer than staircase default).
        self.staircaseITI = 1.0

        # Testing: jump to castle after one correct 2AFC response.
        self.staircaseTestCastleCelebration = False

        self._questPlusEngines = {}
        self._questPlusTrialCounts = {1.0: 0, -1.0: 0}


    def _stimulusTitle(self):
        return 'Contrast Dots Quest+'


    def internalValidation(self):
        tf = True
        errorMessage = []
        minC = float(self.questPlusMinContrast)
        maxC = float(self.questPlusMaxContrast)
        if not (0.0 < minC < maxC <= 1.0):
            tf = False
            errorMessage.append(
                'Quest+ Min Contrast must be > 0 and < Max Contrast, and Max ≤ 1.'
            )
        if int(self.questPlusTrialsPerPolarity) < 5:
            tf = False
            errorMessage.append('Quest+ Trials Per Polarity must be at least 5.')
        if int(self.questPlusNIntensities) < 5:
            tf = False
            errorMessage.append('Quest+ N Intensities must be at least 5.')
        if int(self.questPlusNSlopes) < 3:
            tf = False
            errorMessage.append('Quest+ N Slopes must be at least 3.')
        if not (0.0 < float(self.questPlusLapseRate) < 0.5):
            tf = False
            errorMessage.append('Quest+ Lapse Rate must be between 0 and 0.5.')
        if not (0.0 < float(self.questPlusLowerAsymptote) < 1.0):
            tf = False
            errorMessage.append('Quest+ Lower Asymptote must be between 0 and 1.')
        if float(self.questPlusSlopeMin) <= 0 or float(self.questPlusSlopeMax) <= float(self.questPlusSlopeMin):
            tf = False
            errorMessage.append('Quest+ Slope Min/Max must satisfy 0 < Min < Max.')
        scale = str(getattr(self, 'questPlusStimScale', 'linear')).strip().lower()
        if scale not in ('linear', 'log10', 'db'):
            tf = False
            errorMessage.append("Quest+ Stim Scale must be 'linear', 'log10', or 'dB'.")
        if int(self.questPlusCatchEvery) < 0:
            tf = False
            errorMessage.append('Quest+ Catch Every must be 0 or greater.')
        dirs = getattr(self, 'staircaseDirections', None) or []
        if not dirs:
            tf = False
            errorMessage.append('Staircase Directions must contain at least one direction.')
        mults = list(getattr(self, 'okrThresholdMultipliers', []) or [])
        if not mults or any(float(m) <= 0 for m in mults):
            tf = False
            errorMessage.append('OKR Threshold Multipliers must be a list of values > 0.')

        # Skip staircase-specific validation; keep ContrastDots checks.
        from protocols.ContrastDots import ContrastDots
        cdTf, cdErrors = ContrastDots.internalValidation(self)
        tf = tf and cdTf
        errorMessage += cdErrors
        return tf, errorMessage


    def estimateTime(self):
        nAdapt = 2 * int(self.questPlusTrialsPerPolarity)
        catchEvery = int(self.questPlusCatchEvery)
        nCatch = (nAdapt // catchEvery) if catchEvery > 0 else 0
        nTrials = nAdapt + nCatch
        stairSec = nTrials * (
            float(self.staircaseResponseReminderSec) + float(self.staircaseITI) + 2.0
        )
        # Align catch setting used elsewhere.
        self.staircaseCatchEvery = int(self.questPlusCatchEvery)
        self.staircaseMinContrast = float(self.questPlusMinContrast)
        theta = max(float(self.questPlusMinContrast), 0.1)
        self.contrasts = self._contrastsFromPolarityThresholds(theta, theta)
        okrSec = super(ContrastDotsStaircase, self).estimateTime()
        self._estimatedTime = stairSec + 5.0 + okrSec
        return self._estimatedTime


    def _buildQuestPlusGrids(self):
        minC = float(self.questPlusMinContrast)
        maxC = float(self.questPlusMaxContrast)
        nI = int(self.questPlusNIntensities)
        nS = int(self.questPlusNSlopes)
        intensities = np.logspace(np.log10(minC), np.log10(maxC), num=nI)
        thresholds = np.logspace(np.log10(minC), np.log10(maxC), num=nI)
        slopes = np.logspace(
            np.log10(float(self.questPlusSlopeMin)),
            np.log10(float(self.questPlusSlopeMax)),
            num=nS,
        )
        lower_asymptotes = [float(self.questPlusLowerAsymptote)]
        lapse_rates = [float(self.questPlusLapseRate)]
        return intensities, thresholds, slopes, lower_asymptotes, lapse_rates


    def _newQuestPlusEngine(self):
        try:
            import questplus as qp
        except Exception as err:
            raise RuntimeError('questplus is required for Contrast Dots Quest+: {0}'.format(err))
        intensities, thresholds, slopes, lower_asymptotes, lapse_rates = (
            self._buildQuestPlusGrids()
        )
        scale = str(getattr(self, 'questPlusStimScale', 'linear')).strip()
        # questplus accepts 'dB'; normalize common aliases.
        if scale.lower() == 'db':
            scale = 'dB'
        return qp.QuestPlusWeibull(
            intensities=intensities,
            thresholds=thresholds,
            slopes=slopes,
            lower_asymptotes=lower_asymptotes,
            lapse_rates=lapse_rates,
            stim_scale=scale,
            stim_selection_method='min_entropy',
            param_estimation_method='mean',
        )


    def _questPlusDone(self, polarity):
        return int(self._questPlusTrialCounts.get(polarity, 0)) >= int(
            self.questPlusTrialsPerPolarity
        )


    def _pickQuestPlusPolarity(self, lastPolarity):
        openOnes = [p for p in (1.0, -1.0) if not self._questPlusDone(p)]
        if not openOnes:
            return None
        if len(openOnes) == 1:
            return openOnes[0]
        if lastPolarity is None:
            return random.choice(openOnes)
        other = -float(lastPolarity)
        return other if other in openOnes else openOnes[0]


    def _clampQuestIntensity(self, intensity):
        return float(min(
            float(self.questPlusMaxContrast),
            max(float(self.questPlusMinContrast), float(intensity)),
        ))


    def _snapQuestIntensity(self, intensity):
        '''Clamp and snap to nearest intensity-grid value for stable updates.'''
        intensities, _, _, _, _ = self._buildQuestPlusGrids()
        x = self._clampQuestIntensity(intensity)
        idx = int(np.argmin(np.abs(intensities - x)))
        return float(intensities[idx])


    def _questPlusEstimate(self, engine):
        est = getattr(engine, 'param_estimate', None) or {}
        thr = self._clampQuestIntensity(est.get('threshold', self.questPlusMinContrast))
        slope = float(est.get('slope', float('nan')))
        return thr, slope


    def _finalizeQuestPlusPolarity(self, engine, nTrials, testMode=False):
        '''
        Returns (thetaForOkr, status, slope).
        status: completed | at_floor | at_ceiling | test_castle | incomplete
        '''
        if testMode:
            thr, slope = self._questPlusEstimate(engine)
            return thr, 'test_castle', slope
        if nTrials <= 0:
            return float(self.questPlusMinContrast), 'incomplete', float('nan')
        thr, slope = self._questPlusEstimate(engine)
        # Under-budget runs are incomplete even if the mean sits near a boundary.
        if nTrials < int(self.questPlusTrialsPerPolarity):
            return thr, 'incomplete', slope
        minC = float(self.questPlusMinContrast)
        maxC = float(self.questPlusMaxContrast)
        # Near-boundary posterior mean → censored status for OKR interpretation.
        if thr <= minC * 1.05:
            return minC, 'at_floor', slope
        if thr >= maxC * 0.95:
            return thr, 'at_ceiling', slope
        return thr, 'completed', slope


    def _runStaircasePhase(self, win, informationWin):
        '''Override staircase adaptive phase with dual interleaved Quest+.'''
        return self._runQuestPlusPhase(win, informationWin)


    def _runQuestPlusPhase(self, win, informationWin):
        '''
        Dual interleaved Quest+Weibull (+/− polarity).
        Sets thresholdContrastPositive / Negative / combined.
        Returns False if aborted.
        '''
        self.staircaseCatchEvery = int(self.questPlusCatchEvery)
        self.staircaseMinContrast = float(self.questPlusMinContrast)

        self._informationWin = informationWin
        self._staircaseTrials = []
        self._questPlusTrialCounts = {1.0: 0, -1.0: 0}
        self._initStaircaseSounds()
        ppd_h, ppd_v, dotDiameterPix, dotRadiusPix, dots, fixationCrossArms = (
            self._setupDotsAndCross(win)
        )
        self._initPerRunStimulus(win, (ppd_h, ppd_v))

        try:
            engines = {
                1.0: self._newQuestPlusEngine(),
                -1.0: self._newQuestPlusEngine(),
            }
        except Exception as err:
            print('*** Quest+ init failed:', err)
            return False
        self._questPlusEngines = engines

        if self.userInitiated:
            self.showInformationText(
                win,
                'Stimulus Information: {title}\n'
                'Judge motion direction — brief moving dots, then press UP or DOWN\n'
                'Press any key to begin Quest+'.format(title=self._stimulusTitle()),
            )
            from psychopy import event
            event.waitKeys()

        from psychopy import event

        trialClock = self._startTrialClock()
        trialIndex = 0
        lastAdaptivePolarity = None
        nPer = int(self.questPlusTrialsPerPolarity)
        maxTrials = 2 * nPer + (2 * nPer // max(1, int(self.questPlusCatchEvery) or 10**9)) + 8

        testCastle = bool(getattr(self, 'staircaseTestCastleCelebration', False))
        print(
            '--> Quest+ starting (dual polarity Weibull, {n} trials/polarity, '
            'stim_scale={scale}, lapse={lapse:g}){test}'.format(
                n=nPer,
                scale=getattr(self, 'questPlusStimScale', 'linear'),
                lapse=float(self.questPlusLapseRate),
                test=' [TEST CASTLE: jump after 1 correct]' if testCastle else '',
            )
        )

        while (
            not (self._questPlusDone(1.0) and self._questPlusDone(-1.0))
            and trialIndex < maxTrials
        ):
            trialIndex += 1
            catchEvery = int(self.questPlusCatchEvery)
            isCatch = (
                (not testCastle)
                and catchEvery > 0
                and (trialIndex % catchEvery == 0)
            )

            directionPool = [self.deg0to360(d) for d in self.staircaseDirections]
            directionDeg = random.choice(directionPool)

            intensity = None
            slopeEst = float('nan')
            if isCatch:
                polarity = random.choice([1.0, -1.0])
                contrast = polarity * 1.0
            else:
                polarity = self._pickQuestPlusPolarity(lastAdaptivePolarity)
                if polarity is None:
                    break
                engine = engines[polarity]
                # Snap to grid so update() intensity matches the Quest+ domain exactly.
                intensity = self._snapQuestIntensity(engine.next_stim['intensity'])
                contrast = polarity * intensity
                lastAdaptivePolarity = polarity

            if not self._showFixation(win, fixationCrossArms, trialClock, self.staircaseITI):
                return False

            responseDir, correct, rt, reminderBeep, quitEarly = self._runStaircaseTrial(
                win, dots, dotDiameterPix, dotRadiusPix, ppd_h, ppd_v,
                contrast, directionDeg, trialClock,
            )
            if quitEarly:
                return False

            trialMagnitude = abs(contrast)
            stimLabel = 'up' if abs(self.deg0to360(directionDeg) - 90.0) < 1e-6 else 'down'
            if responseDir is None:
                respLabel = 'none'
            else:
                respLabel = 'up' if abs(self.deg0to360(responseDir) - 90.0) < 1e-6 else 'down'
            qpTrialIndex = 0
            nextIntensity = trialMagnitude
            thrPos, slopePos = self._questPlusEstimate(engines[1.0])
            thrNeg, slopeNeg = self._questPlusEstimate(engines[-1.0])
            runningThreshold = trialMagnitude
            runningSlope = float('nan')

            if not isCatch:
                engine = engines[polarity]
                response = 'Yes' if correct else 'No'
                try:
                    engine.update(intensity=float(intensity), response=response)
                except Exception as err:
                    print('*** Quest+ update failed:', err)
                else:
                    # Only count trials that successfully updated the posterior.
                    self._questPlusTrialCounts[polarity] = int(
                        self._questPlusTrialCounts.get(polarity, 0)
                    ) + 1
                qpTrialIndex = self._questPlusTrialCounts[polarity]
                thrPos, slopePos = self._questPlusEstimate(engines[1.0])
                thrNeg, slopeNeg = self._questPlusEstimate(engines[-1.0])
                runningThreshold = thrPos if polarity > 0 else thrNeg
                runningSlope = slopePos if polarity > 0 else slopeNeg
                nextIntensity = self._snapQuestIntensity(engine.next_stim['intensity'])

            rowPhase = 'catch' if isCatch else 'questplus'
            self._staircaseTrials.append({
                'trialIndex': trialIndex,
                'stairTrialIndex': qpTrialIndex if not isCatch else 0,
                'phase': rowPhase,
                'isCatch': int(isCatch),
                'contrast': contrast,
                'magnitude': trialMagnitude,
                'polarity': polarity,
                'directionDeg': directionDeg,
                'stimulusLabel': stimLabel,
                'responseDeg': float('nan') if responseDir is None else responseDir,
                'responseLabel': respLabel,
                'correct': int(correct),
                'reactionTimeSec': rt,
                'reminderBeep': int(reminderBeep),
                'stepDir': 0,
                'correctStreak': 0,
                'reversal': 0,
                'fineReversalCount': 0,
                'fineReversalCountPos': self._questPlusTrialCounts[1.0],
                'fineReversalCountNeg': self._questPlusTrialCounts[-1.0],
                'runningThreshold': runningThreshold,
                'runningThresholdPos': thrPos,
                'runningThresholdNeg': thrNeg,
                'runningSlope': runningSlope,
                'runningSlopePos': slopePos,
                'runningSlopeNeg': slopeNeg,
                'nextMagnitude': nextIntensity,
            })

            print(
                '--> Quest+ trial {i}: {phase} pol={pol:+.0f} c={c:+.4f} '
                '{stim}->{resp} {ok} RT={rt:.3f}s  '
                '(adapt +{ap}/-{an} of {need}; θ+={tp:.4f} θ-={tn:.4f})'.format(
                    i=trialIndex,
                    phase=rowPhase,
                    pol=polarity,
                    c=contrast,
                    stim=stimLabel,
                    resp=respLabel,
                    ok='CORRECT' if correct else ('TIMEOUT' if responseDir is None else 'WRONG'),
                    rt=rt,
                    ap=self._questPlusTrialCounts[1.0],
                    an=self._questPlusTrialCounts[-1.0],
                    need=nPer,
                    tp=thrPos,
                    tn=thrNeg,
                )
            )

            if testCastle and correct:
                print(
                    '--> TEST: one correct direction response — '
                    'ending Quest+ early for castle celebration'
                )
                break

        testMode = bool(testCastle and (
            self._questPlusTrialCounts[1.0] + self._questPlusTrialCounts[-1.0] < 2 * nPer
        ))
        self.thresholdContrastPositive, self.thresholdStatusPositive, slopeP = (
            self._finalizeQuestPlusPolarity(
                engines[1.0], self._questPlusTrialCounts[1.0], testMode=testMode,
            )
        )
        self.thresholdContrastNegative, self.thresholdStatusNegative, slopeN = (
            self._finalizeQuestPlusPolarity(
                engines[-1.0], self._questPlusTrialCounts[-1.0], testMode=testMode,
            )
        )
        self._questPlusFinalSlopes = {1.0: slopeP, -1.0: slopeN}

        self.thresholdContrast = self._clampMagnitude(
            10.0 ** (
                0.5 * (
                    math.log10(max(self.thresholdContrastPositive, float(self.questPlusMinContrast)))
                    + math.log10(max(self.thresholdContrastNegative, float(self.questPlusMinContrast)))
                )
            )
        )

        catches = [r for r in self._staircaseTrials if r.get('isCatch')]
        if not catches:
            self.staircaseCompliance = 'NA'
        else:
            catchAcc = sum(r['correct'] for r in catches) / float(len(catches))
            self.staircaseCompliance = (
                'ok' if catchAcc >= float(self.staircaseCatchMinAccuracy) else 'unreliable'
            )

        print(
            '--> Quest+ thresholds |contrast|: '
            '+theta={p:.6f} ({sp})  -theta={n:.6f} ({sn})  geo-mean={g:.6f}  '
            'compliance={comp}'.format(
                p=self.thresholdContrastPositive,
                sp=self.thresholdStatusPositive,
                n=self.thresholdContrastNegative,
                sn=self.thresholdStatusNegative,
                g=self.thresholdContrast,
                comp=self.staircaseCompliance,
            )
        )
        for label, status, slope in (
            ('Positive', self.thresholdStatusPositive, slopeP),
            ('Negative', self.thresholdStatusNegative, slopeN),
        ):
            if status == 'at_floor':
                print(
                    '--> {lab} polarity: at_floor (theta <= {floor:g}; '
                    'OKR will use floor)'.format(
                        lab=label, floor=float(self.questPlusMinContrast),
                    )
                )
            elif status == 'at_ceiling':
                thrVal = (
                    self.thresholdContrastPositive if label == 'Positive'
                    else self.thresholdContrastNegative
                )
                print(
                    '*** {lab} polarity near ceiling (theta≈{thr:.4f}); '
                    'interpret with caution.'.format(lab=label, thr=float(thrVal)),
                )
            elif status == 'incomplete':
                print('*** {lab} polarity incomplete.'.format(lab=label))
            else:
                print(
                    '--> {lab} polarity: status={status}, slope≈{s}'.format(
                        lab=label,
                        status=status,
                        s=('{:.3f}'.format(slope) if slope == slope else 'NA'),
                    )
                )
        if self.staircaseCompliance == 'unreliable':
            print('*** Catch-trial compliance unreliable — interpret thresholds with caution.')
        self._writeQuestPlusLog()
        return True


    def _writeQuestPlusLog(self):
        rows = getattr(self, '_staircaseTrials', None) or []
        if (
            not rows
            and self.thresholdContrast is None
            and self.thresholdContrastPositive is None
            and self.thresholdContrastNegative is None
        ):
            return None
        logDir = getattr(self, '_okrLogDir', None)
        logDir = Path.cwd() if logDir is None else Path(logDir)
        logDir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logPath = logDir / ('{name}_QuestPlus_{stamp}.txt'.format(
            name=self.protocolName, stamp=stamp,
        ))
        stats = self._staircaseSummaryStats(rows)
        self._staircaseSummary = stats
        slopes = getattr(self, '_questPlusFinalSlopes', {}) or {}

        def _fmt(v, digits=4):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return 'NA'
            if isinstance(v, float):
                return ('{0:.' + str(digits) + 'f}').format(v)
            return str(v)

        def _thr(v):
            return 'NA' if v is None else '{:.6f}'.format(float(v))

        header = [
            '# Contrast Dots Quest+ Log',
            '# StimulusName: Bassoon {name}'.format(name=self.protocolName),
            '# Method: dual-polarity 2AFC Quest+Weibull; '
            'stim_scale={scale}; min_entropy; posterior-mean; '
            '{n} adaptive trials/polarity; '
            'stim={stim:g}s + blank gray until answer; '
            'preTrialFixation={iti:g}s; '
            'lapse={lapse:g}; lowerAsymptote={la:g}'.format(
                scale=getattr(self, 'questPlusStimScale', 'linear'),
                n=int(self.questPlusTrialsPerPolarity),
                stim=float(getattr(self, 'staircaseStimDurationSec', 1.0)),
                iti=float(self.staircaseITI),
                lapse=float(self.questPlusLapseRate),
                la=float(self.questPlusLowerAsymptote),
            ),
            '# IntensityGrid: logspace {lo:g}..{hi:g} ({n} pts)'.format(
                lo=float(self.questPlusMinContrast),
                hi=float(self.questPlusMaxContrast),
                n=int(self.questPlusNIntensities),
            ),
            '# SlopeGrid: logspace {lo:g}..{hi:g} ({n} pts)'.format(
                lo=float(self.questPlusSlopeMin),
                hi=float(self.questPlusSlopeMax),
                n=int(self.questPlusNSlopes),
            ),
            '# ThresholdContrastPositive: {t}'.format(t=_thr(self.thresholdContrastPositive)),
            '# ThresholdStatusPositive: {s}'.format(s=self.thresholdStatusPositive or 'NA'),
            '# SlopePositive: {s}'.format(s=_fmt(slopes.get(1.0), 4)),
            '# ThresholdContrastNegative: {t}'.format(t=_thr(self.thresholdContrastNegative)),
            '# ThresholdStatusNegative: {s}'.format(s=self.thresholdStatusNegative or 'NA'),
            '# SlopeNegative: {s}'.format(s=_fmt(slopes.get(-1.0), 4)),
            '# ThresholdContrastGeoMean: {t}'.format(t=_thr(self.thresholdContrast)),
            '# StaircaseCompliance: {c}'.format(c=self.staircaseCompliance or 'NA'),
            '# ThresholdRule: completed = Quest+ posterior-mean threshold; '
            'at_floor = theta near min contrast (theta <= floor) for OKR; '
            'at_ceiling = theta near max contrast',
            '# reactionTimeSec: stimulus onset to up/down (mid-trial pauses excluded)',
            '# Summary nTrials={n} nAdaptive={na} nCatch={nc}'.format(
                n=stats.get('nTrials', 0),
                na=stats.get('nAdaptive', 0),
                nc=stats.get('nCatch', 0),
            ),
            '# Summary accAdaptive={aa} accCatch={ac} accUp={au} accDown={ad} '
            'accPosPol={ap} accNegPol={an}'.format(
                aa=_fmt(stats.get('accAdaptive'), 3),
                ac=_fmt(stats.get('accCatch'), 3),
                au=_fmt(stats.get('accUp'), 3),
                ad=_fmt(stats.get('accDown'), 3),
                ap=_fmt(stats.get('accPosPolarity'), 3),
                an=_fmt(stats.get('accNegPolarity'), 3),
            ),
            '# Summary meanRtAdaptive_s={m} medianRtAdaptive_s={med} '
            'meanRtCorrect_s={mc} meanRtIncorrect_s={mi} nReminderBeep={nb}'.format(
                m=_fmt(stats.get('meanRtAdaptive'), 3),
                med=_fmt(stats.get('medianRtAdaptive'), 3),
                mc=_fmt(stats.get('meanRtCorrect'), 3),
                mi=_fmt(stats.get('meanRtIncorrect'), 3),
                nb=stats.get('nReminderBeep', 0),
            ),
            'trialIndex\tquestTrialIndex\tphase\tisCatch\tcontrast\tmagnitude\tpolarity\t'
            'directionDeg\tstimulusLabel\tresponseDeg\tresponseLabel\tcorrect\t'
            'reactionTimeSec\treminderBeep\t'
            'nAdaptivePos\tnAdaptiveNeg\t'
            'runningThreshold\trunningThresholdPos\trunningThresholdNeg\t'
            'runningSlope\trunningSlopePos\trunningSlopeNeg\tnextMagnitude',
        ]
        body = []
        for row in rows:
            body.append('\t'.join([
                str(row['trialIndex']),
                str(row['stairTrialIndex']),
                str(row['phase']),
                str(row['isCatch']),
                '{:.6f}'.format(float(row['contrast'])),
                '{:.6f}'.format(float(row['magnitude'])),
                '{:.1f}'.format(float(row['polarity'])),
                '{:.4f}'.format(float(row['directionDeg'])),
                str(row['stimulusLabel']),
                (
                    'NA' if (
                        row.get('responseDeg') is None
                        or (isinstance(row.get('responseDeg'), float) and math.isnan(row['responseDeg']))
                    ) else '{:.4f}'.format(float(row['responseDeg']))
                ),
                str(row['responseLabel']),
                str(row['correct']),
                '{:.4f}'.format(float(row['reactionTimeSec'])),
                str(row['reminderBeep']),
                str(row.get('fineReversalCountPos', 0)),
                str(row.get('fineReversalCountNeg', 0)),
                '{:.6f}'.format(float(row['runningThreshold'])),
                '{:.6f}'.format(float(row['runningThresholdPos'])),
                '{:.6f}'.format(float(row['runningThresholdNeg'])),
                _fmt(row.get('runningSlope'), 4),
                _fmt(row.get('runningSlopePos'), 4),
                _fmt(row.get('runningSlopeNeg'), 4),
                '{:.6f}'.format(float(row['nextMagnitude'])),
            ]))
        logPath.write_text('\n'.join(header + body) + '\n', encoding='utf-8')
        print('--> Wrote Quest+ log:', logPath)
        # Also print discrimination summary like staircase.
        print(
            '--> Quest+ discrimination: adaptive acc={acc} catch acc={cacc} '
            'mean RT={rt}s median RT={med}s'.format(
                acc=_fmt(stats.get('accAdaptive'), 3),
                cacc=_fmt(stats.get('accCatch'), 3),
                rt=_fmt(stats.get('meanRtAdaptive'), 3),
                med=_fmt(stats.get('medianRtAdaptive'), 3),
            )
        )
        return logPath


    def run(self, win, informationWin):
        self._completed = 0
        self.thresholdContrast = None
        self.thresholdContrastPositive = None
        self.thresholdContrastNegative = None
        self.thresholdStatusPositive = None
        self.thresholdStatusNegative = None
        self.staircaseCompliance = None
        self._staircaseTrials = []
        self._questPlusTrialCounts = {1.0: 0, -1.0: 0}
        self.staircaseCatchEvery = int(self.questPlusCatchEvery)
        self.staircaseMinContrast = float(self.questPlusMinContrast)

        probesForOkr = bool(getattr(self, 'attentionProbe', False))
        self.attentionProbe = False

        print('--> Contrast Dots Quest+: starting 2AFC Quest+ threshold phase')
        try:
            if not self._runQuestPlusPhase(win, informationWin):
                print('*** Quest+ aborted')
                return

            if not self._pauseBeforeOkr(win):
                print('*** Quit before OKR phase')
                return

            self.contrasts = self._contrastsFromPolarityThresholds(
                self.thresholdContrastPositive, self.thresholdContrastNegative,
            )
            self.attentionProbe = probesForOkr
            print(
                '--> OKR phase: +theta={p:.4f} ({sp}) -theta={n:.4f} ({sn}), '
                'attentionProbe={probe}, contrasts={c}'.format(
                    p=self.thresholdContrastPositive,
                    sp=self.thresholdStatusPositive,
                    n=self.thresholdContrastNegative,
                    sn=self.thresholdStatusNegative,
                    probe=int(probesForOkr),
                    c=self.contrasts,
                )
            )
            self._initOkrFixationCountdownSounds()
            wasUserInitiated = self.userInitiated
            self.userInitiated = False
            try:
                # ContrastDots.run (skip staircase parent run)
                from protocols.ContrastDots import ContrastDots
                ContrastDots.run(self, win, informationWin)
            finally:
                self.userInitiated = wasUserInitiated
                self._okrFixationCountdownSounds = {}
        finally:
            self.attentionProbe = probesForOkr
