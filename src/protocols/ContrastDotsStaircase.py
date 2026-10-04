# -*- coding: utf-8 -*-
"""
Contrast Dots Staircase: 2AFC adaptive threshold then OKR at derived contrasts.

Phase 1 — Interleaved polarity staircases (3-down-1-up each):
  Separate adaptive tracks for +contrast and −contrast.
  Coarse: halve/double |contrast| until the first reversal.
  Fine: 0.15 log10 steps until 2 fine reversals per polarity
  (early stop allowed at the contrast floor).
  Targets ~79.4% correct.
  If a polarity reaches the contrast floor with no fine reversals
  (still performing well), status=at_floor and θ is set to the floor
  for OKR (left-censored: θ ≤ floor).
  Fixed ~1 s coherent motion, then a gray Up/Down prompt that stays until
  they answer on blank gray (answers also allowed during motion); short
  black fixation before each trial; marimba click on response.
  Catch trials at |contrast|=1.0 do not update either staircase.
  Once a polarity finishes, only the unfinished polarity is sampled.

Phase 2 — Stacked-jar castle checkpoint (no accuracy readout), then standard
  Contrast Dots OKR blocks at +2/4/8×θ+ and −2/4/8×θ− (capped at ±1.0).
  Attention probes (if enabled) apply only in this OKR phase, not during 2AFC.
  During OKR fixation, soft ticks play at 3, 2, and 1 s remaining.
"""
from __future__ import annotations

import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from psychopy import event, visual

from protocols.ContrastDots import ContrastDots

_STAIRCASE_SOUND_DIR = Path(__file__).resolve().parent / 'sounds'
_STAIRCASE_DIRECTION_SOUND = _STAIRCASE_SOUND_DIR / 'marimba_click.wav'
_STAIRCASE_YAY_SOUND = _STAIRCASE_SOUND_DIR / 'practice_yay.mp3'


class ContrastDotsStaircase(ContrastDots):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsStaircase'
        self.dotSizeDegrees = 1.0
        self.speed = 10.0
        self.directions = [90.0, 270.0]  # OKR phase directions
        # Default on; probes apply only during the OKR phase (not 2AFC).
        self.attentionProbe = True
        self.postStimTime = 0.0
        self.tailTime = 4.0
        # Placeholder until staircase finishes; replaced before OKR phase.
        self.contrasts = [1.0]

        # Staircase parameters (tuned shorter / less time in the guessing zone)
        self.staircaseStartContrast = 0.25
        self.staircaseMinContrast = 0.01  # floor: avoid long invisible guessing runs
        self.staircaseFineLogStep = 0.15  # log10 units
        self.staircaseFineReversals = 2
        self.staircaseCorrectToStepDown = 3  # 3-down-1-up (~79.4% correct)
        # With 2 fine reversals required, early-stop minimum matches full completion.
        self.staircaseEarlyStopMinFineReversals = 2
        self.staircaseCatchEvery = 12  # fewer catches; still a compliance check
        self.staircaseStimDurationSec = 0.5  # fixed coherent-motion presentation
        # Gray Up/Down screen after motion stays until they answer (no timeout).
        self.staircaseResponseReminderSec = 5.0  # soft chime if still waiting to answer
        self.staircaseITI = 0.5  # brief black fixation before each 2AFC trial
        # Allow both polarities to reach the floor under 3-down-1-up.
        self.staircaseMaxTrials = 56
        self.staircaseMaxAdaptivePerPolarity = 24
        # Trials at floor with no fine reversals before coding at_floor (still correct).
        self.staircaseFloorConfirmTrials = 3
        self.staircaseCatchMinAccuracy = 0.80
        self.staircaseDirections = [90.0, 270.0]  # 2AFC motion directions
        self.okrThresholdMultipliers = [2.0, 4.0, 8.0]
        # Soft ticks during OKR fixation when 3/2/1 s remain.
        self.okrFixationCountdownSec = 3
        self._okrFixationCountdownSounds = {}
        # Subject-facing stacked-jar castle between staircase and OKR (no θ readout).
        self.staircaseCastleCelebration = True
        # Testing only: after one correct up/down response, end staircase and show castle.
        self.staircaseTestCastleCelebration = False

        self.thresholdContrast = None  # geometric mean of polarity thresholds (summary)
        self.thresholdContrastPositive = None
        self.thresholdContrastNegative = None
        self.thresholdStatusPositive = None  # completed | completed_early | at_floor | incomplete | ...
        self.thresholdStatusNegative = None
        self.staircaseCompliance = None  # ok | unreliable | NA
        self._staircaseTrials = []
        self._staircaseSummary = {}
        self._confirmSound = None
        self._reminderJingle = None


    def _stimulusTitle(self):
        return 'Contrast Dots Staircase'


    def internalValidation(self):
        tf = True
        errorMessage = []
        start = float(self.staircaseStartContrast)
        minC = float(self.staircaseMinContrast)
        if not (0.0 < start <= 1.0):
            tf = False
            errorMessage.append('Staircase Start Contrast must be > 0 and ≤ 1.')
        if not (0.0 < minC <= 1.0):
            tf = False
            errorMessage.append('Staircase Min Contrast must be > 0 and ≤ 1.')
        if minC > start:
            tf = False
            errorMessage.append('Staircase Min Contrast must be ≤ Start Contrast.')
        if float(self.staircaseFineLogStep) <= 0:
            tf = False
            errorMessage.append('Staircase Fine Log Step must be greater than 0.')
        if int(self.staircaseFineReversals) < 1:
            tf = False
            errorMessage.append('Staircase Fine Reversals must be at least 1.')
        if int(self.staircaseCorrectToStepDown) < 1:
            tf = False
            errorMessage.append('Staircase Correct To Step Down must be at least 1.')
        earlyMin = int(self.staircaseEarlyStopMinFineReversals)
        if earlyMin < 1 or earlyMin > int(self.staircaseFineReversals):
            tf = False
            errorMessage.append(
                'Staircase Early Stop Min Fine Reversals must be between 1 and Fine Reversals.'
            )
        if int(self.staircaseMaxAdaptivePerPolarity) < int(self.staircaseFineReversals) + 4:
            tf = False
            errorMessage.append('Staircase Max Adaptive Per Polarity is too small.')
        if int(self.staircaseFloorConfirmTrials) < 1:
            tf = False
            errorMessage.append('Staircase Floor Confirm Trials must be at least 1.')
        catchMin = float(self.staircaseCatchMinAccuracy)
        if not (0.0 <= catchMin <= 1.0):
            tf = False
            errorMessage.append('Staircase Catch Min Accuracy must be between 0 and 1.')
        if int(self.staircaseCatchEvery) < 0:
            tf = False
            errorMessage.append('Staircase Catch Every must be 0 or greater.')
        if float(self.staircaseResponseReminderSec) <= 0:
            tf = False
            errorMessage.append('Staircase Response Reminder must be greater than 0 seconds.')
        if float(self.staircaseITI) < 0:
            tf = False
            errorMessage.append('Staircase ITI must be 0 or greater.')
        # Dual polarity: need room for two coarse→fine tracks (+ catches).
        minTrials = 2 * (int(self.staircaseFineReversals) + 6)
        if int(self.staircaseMaxTrials) < minTrials:
            tf = False
            errorMessage.append(
                'Staircase Max Trials is too small for dual-polarity staircases '
                '(need at least {n}).'.format(n=minTrials)
            )
        dirs = getattr(self, 'staircaseDirections', None) or []
        if not dirs:
            tf = False
            errorMessage.append('Staircase Directions must contain at least one direction.')
        mults = list(getattr(self, 'okrThresholdMultipliers', []) or [])
        if not mults or any(float(m) <= 0 for m in mults):
            tf = False
            errorMessage.append('OKR Threshold Multipliers must be a list of values > 0.')
        parentTf, parentErrors = super().internalValidation()
        # Parent may fail on contrasts placeholder — ignore contrast-list issues if only that.
        tf = tf and parentTf
        errorMessage += parentErrors
        return tf, errorMessage


    def estimateTime(self):
        # Rough upper bound: max staircase trials × (reminder + ITI) + OKR estimate.
        stairSec = int(self.staircaseMaxTrials) * (
            float(self.staircaseResponseReminderSec) + float(self.staircaseITI) + 2.0
        )
        # Assume threshold ~ start/4 for OKR time estimate before run.
        theta = max(float(self.staircaseMinContrast), float(self.staircaseStartContrast) / 4.0)
        self.contrasts = self._contrastsFromPolarityThresholds(theta, theta)
        okrSec = super().estimateTime()
        self._estimatedTime = stairSec + 5.0 + okrSec
        return self._estimatedTime


    def _contrastsFromPolarityThresholds(self, thetaPos, thetaNeg):
        '''
        Polarity-specific OKR levels: +m×θ+ and −m×θ− for each multiplier,
        clipped to ±1, unique.
        '''
        thetaPos = abs(float(thetaPos))
        thetaNeg = abs(float(thetaNeg))
        levels = []
        seen = set()
        for m in self.okrThresholdMultipliers:
            magP = min(1.0, abs(float(m)) * thetaPos)
            magN = min(1.0, abs(float(m)) * thetaNeg)
            for c in (magP, -magN):
                key = round(c, 6)
                if key not in seen:
                    seen.add(key)
                    levels.append(c)
        if not levels:
            levels = [1.0, -1.0]
        return levels


    def _contrastsFromThreshold(self, theta):
        '''Backward-compatible helper: same θ for both polarities.'''
        return self._contrastsFromPolarityThresholds(theta, theta)


    def _newStairState(self):
        return {
            'magnitude': self._clampMagnitude(self.staircaseStartContrast),
            'phase': 'coarse',
            'correctStreak': 0,
            'lastStepDir': None,  # -1 harder, +1 easier
            'fineRevs': [],
            'stairTrialIndex': 0,
            'trialsAtFloor': 0,
            'forcedDone': False,
            'doneReason': None,
        }


    def _isAtFloor(self, magnitude):
        return float(magnitude) <= float(self.staircaseMinContrast) * 1.001


    def _thresholdFromReversals(self, fineRevs, fallbackMagnitude):
        if fineRevs:
            logs = [math.log10(max(c, float(self.staircaseMinContrast))) for c in fineRevs]
            return self._clampMagnitude(10.0 ** (sum(logs) / len(logs)))
        return self._clampMagnitude(fallbackMagnitude)


    def _polarityDone(self, state):
        if state.get('forcedDone'):
            return True
        return len(state['fineRevs']) >= int(self.staircaseFineReversals)


    def _finalizePolarityResult(self, state):
        '''
        Returns (thetaForOkr, status).

        status:
          completed / completed_early — reversal-based θ
          at_floor — no usable fine reversals; θ set to contrast floor (θ ≤ floor)
          incomplete_few_reversals — some fine reversals but below early-stop minimum
          incomplete — stopped without floor or enough reversals
        '''
        nFine = len(state['fineRevs'])
        need = int(self.staircaseFineReversals)
        earlyNeed = int(self.staircaseEarlyStopMinFineReversals)
        floor = float(self.staircaseMinContrast)
        atFloor = self._isAtFloor(state['magnitude']) or state.get('doneReason') == 'at_floor'

        if nFine >= need:
            return self._thresholdFromReversals(state['fineRevs'], state['magnitude']), 'completed'
        if nFine >= earlyNeed:
            return self._thresholdFromReversals(state['fineRevs'], state['magnitude']), 'completed_early'
        if nFine == 0 and atFloor:
            return self._clampMagnitude(floor), 'at_floor'
        if state.get('doneReason') == 'test_castle':
            return self._clampMagnitude(state['magnitude']), 'test_castle'
        if nFine > 0:
            return (
                self._thresholdFromReversals(state['fineRevs'], state['magnitude']),
                'incomplete_few_reversals',
            )
        return self._clampMagnitude(state['magnitude']), 'incomplete'


    def _maybeEarlyStopPolarity(self, state, correct, trialMagnitude, polarity):
        '''
        End a polarity when:
          - at floor with no fine reversals after enough floor trials (controls),
          - fine phase + enough reversals + error at floor (guessing zone),
          - trial budget only if already at floor or in fine phase.
        '''
        if self._polarityDone(state):
            return
        polTag = '+' if polarity > 0 else '-'
        nFine = len(state['fineRevs'])
        earlyNeed = int(self.staircaseEarlyStopMinFineReversals)
        atFloor = self._isAtFloor(trialMagnitude) or self._isAtFloor(state['magnitude'])
        if atFloor:
            state['trialsAtFloor'] = int(state.get('trialsAtFloor', 0)) + 1

        # Strong performers: stop once clearly still correct at the floor (no reversals).
        if (
            atFloor
            and nFine == 0
            and state['phase'] == 'coarse'
            and state['trialsAtFloor'] >= int(self.staircaseFloorConfirmTrials)
        ):
            state['forcedDone'] = True
            state['doneReason'] = 'at_floor'
            print(
                '--> Staircase ({pol}): at contrast floor with no reversals '
                'after {n} floor trials (theta <= {floor:g})'.format(
                    pol=polTag,
                    n=state['trialsAtFloor'],
                    floor=float(self.staircaseMinContrast),
                )
            )
            return

        # Budget applies only after reaching floor or fine phase (so descent can finish).
        if state['stairTrialIndex'] >= int(self.staircaseMaxAdaptivePerPolarity):
            if state['phase'] == 'fine' or atFloor or nFine >= 1:
                state['forcedDone'] = True
                if atFloor and nFine == 0:
                    state['doneReason'] = 'at_floor'
                else:
                    state['doneReason'] = 'max_adaptive_per_polarity'
                print(
                    '--> Staircase ({pol}): stop after {n} adaptive trials '
                    '({fine} fine reversals, reason={reason})'.format(
                        pol=polTag,
                        n=state['stairTrialIndex'],
                        fine=nFine,
                        reason=state['doneReason'],
                    )
                )
                return

        if (
            state['phase'] == 'fine'
            and nFine >= earlyNeed
            and atFloor
            and not correct
        ):
            state['forcedDone'] = True
            state['doneReason'] = 'floor_guessing'
            print(
                '--> Staircase ({pol}): early stop at contrast floor '
                'after {fine} fine reversals (guessing zone)'.format(
                    pol=polTag, fine=nFine,
                )
            )


    def _pickAdaptivePolarity(self, states, lastPolarity):
        '''Prefer unfinished polarities; alternate when both still open.'''
        openOnes = [p for p in (1.0, -1.0) if not self._polarityDone(states[p])]
        if not openOnes:
            return None
        if len(openOnes) == 1:
            return openOnes[0]
        if lastPolarity is None:
            return random.choice(openOnes)
        other = -float(lastPolarity)
        return other if other in openOnes else openOnes[0]


    def _applyStaircaseStep(self, state, correct, trialMagnitude):
        '''
        Update one polarity's staircase from a non-catch trial.
        Returns (reversal: bool, stepDir: int).
        '''
        state['stairTrialIndex'] += 1
        stepDir = 0
        needCorrect = int(self.staircaseCorrectToStepDown)
        if correct:
            state['correctStreak'] += 1
            if state['correctStreak'] >= needCorrect:
                stepDir = -1  # harder
                state['correctStreak'] = 0
        else:
            state['correctStreak'] = 0
            stepDir = 1  # easier

        reversal = False
        if stepDir == 0:
            return reversal, stepDir

        last = state['lastStepDir']
        if last is not None and stepDir != last:
            reversal = True
            if state['phase'] == 'coarse':
                state['phase'] = 'fine'
                print(
                    '--> Staircase: first reversal at |c|={c:g} -> fine phase'.format(
                        c=trialMagnitude,
                    )
                )
            else:
                state['fineRevs'].append(trialMagnitude)
                print(
                    '--> Staircase: fine reversal {n}/{need} at |c|={c:g}'.format(
                        n=len(state['fineRevs']),
                        need=int(self.staircaseFineReversals),
                        c=trialMagnitude,
                    )
                )

        if state['phase'] == 'coarse':
            if stepDir < 0:
                state['magnitude'] = self._clampMagnitude(state['magnitude'] / 2.0)
            else:
                state['magnitude'] = self._clampMagnitude(state['magnitude'] * 2.0)
        else:
            logC = math.log10(max(state['magnitude'], float(self.staircaseMinContrast)))
            logC += (-1 if stepDir < 0 else 1) * float(self.staircaseFineLogStep)
            state['magnitude'] = self._clampMagnitude(10.0 ** logC)
        state['lastStepDir'] = stepDir
        return reversal, stepDir


    def _makeToneSound(
        self, freqs, noteDur=0.10, gapDur=0.02, volume=0.55,
        name='staircaseTone', soft=False,
    ):
        '''Short multi-note tone (Practice-style). Returns Sound or None.'''
        try:
            from psychopy import sound
        except Exception as err:
            print('*** Staircase sound unavailable ({0}).'.format(err))
            return None
        sampleRate = 44100
        pieces = []
        for freq in freqs:
            n = max(2, int(sampleRate * noteDur))
            t = np.linspace(0.0, noteDur, n, endpoint=False)
            env = np.ones(n)
            if soft:
                attack = max(1, int(0.04 * sampleRate))
                release = max(1, int(0.12 * sampleRate))
                harm = 0.04
            else:
                attack = max(1, int(0.008 * sampleRate))
                release = max(1, int(0.035 * sampleRate))
                harm = 0.16
            # Keep attack/release inside the note (short soft ticks were crashing here).
            if attack + release > n:
                attack = max(1, n // 3)
                release = max(1, n - attack)
            env[:attack] *= np.linspace(0.0, 1.0, attack)
            env[-release:] *= np.linspace(1.0, 0.0, release)
            wave = volume * np.sin(2 * np.pi * freq * t)
            wave += harm * volume * np.sin(2 * np.pi * 3 * freq * t)
            wave *= env
            pieces.append(wave)
            pieces.append(np.zeros(int(gapDur * sampleRate)))
        mono = np.concatenate(pieces).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        try:
            return sound.Sound(value=stereo, sampleRate=sampleRate, stereo=True, name=name)
        except Exception as err:
            print('*** Staircase sound failed ({0}).'.format(err))
            return None


    def _loadFileSound(self, path, volume=0.55, name='staircaseFileSound'):
        '''Load an mp3/wav; return Sound or None.'''
        try:
            from psychopy import sound
        except Exception as err:
            print('*** Staircase sound unavailable ({0}).'.format(err))
            return None
        path = Path(path)
        if not path.is_file():
            print('*** Staircase sound missing:', path)
            return None
        try:
            snd = sound.Sound(value=str(path), name=name)
            try:
                snd.setVolume(float(volume))
            except Exception:
                pass
            return snd
        except Exception as err:
            print('*** Staircase file sound failed ({0}): {1}'.format(path.name, err))
            return None


    def _initStaircaseSounds(self):
        # Marimba click on up/down (same for correct/incorrect — no performance feedback).
        self._confirmSound = self._loadFileSound(
            _STAIRCASE_DIRECTION_SOUND, volume=0.55, name='staircaseDirection',
        )
        if self._confirmSound is None:
            self._confirmSound = self._makeToneSound(
                [880.0], noteDur=0.06, gapDur=0.0, volume=0.40, name='staircaseConfirm',
            )
        # Soft two-note chime if no ↑/↓ within staircaseResponseReminderSec.
        self._reminderJingle = self._makeToneSound(
            [523.25, 659.25],  # C5 → E5
            noteDur=0.22,
            gapDur=0.06,
            volume=0.22,
            name='staircaseReminderJingle',
            soft=True,
        )


    def _playSound(self, snd):
        if snd is None:
            return
        try:
            snd.stop()
            snd.play()
        except Exception:
            pass


    def _playConfirmSound(self):
        self._playSound(getattr(self, '_confirmSound', None))


    def _playReminderBeep(self):
        self._playSound(getattr(self, '_reminderJingle', None))


    def _initOkrFixationCountdownSounds(self):
        '''Soft ticks for 3 / 2 / 1 s remaining on the fixation cross.'''
        self._okrFixationCountdownSounds = {}
        # Slightly rising pitches so the sequence feels like a ready cue.
        pitchBySec = {3: 660.0, 2: 770.0, 1: 880.0}
        n = max(1, int(self.okrFixationCountdownSec))
        for sec in range(n, 0, -1):
            freq = pitchBySec.get(sec, 700.0 + (n - sec) * 80.0)
            self._okrFixationCountdownSounds[sec] = self._makeToneSound(
                [freq],
                noteDur=0.07,
                gapDur=0.0,
                volume=0.28,
                name='okrFixationCountdown{0}'.format(sec),
                soft=True,
            )


    def _onFixationFrame(self, frameIndex, nFrames):
        '''Beep once at the start of each remaining second: 3, then 2, then 1.'''
        nCount = int(getattr(self, 'okrFixationCountdownSec', 0) or 0)
        if nCount <= 0 or nFrames <= 0:
            return
        fr = float(getattr(self, '_FR', 0) or 0)
        if fr <= 0:
            return
        sounds = getattr(self, '_okrFixationCountdownSounds', None) or {}
        for sec in range(nCount, 0, -1):
            targetFrame = int(nFrames) - int(round(sec * fr))
            if targetFrame < 0:
                continue
            if int(frameIndex) == targetFrame:
                self._playSound(sounds.get(sec))
                break


    def _clampMagnitude(self, magnitude):
        return float(min(1.0, max(float(self.staircaseMinContrast), float(magnitude))))


    def _setupDotsAndCross(self, win):
        self.getFR(win)
        if not hasattr(self, 'postStimTime'):
            self.postStimTime = 0.0
        self._interStimulusIntervalNumFrames = round(self._FR * self.interStimulusInterval)
        self._postStimTimeNumFrames = round(self._FR * self.postStimTime)
        ppd_h, ppd_v, dotDiameterPix, _probeDiameterPix, dotRadiusPix = (
            self._stimulusScaleFromWin(win)
        )
        self.dotCoords = np.zeros((self.numberOfDots, 2))
        dots = visual.ElementArrayStim(
            win,
            units='pix',
            nElements=self.numberOfDots,
            elementMask='circle',
            elementTex=None,
            xys=self.dotCoords.tolist(),
            sizes=dotDiameterPix,
            colors=self.dotColor,
        )
        crossHalfX = 0.5 * float(self.fixationCrossSizeDegrees) * ppd_h
        crossHalfY = 0.5 * float(self.fixationCrossSizeDegrees) * ppd_v
        crossLineWidth = max(2.0, min(crossHalfX, crossHalfY) * 2.0 * 0.15)
        fixationCrossArms = (
            visual.ShapeStim(
                win, units='pix',
                vertices=((-crossHalfX, 0.0), (crossHalfX, 0.0)),
                lineWidth=crossLineWidth, closeShape=False,
                lineColor=self.fixationCrossColor,
            ),
            visual.ShapeStim(
                win, units='pix',
                vertices=((0.0, -crossHalfY), (0.0, crossHalfY)),
                lineWidth=crossLineWidth, closeShape=False,
                lineColor=self.fixationCrossColor,
            ),
        )
        return ppd_h, ppd_v, dotDiameterPix, dotRadiusPix, dots, fixationCrossArms


    def _refreshTrialGeometry(self, win, dots, fixationCrossArms):
        '''Recompute ppd / sizes after a window resize (windowed capture).'''
        ppd_h, ppd_v, dotDiameterPix, _probe, dotRadiusPix = self._stimulusScaleFromWin(win)
        self._syncFixationCrossGeometry(fixationCrossArms, ppd_h, ppd_v)
        try:
            dots.sizes = dotDiameterPix
        except Exception:
            pass
        return ppd_h, ppd_v, dotDiameterPix, dotRadiusPix


    def _showFixation(self, win, fixationCrossArms, trialClock, durationSec):
        nFrames = max(0, int(round(self._FR * float(durationSec))))
        win.color = self.backgroundColor
        self._staircaseFixationCrossArms = fixationCrossArms
        for _ in range(nFrames):
            if self._winSizeKey(win) != getattr(self, '_lastWinSizeKey', None):
                ppd_h, ppd_v, _dot, _probe, _rad = self._stimulusScaleFromWin(win)
                self._syncFixationCrossGeometry(fixationCrossArms, ppd_h, ppd_v)
            for arm in fixationCrossArms:
                arm.draw()
            win.flip()
            if self.checkQuitOrPause():
                return False
        return True


    def _pollStaircaseResponseKeys(self, pauseSecRef):
        '''
        Handle q / pause / up / down during a 2AFC trial.
        pauseSecRef is a one-element list mutated with added pause time.
        Returns (responseDir or None, quitEarly).
        '''
        keys = [k.lower() for k in event.getKeys()]
        if not keys:
            return None, False
        if 'q' in keys:
            self._stoppedEarly = 1
            return None, True
        if 'p' in keys:
            self._userPauseCount += 1
            print('*** STIMULUS HAS PAUSED. Press any key to resume')
            startPause = time.time()
            event.waitKeys()
            dur = time.time() - startPause
            pauseSecRef[0] += dur
            self._userPauseDurations.append(dur)
            return None, False
        if 'up' in keys:
            return 90.0, False
        if 'down' in keys:
            return 270.0, False
        return None, False


    def _runStaircaseTrial(
        self, win, dots, dotDiameterPix, dotRadiusPix, ppd_h, ppd_v,
        contrast, directionDeg, trialClock,
    ):
        '''
        Fixed-duration coherent motion, then blank gray until they answer.
        Responses are accepted during motion and during the open-ended response screen.
        No response timeout (experimenter can verbally encourage an answer).

        Returns (responseDir, correct, reactionTimeSec, reminderBeep, quit).
        reactionTimeSec excludes mid-trial pause time.
        '''
        stimSec = float(getattr(self, 'staircaseStimDurationSec', 1.0) or 1.0)
        stimFrames = max(1, int(round(self._FR * stimSec)))

        ppd_h, ppd_v, dotDiameterPix, dotRadiusPix = self._refreshTrialGeometry(
            win, dots, getattr(self, '_staircaseFixationCrossArms', None),
        )
        directionRad = math.radians(directionDeg)
        speedComponents = np.array([
            self.speed * ppd_h * (1 / self._FR) * math.cos(directionRad),
            self.speed * ppd_v * (1 / self._FR) * math.sin(directionRad),
        ])
        self.initDotPositions(win, dotRadiusPix)
        if not self._usePersistentDots():
            self.initDotSpawnStagger()
        dotLifetimeFrames = max(1, round(self._FR * float(self.dotLifetime)))
        colors = self.dotColorAtContrast(contrast)
        dots.colors = colors
        dots.sizes = dotDiameterPix

        win.color = self.backgroundColor
        event.clearEvents()
        t0 = trialClock.getTime()
        pauseSecRef = [0.0]
        reminded = False
        responseDir = None

        def elapsedNow():
            return trialClock.getTime() - t0 - pauseSecRef[0]

        # --- Motion epoch ---
        for _ in range(stimFrames):
            if self._winSizeKey(win) != getattr(self, '_lastWinSizeKey', None):
                ppd_h, ppd_v, dotDiameterPix, dotRadiusPix = self._refreshTrialGeometry(
                    win, dots, getattr(self, '_staircaseFixationCrossArms', None),
                )
                speedComponents = np.array([
                    self.speed * ppd_h * (1 / self._FR) * math.cos(directionRad),
                    self.speed * ppd_v * (1 / self._FR) * math.sin(directionRad),
                ])
                colors = self.dotColorAtContrast(contrast)
                dots.colors = colors
            if not self._usePersistentDots():
                self.currentFrames += 1
                for dot in range(self.numberOfDots):
                    if self.currentFrames[dot] >= dotLifetimeFrames:
                        self.respawnDot(win, dotRadiusPix, dot=dot)

            self.dotCoords[:, 0] += speedComponents[0]
            self.dotCoords[:, 1] += speedComponents[1]
            self._afterDotMotion(win, dotRadiusPix, speedComponents)
            dots.xys = self.dotCoords.tolist()
            dots.colors = colors
            self._renderDotsFrame(win, dots)
            win.flip()

            elapsed = elapsedNow()
            if (not reminded) and elapsed >= float(self.staircaseResponseReminderSec):
                self._playReminderBeep()
                reminded = True

            responseDir, quitEarly = self._pollStaircaseResponseKeys(pauseSecRef)
            if quitEarly:
                return None, False, elapsed, int(reminded), True
            if responseDir is not None:
                break

        # --- Blank gray until they answer (no on-screen prompt) ---
        while responseDir is None:
            win.color = self.backgroundColor
            win.flip()

            elapsed = elapsedNow()
            if (not reminded) and elapsed >= float(self.staircaseResponseReminderSec):
                self._playReminderBeep()
                reminded = True

            responseDir, quitEarly = self._pollStaircaseResponseKeys(pauseSecRef)
            if quitEarly:
                return None, False, elapsed, int(reminded), True

        rt = elapsedNow()
        self._playConfirmSound()
        correct = abs(self.deg0to360(responseDir) - self.deg0to360(directionDeg)) < 1e-6
        return responseDir, correct, rt, int(reminded), False


    def _runStaircasePhase(self, win, informationWin):
        '''
        Dual interleaved 2AFC staircases (+/− polarity).
        Sets thresholdContrastPositive / Negative / combined.
        Returns False if aborted.
        '''
        self._informationWin = informationWin
        self._staircaseTrials = []
        self._initStaircaseSounds()
        ppd_h, ppd_v, dotDiameterPix, dotRadiusPix, dots, fixationCrossArms = (
            self._setupDotsAndCross(win)
        )
        self._staircaseFixationCrossArms = fixationCrossArms
        self._initPerRunStimulus(win, (ppd_h, ppd_v))

        if self.userInitiated:
            self.showInformationText(
                win,
                'Stimulus Information: {title}\n'
                'Judge motion direction — brief moving dots, then UP or DOWN\n'
                'Press any key to begin staircase'.format(title=self._stimulusTitle()),
            )
            event.waitKeys()

        trialClock = self._startTrialClock()
        states = {
            1.0: self._newStairState(),
            -1.0: self._newStairState(),
        }
        trialIndex = 0
        lastAdaptivePolarity = None
        need = int(self.staircaseFineReversals)

        testCastle = bool(getattr(self, 'staircaseTestCastleCelebration', False))
        print(
            '--> Staircase starting (dual polarity, {n}-down-1-up, '
            'coarse then fine 0.15 log10){test}'.format(
                n=int(self.staircaseCorrectToStepDown),
                test=' [TEST CASTLE: jump after 1 correct]' if testCastle else '',
            )
        )

        while (
            not (self._polarityDone(states[1.0]) and self._polarityDone(states[-1.0]))
            and trialIndex < int(self.staircaseMaxTrials)
        ):
            trialIndex += 1
            catchEvery = int(self.staircaseCatchEvery)
            # In castle test mode, never insert catches — first correct ends the phase.
            isCatch = (
                (not testCastle)
                and catchEvery > 0
                and (trialIndex % catchEvery == 0)
            )

            directionPool = [self.deg0to360(d) for d in self.staircaseDirections]
            directionDeg = random.choice(directionPool)

            if isCatch:
                polarity = random.choice([1.0, -1.0])
                contrast = polarity * 1.0
                state = None
            else:
                polarity = self._pickAdaptivePolarity(states, lastAdaptivePolarity)
                if polarity is None:
                    break
                state = states[polarity]
                contrast = polarity * state['magnitude']
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
            reversal = False
            stepDir = 0
            phaseAtTrial = 'catch'
            stairTrialIndex = 0
            correctStreak = 0
            nextMagnitude = trialMagnitude
            fineCount = 0
            runningThreshold = trialMagnitude

            if not isCatch and state is not None:
                phaseAtTrial = state['phase']
                reversal, stepDir = self._applyStaircaseStep(state, correct, trialMagnitude)
                self._maybeEarlyStopPolarity(state, correct, trialMagnitude, polarity)
                stairTrialIndex = state['stairTrialIndex']
                correctStreak = state['correctStreak']
                nextMagnitude = state['magnitude']
                fineCount = len(state['fineRevs'])
                runningThreshold = self._thresholdFromReversals(
                    state['fineRevs'], state['magnitude'],
                )

            rowPhase = 'catch' if isCatch else phaseAtTrial
            self._staircaseTrials.append({
                'trialIndex': trialIndex,
                'stairTrialIndex': stairTrialIndex if not isCatch else 0,
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
                'stepDir': int(stepDir),
                'correctStreak': int(correctStreak),
                'reversal': int(reversal),
                'fineReversalCount': fineCount,
                'fineReversalCountPos': len(states[1.0]['fineRevs']),
                'fineReversalCountNeg': len(states[-1.0]['fineRevs']),
                'runningThreshold': runningThreshold,
                'runningThresholdPos': self._thresholdFromReversals(
                    states[1.0]['fineRevs'], states[1.0]['magnitude'],
                ),
                'runningThresholdNeg': self._thresholdFromReversals(
                    states[-1.0]['fineRevs'], states[-1.0]['magnitude'],
                ),
                'nextMagnitude': nextMagnitude,
            })

            print(
                '--> Stair trial {i}: {phase} pol={pol:+.0f} c={c:+.4f} '
                '{stim}->{resp} {ok} RT={rt:.3f}s{rev}  '
                '(fine +{fp}/-{fn} of {need})'.format(
                    i=trialIndex,
                    phase=rowPhase,
                    pol=polarity,
                    c=contrast,
                    stim=stimLabel,
                    resp=respLabel,
                    ok='CORRECT' if correct else ('TIMEOUT' if responseDir is None else 'WRONG'),
                    rt=rt,
                    rev=' REVERSAL' if reversal else '',
                    fp=len(states[1.0]['fineRevs']),
                    fn=len(states[-1.0]['fineRevs']),
                    need=need,
                )
            )

            if testCastle and correct:
                print(
                    '--> TEST: one correct direction response — '
                    'ending staircase early for castle celebration'
                )
                for pol in (1.0, -1.0):
                    states[pol]['forcedDone'] = True
                    states[pol]['doneReason'] = 'test_castle'
                break

        # If global max trials hit mid-descent but a polarity is already at floor
        # with no reversals, finalize as at_floor.
        for pol in (1.0, -1.0):
            st = states[pol]
            if (
                not self._polarityDone(st)
                and len(st['fineRevs']) == 0
                and self._isAtFloor(st['magnitude'])
            ):
                st['forcedDone'] = True
                st['doneReason'] = 'at_floor'

        self.thresholdContrastPositive, self.thresholdStatusPositive = (
            self._finalizePolarityResult(states[1.0])
        )
        self.thresholdContrastNegative, self.thresholdStatusNegative = (
            self._finalizePolarityResult(states[-1.0])
        )
        # Summary θ: geometric mean of the two polarity OKR θ values.
        self.thresholdContrast = self._clampMagnitude(
            10.0 ** (
                0.5 * (
                    math.log10(max(self.thresholdContrastPositive, float(self.staircaseMinContrast)))
                    + math.log10(max(self.thresholdContrastNegative, float(self.staircaseMinContrast)))
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
            '--> Staircase thresholds |contrast|: '
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
        for pol, label, status in (
            (1.0, 'positive', self.thresholdStatusPositive),
            (-1.0, 'negative', self.thresholdStatusNegative),
        ):
            st = states[pol]
            if status == 'at_floor':
                print(
                    '--> {lab} polarity: at_floor (theta <= {floor:g}; '
                    'OKR will use floor)'.format(
                        lab=label.capitalize(),
                        floor=float(self.staircaseMinContrast),
                    )
                )
            elif status.startswith('incomplete'):
                print(
                    '*** {lab} polarity incomplete ({n}/{need} fine reversals, '
                    'status={status}).'.format(
                        lab=label.capitalize(),
                        n=len(st['fineRevs']),
                        need=need,
                        status=status,
                    )
                )
        if self.staircaseCompliance == 'unreliable':
            print('*** Catch-trial compliance unreliable — interpret thresholds with caution.')
        self._writeStaircaseLog()
        return True


    def _staircaseSummaryStats(self, rows):
        '''Aggregate accuracy / RT for log header and pause screen.'''
        if not rows:
            return {}
        adaptive = [r for r in rows if not r['isCatch']]
        catches = [r for r in rows if r['isCatch']]

        def _acc(subset):
            if not subset:
                return float('nan')
            return sum(r['correct'] for r in subset) / len(subset)

        def _mean_rt(subset):
            if not subset:
                return float('nan')
            return sum(float(r['reactionTimeSec']) for r in subset) / len(subset)

        def _median_rt(subset):
            if not subset:
                return float('nan')
            vals = sorted(float(r['reactionTimeSec']) for r in subset)
            mid = len(vals) // 2
            if len(vals) % 2:
                return vals[mid]
            return 0.5 * (vals[mid - 1] + vals[mid])

        upTrials = [r for r in adaptive if r['stimulusLabel'] == 'up']
        downTrials = [r for r in adaptive if r['stimulusLabel'] == 'down']
        posPol = [r for r in adaptive if float(r['polarity']) > 0]
        negPol = [r for r in adaptive if float(r['polarity']) < 0]
        correctAdaptive = [r for r in adaptive if r['correct']]
        incorrectAdaptive = [r for r in adaptive if not r['correct']]

        return {
            'nTrials': len(rows),
            'nAdaptive': len(adaptive),
            'nCatch': len(catches),
            'accOverall': _acc(rows),
            'accAdaptive': _acc(adaptive),
            'accCatch': _acc(catches),
            'accUp': _acc(upTrials),
            'accDown': _acc(downTrials),
            'accPosPolarity': _acc(posPol),
            'accNegPolarity': _acc(negPol),
            'meanRtAdaptive': _mean_rt(adaptive),
            'medianRtAdaptive': _median_rt(adaptive),
            'meanRtCorrect': _mean_rt(correctAdaptive),
            'meanRtIncorrect': _mean_rt(incorrectAdaptive),
            'meanRtCatch': _mean_rt(catches),
            'nReminderBeep': sum(int(r.get('reminderBeep', 0)) for r in rows),
        }


    def _writeStaircaseLog(self):
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
        logPath = logDir / ('{name}_Staircase_{stamp}.txt'.format(
            name=self.protocolName, stamp=stamp,
        ))
        stats = self._staircaseSummaryStats(rows)
        self._staircaseSummary = stats

        def _fmt(v, digits=4):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return 'NA'
            if isinstance(v, float):
                return ('{0:.' + str(digits) + 'f}').format(v)
            return str(v)

        def _thr(v):
            return 'NA' if v is None else '{:.6f}'.format(float(v))

        header = [
            '# Contrast Dots Staircase Log',
            '# StimulusName: Bassoon {name}'.format(name=self.protocolName),
            '# Method: dual-polarity 2AFC {down}-down-1-up; coarse halve; '
            'fine {log:g} log10; {n} fine reversals per polarity; '
            'stim={stim:g}s + open response screen until answer; '
            'preTrialFixation={iti:g}s'.format(
                down=int(self.staircaseCorrectToStepDown),
                log=float(self.staircaseFineLogStep),
                n=int(self.staircaseFineReversals),
                stim=float(getattr(self, 'staircaseStimDurationSec', 1.0)),
                iti=float(self.staircaseITI),
            ),
            '# ThresholdContrastPositive: {t}'.format(t=_thr(self.thresholdContrastPositive)),
            '# ThresholdStatusPositive: {s}'.format(
                s=self.thresholdStatusPositive or 'NA',
            ),
            '# ThresholdContrastNegative: {t}'.format(t=_thr(self.thresholdContrastNegative)),
            '# ThresholdStatusNegative: {s}'.format(
                s=self.thresholdStatusNegative or 'NA',
            ),
            '# ThresholdContrastGeoMean: {t}'.format(t=_thr(self.thresholdContrast)),
            '# StaircaseCompliance: {c}'.format(c=self.staircaseCompliance or 'NA'),
            '# ThresholdRule: completed = mean of fine-phase reversal magnitudes in log10 space; '
            'at_floor = no fine reversals at contrast floor, theta set to floor (theta <= floor) for OKR',
            '# reactionTimeSec: stimulus onset to up/down (mid-trial pauses excluded)',
            '# Offline discrimination: use magnitude vs correct within polarity; '
            'stimulusLabel/responseLabel (confusion); isCatch (compliance); '
            'reactionTimeSec (speed); treat at_floor as left-censored',
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
            'trialIndex\tstairTrialIndex\tphase\tisCatch\tcontrast\tmagnitude\tpolarity\t'
            'directionDeg\tstimulusLabel\tresponseDeg\tresponseLabel\tcorrect\t'
            'reactionTimeSec\treminderBeep\tstepDir\tcorrectStreak\treversal\t'
            'fineReversalCount\tfineReversalCountPos\tfineReversalCountNeg\t'
            'runningThreshold\trunningThresholdPos\trunningThresholdNeg\tnextMagnitude',
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
                str(row['stepDir']),
                str(row['correctStreak']),
                str(row['reversal']),
                str(row['fineReversalCount']),
                str(row['fineReversalCountPos']),
                str(row['fineReversalCountNeg']),
                '{:.6f}'.format(float(row['runningThreshold'])),
                '{:.6f}'.format(float(row['runningThresholdPos'])),
                '{:.6f}'.format(float(row['runningThresholdNeg'])),
                '{:.6f}'.format(float(row['nextMagnitude'])),
            ]))
        logPath.write_text('\n'.join(header + body) + '\n', encoding='utf-8')
        print('--> Wrote staircase log:', logPath)
        if stats:
            print(
                '--> Staircase discrimination: adaptive acc={aa:.1%} catch acc={ac} '
                'mean RT={rt:.3f}s median RT={med:.3f}s'.format(
                    aa=stats['accAdaptive'] if not math.isnan(stats['accAdaptive']) else 0.0,
                    ac=(
                        '{:.1%}'.format(stats['accCatch'])
                        if not math.isnan(stats.get('accCatch', float('nan')))
                        else 'NA'
                    ),
                    rt=stats['meanRtAdaptive'] if not math.isnan(stats['meanRtAdaptive']) else float('nan'),
                    med=stats['medianRtAdaptive'] if not math.isnan(stats['medianRtAdaptive']) else float('nan'),
                )
            )
        return logPath


    def _logStaircasePauseSummary(self):
        '''Print experimenter-facing θ / RT summary (not shown to the subject).'''
        thetaPos = float(self.thresholdContrastPositive)
        thetaNeg = float(self.thresholdContrastNegative)
        levels = self._contrastsFromPolarityThresholds(thetaPos, thetaNeg)
        levelStr = ', '.join('{0:+g}'.format(c) for c in levels)
        stats = getattr(self, '_staircaseSummary', None) or self._staircaseSummaryStats(
            getattr(self, '_staircaseTrials', []) or []
        )
        print(
            '--> Staircase complete: +theta={p:.4f} ({sp}) -theta={n:.4f} ({sn}); '
            'OKR levels=[{levels}]; acc={acc} meanRT={rt}'.format(
                p=thetaPos,
                sp=self.thresholdStatusPositive or 'NA',
                n=thetaNeg,
                sn=self.thresholdStatusNegative or 'NA',
                levels=levelStr,
                acc=stats.get('accAdaptive', float('nan')),
                rt=stats.get('meanRtAdaptive', float('nan')),
            )
        )


    def _makeJarBundle(self, win, jarX, jarY, jarW, jarH, rimW, rimH, ppd_h, ppd_v,
                       glassColor, fillColor, fillLevel=1.0):
        '''Practice-style jar pieces centered at (jarX, jarY).'''
        jarGlass = visual.Rect(
            win, width=jarW, height=jarH, pos=(jarX, jarY),
            units='pix', lineColor=None, fillColor=[0.15, 0.08, 0.02], opacity=0.55,
        )
        jarOutline = visual.Rect(
            win, width=jarW, height=jarH, pos=(jarX, jarY),
            units='pix', lineColor=glassColor, fillColor=None, lineWidth=8,
        )
        jarRim = visual.Rect(
            win, width=rimW, height=rimH,
            pos=(jarX, jarY + jarH / 2.0 + rimH / 2.0 - 0.1 * ppd_v),
            units='pix', lineColor=glassColor, fillColor=[-0.2, -0.25, -0.3],
            lineWidth=6,
        )
        jarFill = visual.Rect(
            win, width=jarW - 0.45 * ppd_h, height=0.01,
            pos=(jarX, jarY - jarH / 2.0),
            units='pix', lineColor=None, fillColor=fillColor, opacity=0.8,
        )
        return {
            'x': jarX, 'y': jarY, 'w': jarW, 'h': jarH,
            'glass': jarGlass, 'outline': jarOutline, 'rim': jarRim, 'fill': jarFill,
            'fillLevel': float(fillLevel),
            'glassColor': glassColor, 'fillColor': fillColor,
        }


    def _placeJarBundle(self, jar, ox=0.0, oy=0.0, ppd_v=1.0, fillLevel=None):
        x = jar['x'] + ox
        y = jar['y'] + oy
        jar['glass'].pos = (x, y)
        jar['outline'].pos = (x, y)
        jar['rim'].pos = (x, y + jar['h'] / 2.0 + jar['rim'].height / 2.0 - 0.1 * ppd_v)
        level = jar['fillLevel'] if fillLevel is None else float(fillLevel)
        level = max(0.0, min(1.0, level))
        fillH = max(0.01, (jar['h'] - 0.35 * ppd_v) * level)
        jar['fill'].height = fillH
        jar['fill'].pos = (x, y - jar['h'] / 2.0 + fillH / 2.0 + 0.1 * ppd_v)


    def _drawJarBundle(self, jar, ox=0.0, oy=0.0, ppd_v=1.0, fillLevel=None):
        self._placeJarBundle(jar, ox=ox, oy=oy, ppd_v=ppd_v, fillLevel=fillLevel)
        jar['glass'].draw()
        jar['fill'].draw()
        jar['outline'].draw()
        jar['rim'].draw()


    def _showStaircaseCastleCelebration(self, win):
        '''
        Stacked Practice-style jars: base cup, second cup stacks on top,
        top fills, Practice-style outward confetti burst — then wait for Space/Enter.
        Returns True to continue, False if quit.
        '''
        self.getFR(win)
        fr = float(getattr(self, '_FR', 60) or 60)
        ppd_h, ppd_v = self.getPixPerDegXY(win.monitor, win=win)
        # Slightly smaller than Practice so two jars fit the viewport.
        jarW, jarH = 4.6 * ppd_h, 5.8 * ppd_v
        rimW, rimH = 5.5 * ppd_h, 0.85 * ppd_v
        stackGap = 0.05 * ppd_v
        baseY = -3.2 * ppd_v
        topY = baseY + jarH + stackGap
        jarX = 0.0
        glassColor = [0.92, 0.86, 0.78]
        fillColor = [1.0, 0.12, 0.02]
        candyColors = [
            [1.0, -0.15, -0.15],
            [1.0, 0.55, -0.55],
            [-0.15, 0.65, 1.0],
            [0.95, 0.9, -0.65],
            [0.65, -0.35, 0.95],
        ]
        win.color = [-0.88, -0.62, -0.42]

        baseJar = self._makeJarBundle(
            win, jarX, baseY, jarW, jarH, rimW, rimH, ppd_h, ppd_v,
            glassColor, fillColor, fillLevel=1.0,
        )
        topJar = self._makeJarBundle(
            win, jarX, topY, jarW, jarH, rimW, rimH, ppd_h, ppd_v,
            glassColor, fillColor, fillLevel=0.0,
        )

        titleHeightFinal = 0.85 * ppd_v
        titleText = visual.TextStim(
            win, text='Level Complete!',
            pos=(0.0, baseY - jarH / 2.0 - 1.35 * ppd_v),
            units='pix', height=titleHeightFinal,
            color=[0.98, 0.92, 0.72], bold=True, opacity=0.0,
        )

        # Practice-style outward burst from the top of the stacked castle.
        rng = np.random.default_rng(42)
        confetti = []
        for i in range(36):
            confetti.append({
                'stim': visual.Circle(
                    win, radius=0.14 * min(ppd_h, ppd_v),
                    pos=(0.0, 0.0), units='pix',
                    fillColor=candyColors[i % len(candyColors)], lineColor=None,
                ),
                'vx': float(rng.uniform(-10, 10)) * ppd_h,
                'vy': float(rng.uniform(5, 14)) * ppd_v,
                'x': jarX + float(rng.uniform(-0.8, 0.8)) * ppd_h,
                'y': topY + jarH / 2.0,
            })

        yayTone = self._loadFileSound(_STAIRCASE_YAY_SOUND, volume=0.55, name='staircaseYay')
        if yayTone is None:
            yayTone = self._makeToneSound(
                [523.25, 659.25, 783.99],
                noteDur=0.12, gapDur=0.03, volume=0.50, name='staircaseYayFallback',
            )
        # Keep a ref so PTB does not GC the clip mid-celebration.
        self._castleSoundRefs = [s for s in (yayTone,) if s is not None]

        def checkQuitOrSkip():
            keys = [k.lower() for k in event.getKeys()]
            if not keys:
                return None
            if 'q' in keys:
                self._stoppedEarly = 1
                return 'quit'
            if 'space' in keys or 'return' in keys or 'enter' in keys:
                return 'continue'
            return None

        def drawScene(
            showTop=True, topOy=0.0, topFill=0.0,
            showConfetti=False, confettiT=0.0, shakeX=0.0,
            titleScale=0.0,
        ):
            self._drawJarBundle(baseJar, ox=shakeX, ppd_v=ppd_v, fillLevel=1.0)
            if showTop:
                self._drawJarBundle(
                    topJar, ox=shakeX, oy=topOy, ppd_v=ppd_v, fillLevel=topFill,
                )
            if showConfetti:
                for c in confetti:
                    c['stim'].pos = (
                        c['x'] + c['vx'] * confettiT + shakeX,
                        c['y'] + c['vy'] * confettiT - 6.0 * ppd_v * confettiT * confettiT,
                    )
                    c['stim'].draw()
            if titleScale > 0.01:
                # Ease from tiny/invisible to full size below the castle.
                s = max(0.01, min(1.15, float(titleScale)))
                titleText.height = titleHeightFinal * s
                titleText.opacity = min(1.0, s / 0.85)
                titleText.draw()

        event.clearEvents()
        self._playSound(yayTone)

        # 1) Base cup appears full.
        appearFrames = max(8, int(round(0.35 * fr)))
        for f in range(appearFrames):
            drawScene(showTop=False, titleScale=0.0)
            win.flip()
            action = checkQuitOrSkip()
            if action == 'quit':
                return False
            if action == 'continue':
                return True

        # 2) Second cup drops in and stacks.
        dropFrames = max(10, int(round(0.55 * fr)))
        dropStartOy = 4.5 * ppd_v
        for f in range(dropFrames):
            u = (f + 1) / float(dropFrames)
            eased = 1.0 - (1.0 - u) ** 2
            bounce = 0.0
            if u > 0.82:
                bounce = 0.22 * ppd_v * math.sin((u - 0.82) / 0.18 * math.pi)
            topOy = dropStartOy * (1.0 - eased) - bounce
            drawScene(showTop=True, topOy=topOy, topFill=0.0, titleScale=0.0)
            win.flip()
            action = checkQuitOrSkip()
            if action == 'quit':
                return False
            if action == 'continue':
                return True

        # Brief settle / stack bounce shake.
        settleFrames = max(6, int(round(0.25 * fr)))
        for f in range(settleFrames):
            t = (f + 1) / float(settleFrames)
            shake = (0.12 * ppd_h) * math.sin(t * 8.0 * math.pi) * (1.0 - t)
            drawScene(showTop=True, topFill=0.0, shakeX=shake, titleScale=0.0)
            win.flip()
            action = checkQuitOrSkip()
            if action == 'quit':
                return False
            if action == 'continue':
                return True

        # 3) "Level Complete!" pops in below the castle after the stack lands.
        titleFrames = max(10, int(round(0.45 * fr)))
        for f in range(titleFrames):
            u = (f + 1) / float(titleFrames)
            # Ease-out with a slight overshoot, then settle to 1.0.
            eased = 1.0 - (1.0 - u) ** 3
            if u < 0.85:
                scale = 1.12 * eased / 0.85
            else:
                settle = (u - 0.85) / 0.15
                scale = 1.12 + (1.0 - 1.12) * settle
            drawScene(showTop=True, topFill=0.0, titleScale=scale)
            win.flip()
            action = checkQuitOrSkip()
            if action == 'quit':
                return False
            if action == 'continue':
                return True

        # 4) Top cup fills (red candy fill).
        fillFrames = max(12, int(round(0.55 * fr)))
        for f in range(fillFrames):
            t = (f + 1) / float(fillFrames)
            drawScene(showTop=True, topFill=t, titleScale=1.0)
            win.flip()
            action = checkQuitOrSkip()
            if action == 'quit':
                return False
            if action == 'continue':
                return True

        # 5) Confetti explodes outward (same motion model as Practice).
        flourishFrames = max(12, int(round(0.9 * fr)))
        holdConfettiT = 0.85
        for f in range(flourishFrames):
            t = (f + 1) / float(flourishFrames)
            shake = (0.18 * ppd_h) * math.sin(t * 10.0 * math.pi) * (1.0 - t)
            drawScene(
                showTop=True, topFill=1.0,
                showConfetti=True, confettiT=t * 0.9, shakeX=shake,
                titleScale=1.0,
            )
            win.flip()
            action = checkQuitOrSkip()
            if action == 'quit':
                return False
            if action == 'continue':
                return True

        # Hold / wait for continue.
        event.clearEvents()
        while True:
            drawScene(
                showTop=True, topFill=1.0,
                showConfetti=True, confettiT=holdConfettiT,
                titleScale=1.0,
            )
            win.flip()
            action = checkQuitOrSkip()
            if action == 'quit':
                return False
            if action == 'continue':
                return True


    def _pauseBeforeOkr(self, win):
        '''Castle checkpoint (subject) + console θ summary. Returns False if quit.'''
        self._logStaircasePauseSummary()
        if getattr(self, 'staircaseCastleCelebration', True):
            try:
                return self._showStaircaseCastleCelebration(win)
            except Exception as err:
                print('*** Staircase castle celebration failed:', err)
        # Fallback: plain continue prompt (no performance numbers on screen).
        win.color = [-0.88, -0.62, -0.42]
        label = visual.TextStim(
            win,
            text=(
                'Staircase complete!\n\n'
                'Press Space or Enter to continue to eye movements\n'
                'Press q to quit'
            ),
            color=[0.95, 0.88, 0.75],
            height=36,
            wrapWidth=win.size[0] * 0.8,
        )
        event.clearEvents()
        while True:
            label.draw()
            win.flip()
            keys = [k.lower() for k in event.getKeys()]
            if not keys:
                continue
            if 'q' in keys:
                self._stoppedEarly = 1
                return False
            if 'space' in keys or 'return' in keys or 'enter' in keys:
                return True


    def run(self, win, informationWin):
        self._completed = 0
        self.thresholdContrast = None
        self.thresholdContrastPositive = None
        self.thresholdContrastNegative = None
        self.thresholdStatusPositive = None
        self.thresholdStatusNegative = None
        self.staircaseCompliance = None
        self._staircaseTrials = []

        # Attention probes are OKR-only: keep user's setting for phase 2, force off in 2AFC.
        probesForOkr = bool(getattr(self, 'attentionProbe', False))
        self.attentionProbe = False

        print('--> Contrast Dots Staircase: starting 2AFC threshold phase')
        try:
            if not self._runStaircasePhase(win, informationWin):
                print('*** Staircase aborted')
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
            # Enter pause already gated OKR; skip parent's second "press any key".
            self._initOkrFixationCountdownSounds()
            wasUserInitiated = self.userInitiated
            self.userInitiated = False
            try:
                super().run(win, informationWin)
            finally:
                self.userInitiated = wasUserInitiated
                self._okrFixationCountdownSounds = {}
        finally:
            self.attentionProbe = probesForOkr
