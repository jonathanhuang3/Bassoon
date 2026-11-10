# -*- coding: utf-8 -*-
"""
Contrast Dots presents a black fixation cross, then coherent moving dots on a gray
background, then an optional gray blank for afternystagmus, then another black
fixation cross (also used between trials).

Optional attention probe: a set number of brief full-red dots appear at random
times in a disk around current gaze (each lasting one dotLifetime). Probe size is
settable (independent of field dots) and uses the last ElementArrayStim slot so it
draws on top of neighboring dots. The subject presses space when they notice each
one; a short sound plays on each press.
"""
from protocols.protocol import protocol
from psychopy import visual, event, core
from datetime import datetime
from pathlib import Path
import random, math
import time
import numpy as np

_ATTENTION_PROBE_REWARD_SOUND = (
    Path(__file__).resolve().parent / 'sounds' / 'attention_probe_reward.mp3'
)


class ContrastDots(protocol):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDots'
        self.stimulusReps = 1 #number of repetitions through all contrast levels
        self.interStimulusInterval = 0.0 #seconds - wait time between epochs
        self.preTime = 0.0 #seconds - stationary period before dot motion
        self.stimTime = 20.0 #seconds - moving dots are shown for this duration
        self.postStimTime = 0.0 #seconds - gray background after dots end, before the fixation cross (for OKR afternystagmus)
        self.tailTime = 4.0 #seconds - black fixation cross before the first trial and after each post-stim gray blank
        self.backgroundColor = [0.0, 0.0, 0.0] #gray background (in RGB, -1 to 1)

        # Dot parameters
        self.numberOfDots = 200 #number of dots displayed at once
        self.dotColor = [1.0, 1.0, 1.0] #white dots at full contrast (in RGB, -1 to 1)
        self.contrasts = [1.0] #list of contrast levels from -1 to 1. Total epochs = len(contrasts) * len(directions) * stimulusReps
        self.dotSizeDegrees = 1.0 #degrees - dot diameter
        self.dotLifetime = 0.15 #seconds - how long each dot stays visible before respawning; also the attention-probe flash duration
        self.spawnStagger = 0.15 #seconds - max random delay before each dot's first lifetime expiry (spreads respawns across time)
        self.direction = 90.0 #degrees - legacy single direction; ignored when directions has more than one entry
        self.directions = [0.0, 90.0, 180.0, 270.0] #degrees - right, up, left, down
        self.maxConsecutiveSameDirection = 3 #maximum blocks in a row with the same direction before forcing a switch
        self.speed = 10.0 #degrees per second

        # Fixation cross shown during tail time
        self.fixationCrossColor = [-1.0, -1.0, -1.0] #black (in RGB, -1 to 1)
        self.fixationCrossSizeDegrees = 1.0 #degrees - tip-to-tip span (same visual extent scale as dotSizeDegrees)

        # Attention probe: N discrete full-red flashes near gaze during each motion block.
        self.attentionProbe = True
        self.attentionProbeCount = 3 #how many red probes appear during each stimTime block
        self.attentionProbeDiameterDegrees = 0.0 #disk diameter around gaze (0 = place exactly at gaze)
        self.attentionProbeSizeDegrees = 2.0 #probe dot diameter (independent of field dotSizeDegrees)
        self.attentionProbeColor = [1.0, -1.0, -1.0] #full red (independent of grayscale contrast block)
        self.attentionProbeKey = 'space'
        self.attentionProbeSound = True #play a short ding on each spacebar press during motion
        # How long after probe onset a spacebar still counts as a hit (visual flash may be shorter).
        self.attentionProbeResponseWindowSec = 1.0


    def _usePersistentDots(self):
        '''When True, dots are placed once per epoch and never respawn.'''
        return False


    def internalValidation(self):
        tf = True
        errorMessage = []

        if self.dotSizeDegrees <= 0:
            tf = False
            errorMessage.append('Dot Size must be greater than 0 degrees.')
        if not self._usePersistentDots() and self.dotLifetime <= 0:
            tf = False
            errorMessage.append('Dot Lifetime must be greater than 0 seconds.')
        if self.speed < 0:
            tf = False
            errorMessage.append('Speed must be 0 or greater.')
        if self.spawnStagger < 0:
            tf = False
            errorMessage.append('Spawn Stagger must be 0 or greater.')
        if getattr(self, 'postStimTime', 0.0) < 0:
            tf = False
            errorMessage.append('Post Stim Time must be 0 or greater.')
        if len(self.contrasts) == 0:
            tf = False
            errorMessage.append('Contrasts must contain at least one value.')
        for contrast in self.contrasts:
            if contrast < -1 or contrast > 1:
                tf = False
                errorMessage.append('Contrast values must be between -1 and 1.')
                break
        if self.fixationCrossSizeDegrees <= 0:
            tf = False
            errorMessage.append('Fixation Cross Size must be greater than 0 degrees.')
        directionPool = self._directionPool()
        if len(directionPool) == 0:
            tf = False
            errorMessage.append('Directions must contain at least one value.')
        for direction in directionPool:
            if direction < 0 or direction >= 360:
                tf = False
                errorMessage.append('Direction values must be between 0 and 360 degrees.')
                break
        if self.maxConsecutiveSameDirection < 1:
            tf = False
            errorMessage.append('Max Consecutive Same Direction must be at least 1.')
        if getattr(self, 'attentionProbe', False):
            if float(self.attentionProbeDiameterDegrees) < 0:
                tf = False
                errorMessage.append('Attention Probe Diameter must be 0 or greater degrees (0 = exact gaze).')
            if float(getattr(self, 'attentionProbeSizeDegrees', self.dotSizeDegrees)) <= 0:
                tf = False
                errorMessage.append('Attention Probe Size must be greater than 0 degrees.')
            if int(self.numberOfDots) < 1:
                tf = False
                errorMessage.append('Number of Dots must be at least 1 when attention probes are enabled.')
            keyName = str(getattr(self, 'attentionProbeKey', 'space')).strip()
            if not keyName:
                tf = False
                errorMessage.append('Attention Probe Key must be a non-empty key name (e.g. space).')
            if int(self.attentionProbeCount) > 0:
                duration = float(self.dotLifetime)
                responseWindow = float(getattr(self, 'attentionProbeResponseWindowSec', 1.0))
                spacing = max(duration, responseWindow)
                if duration <= 0:
                    tf = False
                    errorMessage.append(
                        'Dot Lifetime must be greater than 0 seconds when attention probes are used '
                        '(probe flash duration equals dot lifetime).'
                    )
                elif responseWindow <= 0:
                    tf = False
                    errorMessage.append(
                        'Attention Probe Response Window must be greater than 0 seconds.'
                    )
                elif float(self.stimTime) < int(self.attentionProbeCount) * spacing:
                    tf = False
                    errorMessage.append(
                        'Stim Time must be at least Attention Probe Count × '
                        'max(Dot Lifetime, Response Window) so probe response windows do not overlap.'
                    )

        tfColors, colorErrorMessages = self.validateColorInput()
        tf = tf and tfColors
        errorMessage += colorErrorMessages
        return tf, errorMessage


    def estimateTime(self):
        postStimTime = getattr(self, 'postStimTime', 0.0)
        timePerEpoch = (
            self.preTime
            + self.stimTime
            + postStimTime
            + self.tailTime
            + self.interStimulusInterval
        )
        numberOfEpochs = self.stimulusReps * len(self.contrasts) * len(self._directionPool())
        # One lead-in fixation (tailTime) before the first dots trial.
        self._estimatedTime = timePerEpoch * numberOfEpochs + float(self.tailTime)
        return self._estimatedTime


    def dotColorAtContrast(self, contrast):
        '''Linear contrast around background: 0 = background, +1 = dotColor, -1 = mirrored decrement.'''
        return [
            self.backgroundColor[i] + contrast * (self.dotColor[i] - self.backgroundColor[i])
            for i in range(3)
        ]


    def _winSizeKey(self, win):
        return (int(win.size[0]), int(win.size[1]))


    def _stimulusScaleFromWin(self, win):
        '''
        Deg→pix scale for the current drawable size.

        Uses the monitor's physical FOV with win.size so resizing a windowed
        stimulus keeps degree-specified size/speed matched to the window.
        '''
        ppd_h, ppd_v = self.getPixPerDegXY(win.monitor, win=win)
        dotDiameterPix = [
            float(self.dotSizeDegrees) * ppd_h,
            float(self.dotSizeDegrees) * ppd_v,
        ]
        probeDiameterPix = [
            self._attentionProbeSizeDegrees() * ppd_h,
            self._attentionProbeSizeDegrees() * ppd_v,
        ]
        dotRadiusPix = 0.5 * max(dotDiameterPix)
        self._lastWinSizeKey = self._winSizeKey(win)
        return ppd_h, ppd_v, dotDiameterPix, probeDiameterPix, dotRadiusPix


    def _syncFixationCrossGeometry(self, fixationCrossArms, ppd_h, ppd_v):
        '''Update black-cross arm length/width after a window resize.'''
        if not fixationCrossArms:
            return
        crossHalfX = 0.5 * float(self.fixationCrossSizeDegrees) * ppd_h
        crossHalfY = 0.5 * float(self.fixationCrossSizeDegrees) * ppd_v
        crossLineWidth = max(2.0, min(crossHalfX, crossHalfY) * 2.0 * 0.15)
        try:
            fixationCrossArms[0].vertices = ((-crossHalfX, 0.0), (crossHalfX, 0.0))
            fixationCrossArms[0].lineWidth = crossLineWidth
            fixationCrossArms[1].vertices = ((0.0, -crossHalfY), (0.0, crossHalfY))
            fixationCrossArms[1].lineWidth = crossLineWidth
        except Exception:
            pass


    def _motionContrast(self, epoch, frameIndex):
        '''
        Contrast applied on motion frame frameIndex within the current epoch.
        Subclasses (e.g. threshold ramp) may vary this over time.
        '''
        return float(epoch['contrast'])


    def _epochContrastDisplay(self, epoch):
        '''Short contrast label for the information window.'''
        return epoch['contrast']


    def _probeColor(self):
        '''Full-strength attention-probe color (not tied to the grayscale contrast block).'''
        return list(getattr(self, 'attentionProbeColor', [1.0, -1.0, -1.0]))


    def _attentionProbeSizeDegrees(self):
        '''Probe diameter in degrees (falls back to field dot size if unset).'''
        return float(getattr(self, 'attentionProbeSizeDegrees', self.dotSizeDegrees))


    def _attentionProbeIndex(self):
        '''
        Last element index for the probe so ElementArrayStim draws it after
        (on top of) the other field dots.
        '''
        total = int(self.numberOfDots)
        return total - 1 if total > 0 else 0


    def _initAttentionProbeSound(self):
        '''Preload positive arcade-UI reward SFX (fallback: short synthetic ding).'''
        if not getattr(self, 'attentionProbe', False):
            self._attentionProbeSound = None
            return
        if not getattr(self, 'attentionProbeSound', True):
            self._attentionProbeSound = None
            return
        if getattr(self, '_attentionProbeSound', None) is not None:
            return
        try:
            from psychopy import sound
            rewardPath = _ATTENTION_PROBE_REWARD_SOUND
            if rewardPath.is_file():
                self._attentionProbeSound = sound.Sound(
                    value=str(rewardPath), name='attentionProbeReward',
                )
                try:
                    self._attentionProbeSound.setVolume(0.55)
                except Exception:
                    pass
                return
            print('*** Attention probe reward sound missing; using synthetic ding.')
            sampleRate = 44100
            duration = 0.18
            t = np.linspace(0.0, duration, int(sampleRate * duration), endpoint=False)
            wave = 0.35 * np.sin(2.0 * np.pi * 880.0 * t) * np.exp(-t / 0.06)
            wave += 0.12 * np.sin(2.0 * np.pi * 1320.0 * t) * np.exp(-t / 0.04)
            attackN = max(1, int(0.002 * sampleRate))
            wave[:attackN] *= np.linspace(0.0, 1.0, attackN)
            self._attentionProbeSound = sound.Sound(
                value=wave.astype(np.float32),
                sampleRate=sampleRate,
                stereo=True,
                name='attentionProbeHit',
            )
        except Exception as e:
            print('*** Attention probe sound unavailable (' + str(e) + '). Continuing silently.')


    def _playAttentionProbeSound(self):
        snd = getattr(self, '_attentionProbeSound', None)
        if snd is None:
            return
        try:
            snd.stop()
            snd.play()
        except Exception:
            pass


    def _getAttentionProbeOriginPix(self, win):
        '''
        Gaze position in PsychoPy pix for the probe disk center.
        Prefers the latest mask-protocol gaze, then a fresh EyeLink sample,
        then screen center if tracking is unavailable.
        '''
        last = getattr(self, '_lastGaze', None)
        if last is not None:
            return float(last[0]), float(last[1])

        readGaze = getattr(self, '_readGazePix', None)
        if callable(readGaze):
            try:
                gaze = readGaze(win)
                if gaze is not None:
                    return float(gaze[0]), float(gaze[1])
            except Exception:
                pass

        tracker = getattr(self, '_elTracker', None) or getattr(self, '_tracker', None)
        if tracker is None:
            return 0.0, 0.0
        try:
            sample = tracker.getNewestSample()
            if sample is None:
                return 0.0, 0.0
            gazes = []
            if sample.isLeftSample():
                left = sample.getLeftEye()
                g = left.getGaze()
                if g[0] != getattr(self, '_missingData', -32768) and g[1] != getattr(self, '_missingData', -32768):
                    gazes.append(g)
            if sample.isRightSample():
                right = sample.getRightEye()
                g = right.getGaze()
                if g[0] != getattr(self, '_missingData', -32768) and g[1] != getattr(self, '_missingData', -32768):
                    gazes.append(g)
            if not gazes:
                return 0.0, 0.0
            if len(gazes) == 2:
                gx = 0.5 * (float(gazes[0][0]) + float(gazes[1][0]))
                gy = 0.5 * (float(gazes[0][1]) + float(gazes[1][1]))
            else:
                gx = float(gazes[0][0])
                gy = float(gazes[0][1])
            halfW = win.size[0] / 2.0
            halfH = win.size[1] / 2.0
            return gx - halfW, halfH - gy
        except Exception:
            return 0.0, 0.0


    def _randomPointInProbeDisk(self, ppd_h, ppd_v, originPix=(0.0, 0.0), win=None):
        '''
        Return (xPix, yPix, offsetXDeg, offsetYDeg) uniformly inside the probe
        disk centered on originPix (gaze). Offsets are degrees relative to gaze.
        Diameter 0 places the probe exactly at gaze.
        '''
        radiusDeg = float(self.attentionProbeDiameterDegrees) / 2.0
        marginDeg = max(float(self.dotSizeDegrees), self._attentionProbeSizeDegrees())
        if radiusDeg <= 0:
            xPix = float(originPix[0])
            yPix = float(originPix[1])
            if win is not None:
                margin = max(ppd_h, ppd_v) * marginDeg
                xMax = win.size[0] / 2.0 - margin
                yMax = win.size[1] / 2.0 - margin
                xPix = float(np.clip(xPix, -xMax, xMax))
                yPix = float(np.clip(yPix, -yMax, yMax))
            return xPix, yPix, 0.0, 0.0
        theta = random.uniform(0.0, 2.0 * math.pi)
        rDeg = radiusDeg * math.sqrt(random.uniform(0.0, 1.0))
        xDeg = rDeg * math.cos(theta)
        yDeg = rDeg * math.sin(theta)
        xPix = float(originPix[0]) + xDeg * ppd_h
        yPix = float(originPix[1]) + yDeg * ppd_v
        if win is not None:
            margin = max(ppd_h, ppd_v) * marginDeg
            xMax = win.size[0] / 2.0 - margin
            yMax = win.size[1] / 2.0 - margin
            xPix = float(np.clip(xPix, -xMax, xMax))
            yPix = float(np.clip(yPix, -yMax, yMax))
        return xPix, yPix, xDeg, yDeg


    def _scheduleProbeOnsetFrames(self, nStimFrames, durationFrames, count):
        '''
        Random non-overlapping probe onset frames within a motion block.
        Each probe occupies [onset, onset + durationFrames).
        '''
        count = int(count)
        durationFrames = int(durationFrames)
        if (
            not getattr(self, 'attentionProbe', False)
            or count <= 0
            or durationFrames <= 0
            or nStimFrames < durationFrames
        ):
            return []
        maxCount = nStimFrames // durationFrames
        count = min(count, maxCount)
        usable = nStimFrames - durationFrames  # inclusive max onset frame
        if count == 1:
            return [random.randint(0, usable)]
        segment = float(usable + 1) / float(count)
        onsets = []
        for i in range(count):
            lo = int(math.floor(i * segment))
            hi = int(math.floor((i + 1) * segment) - 1)
            hi = min(hi, usable)
            lo = min(lo, hi)
            onsets.append(random.randint(lo, hi))
        onsets.sort()
        for i in range(1, len(onsets)):
            minStart = onsets[i - 1] + durationFrames
            if onsets[i] < minStart:
                onsets[i] = minStart
        return [o for o in onsets if o >= 0 and o + durationFrames <= nStimFrames]


    def _startAttentionProbe(
        self, probeNumber, probeIndex, ppd_h, ppd_v, trialClock, blockIndex, contrast, frameIndex,
        win=None,
    ):
        # Close any still-open response window from the previous probe.
        self._finalizeAttentionProbeResponse(trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex)
        origin = self._getAttentionProbeOriginPix(win) if win is not None else (0.0, 0.0)
        xPix, yPix, xDeg, yDeg = self._randomPointInProbeDisk(
            ppd_h, ppd_v, originPix=origin, win=win,
        )
        self.dotCoords[probeIndex] = [xPix, yPix]
        if getattr(self, 'currentFrames', None) is not None:
            self.currentFrames[probeIndex] = 0
        spawnTime = trialClock.getTime()
        responseWindow = max(0.05, float(getattr(self, 'attentionProbeResponseWindowSec', 1.0)))
        self._probeActive = True  # red flash visible
        self._probeResponseOpen = True  # spacebar still counts as a hit
        self._probeNumber = probeNumber
        self._probeSpawnTime = spawnTime
        self._probeResponseUntil = spawnTime + responseWindow
        self._probeSpawnFrame = frameIndex
        self._probePressCount = 0
        self._probeLastXPix = xPix
        self._probeLastYPix = yPix
        self._attentionEvents.append({
            'eventType': 'ProbeSpawn',
            'contrastBlockIndex': blockIndex,
            'contrastLevel': contrast,
            'probeNumber': probeNumber,
            'time': spawnTime,
            'reactionTime': 'NA',
            'probeXDeg': xDeg,
            'probeYDeg': yDeg,
            'gazeXDeg': origin[0] / ppd_h,
            'gazeYDeg': origin[1] / ppd_v,
            'pressCount': 'NA',
            'responseIndex': 'NA',
        })
        if getattr(self, '_openOkrProbeEvent', None) is not None:
            self._closeOkrAttentionProbe(spawnTime)
        self._appendOkrAttentionProbe(
            getattr(self, '_okrEvents', None),
            getattr(self, '_okrEventCounter', None),
            blockIndex, contrast, probeNumber, spawnTime,
        )
        self._sendOkrEyeLinkMessage(
            'OKR AttentionProbeSpawn B{bi} P{pn} contrast {c:g} t={t:.3f}'.format(
                bi=blockIndex, pn=probeNumber, c=contrast, t=spawnTime,
            )
        )


    def _endAttentionProbe(self, trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex):
        '''End the red flash only; response window may stay open a bit longer.'''
        if not getattr(self, '_probeActive', False):
            return
        self._closeOkrAttentionProbe(trialClock.getTime())
        if probeIndex is not None:
            xPix, yPix = self.dotCoords[probeIndex]
            self._probeLastXPix = xPix
            self._probeLastYPix = yPix
        self._probeActive = False
        self._probeSpawnFrame = None
        # If the response window already elapsed, finalize now; otherwise wait for
        # spacebar / deadline / next probe.
        self._maybeFinalizeAttentionProbeResponse(
            trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex,
        )


    def _maybeFinalizeAttentionProbeResponse(
        self, trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex,
    ):
        if not getattr(self, '_probeResponseOpen', False):
            return
        deadline = getattr(self, '_probeResponseUntil', None)
        if deadline is not None and trialClock.getTime() < deadline:
            return
        self._finalizeAttentionProbeResponse(
            trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex,
        )


    def _finalizeAttentionProbeResponse(
        self, trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex,
    ):
        '''Log ProbeEnd / hit once the response window closes.'''
        if not getattr(self, '_probeResponseOpen', False):
            return
        endTime = trialClock.getTime()
        pressCount = int(getattr(self, '_probePressCount', 0))
        if probeIndex is not None and getattr(self, '_probeActive', False):
            xPix, yPix = self.dotCoords[probeIndex]
        else:
            xPix = float(getattr(self, '_probeLastXPix', 0.0))
            yPix = float(getattr(self, '_probeLastYPix', 0.0))
        self._attentionEvents.append({
            'eventType': 'ProbeEnd',
            'contrastBlockIndex': blockIndex,
            'contrastLevel': contrast,
            'probeNumber': getattr(self, '_probeNumber', 'NA'),
            'time': endTime,
            'reactionTime': 'NA',
            'probeXDeg': xPix / ppd_h,
            'probeYDeg': yPix / ppd_v,
            'pressCount': pressCount,
            'responseIndex': 'NA',
            'hit': int(pressCount > 0),
        })
        self._sendOkrEyeLinkMessage(
            'OKR AttentionProbeEnd B{bi} P{pn} presses={n} hit={h} t={t:.3f}'.format(
                bi=blockIndex,
                pn=getattr(self, '_probeNumber', 'NA'),
                n=pressCount,
                h=int(pressCount > 0),
                t=endTime,
            )
        )
        self._probeResponseOpen = False
        self._probeActive = False
        self._probeSpawnTime = None
        self._probeSpawnFrame = None
        self._probeResponseUntil = None


    def _applyDotAppearance(self, dots, contrast, probeIndex, fieldDiameterPix, probeDiameterPix):
        '''Field contrast colors; active probe is full red and uses probeDiameterPix.'''
        base = self.dotColorAtContrast(contrast)
        probeActive = (
            getattr(self, 'attentionProbe', False)
            and probeIndex is not None
            and getattr(self, '_probeActive', False)
        )
        if not probeActive:
            dots.colors = base
            dots.sizes = fieldDiameterPix
            return
        colors = np.tile(np.asarray(base, dtype=float), (self.numberOfDots, 1))
        colors[probeIndex] = self._probeColor()
        dots.colors = colors
        sizes = np.tile(np.asarray(fieldDiameterPix, dtype=float), (self.numberOfDots, 1))
        sizes[probeIndex] = np.asarray(probeDiameterPix, dtype=float)
        dots.sizes = sizes


    def _handleMotionKeys(self, trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex):
        '''
        Single key poll during motion: quit / pause / attention-probe response.
        Returns True if the stimulus should abort.
        '''
        keys = event.getKeys()
        if not keys:
            return False
        if 'q' in keys:
            self._stoppedEarly = 1
            print('*** Quiting stimulus early')
            return True
        if 'p' in keys:
            self._userPauseCount += 1
            print('*** STIMULUS HAS PAUSED. Press any key to resume')
            startTime = time.time()
            event.waitKeys()
            endTime = time.time()
            pauseTime = endTime - startTime
            print('*** Resuming Stimulus. Total pause time was %s seconds' % pauseTime)
            self._userPauseDurations.append(pauseTime)
            return False
        probeKey = str(getattr(self, 'attentionProbeKey', 'space')).strip().lower()
        if not (getattr(self, 'attentionProbe', False) and probeKey in [k.lower() for k in keys]):
            return False

        self._playAttentionProbeSound()
        responseTime = trialClock.getTime()
        # Accept presses while the response window is open (may outlast the red flash).
        self._maybeFinalizeAttentionProbeResponse(
            trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex,
        )
        responseOpen = (
            getattr(self, '_probeResponseOpen', False)
            and responseTime <= float(getattr(self, '_probeResponseUntil', responseTime))
        )
        if responseOpen:
            self._probePressCount = int(getattr(self, '_probePressCount', 0)) + 1
            spawnTime = getattr(self, '_probeSpawnTime', None)
            rt = (responseTime - spawnTime) if spawnTime is not None else None
            if probeIndex is not None and getattr(self, '_probeActive', False):
                xPix, yPix = self.dotCoords[probeIndex]
            else:
                xPix = float(getattr(self, '_probeLastXPix', 0.0))
                yPix = float(getattr(self, '_probeLastYPix', 0.0))
            self._attentionEvents.append({
                'eventType': 'ProbeResponse',
                'contrastBlockIndex': blockIndex,
                'contrastLevel': contrast,
                'probeNumber': getattr(self, '_probeNumber', 'NA'),
                'time': responseTime,
                'reactionTime': rt if rt is not None else 'NA',
                'probeXDeg': xPix / ppd_h,
                'probeYDeg': yPix / ppd_v,
                'pressCount': self._probePressCount,
                'responseIndex': self._probePressCount,
            })
            self._sendOkrEyeLinkMessage(
                'OKR AttentionProbeResponse B{bi} P{pn} n={n} t={t:.3f} rt={rt}'.format(
                    bi=blockIndex,
                    pn=getattr(self, '_probeNumber', 'NA'),
                    n=self._probePressCount,
                    t=responseTime,
                    rt='NA' if rt is None else '{:.3f}'.format(rt),
                )
            )
        else:
            self._attentionEvents.append({
                'eventType': 'ProbeFalseAlarm',
                'contrastBlockIndex': blockIndex,
                'contrastLevel': contrast,
                'probeNumber': 'NA',
                'time': responseTime,
                'reactionTime': 'NA',
                'probeXDeg': 'NA',
                'probeYDeg': 'NA',
                'pressCount': 'NA',
                'responseIndex': 'NA',
            })
            self._sendOkrEyeLinkMessage(
                'OKR AttentionProbeFalseAlarm B{bi} t={t:.3f}'.format(
                    bi=blockIndex, t=responseTime,
                )
            )
        return False


    def _writeAttentionProbeLog(self, events):
        if not events:
            return None
        logDir = getattr(self, '_okrLogDir', None)
        logDir = Path.cwd() if logDir is None else Path(logDir)
        logDir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logPath = logDir / ('{name}_AttentionProbe_{stamp}.txt'.format(
            name=self.protocolName, stamp=stamp,
        ))
        header = [
            '# Attention Probe Log',
            '# StimulusName: Bassoon {name}'.format(name=self.protocolName),
            '# ProbeCountPerBlock: {n}'.format(n=int(self.attentionProbeCount)),
            '# ProbeDurationSec: {d:g} (equals Dot Lifetime)'.format(d=float(self.dotLifetime)),
            '# ProbeDiameterDeg: {d:g} (disk around gaze; 0 = exact gaze)'.format(
                d=float(self.attentionProbeDiameterDegrees),
            ),
            '# ProbeSizeDeg: {d:g}'.format(d=self._attentionProbeSizeDegrees()),
            '# ProbeKey: {k}'.format(k=getattr(self, 'attentionProbeKey', 'space')),
            '# TimeBase: seconds from protocol SYNCTIME',
            '# probeXDeg/probeYDeg are offsets from gaze at spawn; gazeXDeg/gazeYDeg are gaze at spawn',
            'eventIndex\teventType\tcontrastBlockIndex\tcontrastLevel\tprobeNumber\ttime\t'
            'reactionTime\tprobeXDeg\tprobeYDeg\tgazeXDeg\tgazeYDeg\tpressCount\tresponseIndex\thit',
        ]
        rows = []
        for i, eventRow in enumerate(events, start=1):
            def fmt(value, digits=6):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return '{:.{n}f}'.format(value, n=digits)
                return 'NA' if value is None else str(value)

            rows.append('\t'.join([
                str(i),
                str(eventRow.get('eventType', 'NA')),
                str(eventRow.get('contrastBlockIndex', 'NA')),
                str(eventRow.get('contrastLevel', 'NA')),
                str(eventRow.get('probeNumber', 'NA')),
                fmt(eventRow.get('time')),
                fmt(eventRow.get('reactionTime')),
                fmt(eventRow.get('probeXDeg'), 4),
                fmt(eventRow.get('probeYDeg'), 4),
                fmt(eventRow.get('gazeXDeg'), 4),
                fmt(eventRow.get('gazeYDeg'), 4),
                str(eventRow.get('pressCount', 'NA')),
                str(eventRow.get('responseIndex', 'NA')),
                str(eventRow.get('hit', 'NA')),
            ]))
        logPath.write_text('\n'.join(header + rows) + '\n', encoding='utf-8')
        return logPath


    def createContrastLog(self):
        '''Build a randomized sequence of contrast levels, one per epoch.'''
        self.createEpochLog()


    def _directionPool(self):
        '''Return normalized direction angles used for this protocol.'''
        pool = getattr(self, 'directions', None)
        if not pool:
            return [self.deg0to360(self.direction)]
        return [self.deg0to360(d) for d in pool]


    def _directionRunIsValid(self, epoch_log):
        max_run = int(self.maxConsecutiveSameDirection)
        if max_run < 1 or len(epoch_log) <= max_run:
            return True
        run_count = 1
        for index in range(1, len(epoch_log)):
            if epoch_log[index]['direction'] == epoch_log[index - 1]['direction']:
                run_count += 1
                if run_count > max_run:
                    return False
            else:
                run_count = 1
        return True


    def _buildFactorialEpochPairs(self):
        '''Every contrast paired with every direction, repeated per stimulusRep.'''
        direction_pool = self._directionPool()
        pairs = []
        for _ in range(self.stimulusReps):
            for contrast in self.contrasts:
                for direction in direction_pool:
                    pairs.append({'contrast': contrast, 'direction': direction})
        return pairs


    def _shuffleEpochLog(self, pairs):
        '''Randomize block order; retry if direction consecutive limit is exceeded.'''
        if not pairs:
            return pairs
        shuffled = list(pairs)
        if len(self._directionPool()) == 1:
            random.shuffle(shuffled)
            return shuffled
        for _ in range(1000):
            random.shuffle(shuffled)
            if self._directionRunIsValid(shuffled):
                return shuffled
        return shuffled


    def createEpochLog(self):
        '''Build every contrast x direction pair, then shuffle block order.'''
        random.seed(self.randomSeed)
        pairs = self._buildFactorialEpochPairs()
        self._epochLog = self._shuffleEpochLog(pairs)
        self._contrastLog = [epoch['contrast'] for epoch in self._epochLog]


    def initDotPositions(self, win, dotRadiusPix):
        for dot in range(self.numberOfDots):
            xPos, yPos = self.respawnDot(win, dotRadiusPix)
            self.dotCoords[dot][0] = xPos
            self.dotCoords[dot][1] = yPos


    def deg0to360(self, angle):
        factor = abs(int(angle / 360.0))
        if angle < 0:
            angle += 360.0 * (factor + 1)
        if angle >= 360.0:
            angle -= 360.0 * factor
        return angle


    def initDotSpawnStagger(self):
        '''Start each dot's lifetime timer at a random negative frame count.'''
        self.currentFrames = -np.array([
            random.uniform(0, self.spawnStagger) * self._FR
            for _ in range(self.numberOfDots)
        ])


    def _directionLabel(self, direction=None):
        '''Map motion direction (degrees) to slowphase-okr direction names.

        Angles match the motion convention used in run(): 0° = right (+x),
        90° = up (+y), 180° = left (-x), 270° = down (-y).
        '''
        angle = self.deg0to360(self.direction if direction is None else direction)
        if 45.0 <= angle < 135.0:
            return 'Up'
        if 135.0 <= angle < 225.0:
            return 'Left'
        if 225.0 <= angle < 315.0:
            return 'Down'
        return 'Right'


    def _dotColorLabel(self):
        if self.dotColor[0] > 0.5 and self.dotColor[1] > 0.5 and self.dotColor[2] > 0.5:
            return 'White'
        if self.dotColor[0] < -0.5 and self.dotColor[1] < -0.5 and self.dotColor[2] < -0.5:
            return 'Black'
        return 'NA'


    def _nextOkrEventIndex(self, counter):
        counter[0] += 1
        return counter[0]


    def _sendOkrEyeLinkMessage(self, text):
        sendMessage = getattr(self, '_sendEyeLinkMessage', None)
        if sendMessage is not None:
            sendMessage(text)


    def _okrEpoch(self, blockIndex):
        try:
            idx = int(blockIndex)
        except (TypeError, ValueError):
            return None
        epochLog = getattr(self, '_epochLog', None)
        if epochLog is None or idx < 0 or idx >= len(epochLog):
            return None
        return epochLog[idx]


    def _okrDirectionDeg(self, blockIndex, direction=None):
        if direction is None:
            epoch = self._okrEpoch(blockIndex)
            if epoch is not None:
                direction = epoch.get('direction')
        try:
            return '{g:g}'.format(g=float(direction))
        except (TypeError, ValueError):
            return 'NA'


    def _okrConditionText(self, blockIndex, direction=None):
        '''One line naming the variables this trial actually ran.'''
        try:
            if int(blockIndex) < 0:
                return 'lead-in fixation'
        except (TypeError, ValueError):
            return 'NA'
        epoch = self._okrEpoch(blockIndex)
        if epoch is None:
            return 'NA'
        shown = self._epochContrastDisplay(epoch)
        try:
            shown = 'contrast {g:g}'.format(g=float(shown))
        except (TypeError, ValueError):
            shown = str(shown).replace('\t', ' ').replace('\n', ' ')
        if direction is None:
            direction = epoch.get('direction')
        parts = []
        try:
            parts.append('direction {g:g} deg ({label})'.format(
                g=float(direction),
                label=self._directionLabel(direction),
            ))
        except (TypeError, ValueError):
            pass
        if hasattr(self, 'blockedHemifield'):
            parts.append('{side} hemifield'.format(
                side=str(self.blockedHemifield).strip().lower(),
            ))
            parts.append('{eye} eye'.format(eye=str(self.eye).strip().lower()))
        if (
            hasattr(self, 'tunnelVisibleDiameterDegrees')
            and not epoch.get('ramp')
        ):
            parts.append('tunnel diameter {g:g} deg'.format(
                g=float(self.tunnelVisibleDiameterDegrees),
            ))
        parts.append(shown)
        if 'baseDirection' in epoch:
            parts.append('base {g:g} deg'.format(g=float(epoch['baseDirection'])))
        if 'directionOffset' in epoch:
            parts.append('offset {o:+g} deg'.format(o=float(epoch['directionOffset'])))
        parts.append('speed {g:g} deg/s'.format(g=float(self.speed)))
        if self._usePersistentDots():
            parts.append('persistent dots')
        if getattr(self, 'attentionProbe', False):
            parts.append('{n} probes'.format(n=int(self.attentionProbeCount)))
        return '; '.join(parts)


    def _okrDirectionLabel(self, blockIndex, direction=None):
        if direction is None:
            epoch = self._okrEpoch(blockIndex)
            if epoch is not None:
                direction = epoch.get('direction')
        if direction is None:
            return 'NA'
        return self._directionLabel(direction)


    def _okrExtraFields(self, blockIndex, direction=None):
        return {
            'hemifield': 'NA',
            'tunnelDiameterDeg': 'NA',
        }


    def _okrTrialFields(self, blockIndex, direction=None):
        fields = {
            'directionDeg': self._okrDirectionDeg(blockIndex, direction),
            'condition': self._okrConditionText(blockIndex, direction),
        }
        fields.update(self._okrExtraFields(blockIndex, direction))
        return fields


    def _stampOkrTrialDirection(self, row, blockIndex, direction=None):
        '''Fill the direction column for fixation, probes, and the gray blank.'''
        label = self._okrDirectionLabel(blockIndex, direction)
        if label != 'NA':
            row['direction'] = label


    def _onOkrMotionSample(self, trialClock, blockIndex, direction):
        '''Optional per-frame OKR note during motion (tunnel diameter steps).'''
        pass


    def _closeOkrMotionAnnotations(self, endTime):
        '''Close any open per-frame OKR interval at the end of motion.'''
        pass


    def _appendOkrContrastBlock(self, events, counter, blockIndex, contrast, direction, startTime, endTime,
                                baseDirection=None, directionOffset=None):
        self._closeOkrMotionAnnotations(endTime)
        eventIndex = self._nextOkrEventIndex(counter)
        directionLabel = self._directionLabel(direction)
        dotColor = self._dotColorLabel()
        isAnchor100 = 1 if contrast >= 1.0 or contrast <= -1.0 else 0
        row = {
            'eventIndex': eventIndex,
            'eventType': 'ContrastBlock',
            'contrastBlockIndex': blockIndex,
            'startTime': startTime,
            'endTime': endTime,
            'direction': directionLabel,
            'contrastLevel': contrast,
            'dotColor': dotColor,
            'usePersistentDots': int(self._usePersistentDots()),
            'isAnchor100': isAnchor100,
        }
        row.update(self._okrTrialFields(blockIndex, direction))
        events.append(row)
        self._recordOkrSessionEvent(
            'ContrastBlock', startTime, endTime,
            direction=directionLabel,
            contrastLevel=contrast,
            blockOrEpochIndex=blockIndex,
            dotColor=dotColor,
            usePersistentDots=int(self._usePersistentDots()),
            isAnchor100=isAnchor100,
        )
        self._sendOkrEyeLinkMessage(
            'OKR ContrastBlock B{bi} contrast {c:g} dir {d} {t0:.3f}-{t1:.3f}'.format(
                bi=blockIndex, c=contrast, d=directionLabel, t0=startTime, t1=endTime,
            ),
        )


    def _epochDirectionExtra(self, epoch):
        '''Optional extra text for the on-screen epoch info (subclasses may override).'''
        return ''



    def _appendOkrAfternystagmus(self, events, counter, blockIndex, startTime, endTime):
        eventIndex = self._nextOkrEventIndex(counter)
        events.append({
            'eventIndex': eventIndex,
            'eventType': 'Afternystagmus',
            'contrastBlockIndex': blockIndex,
            'startTime': startTime,
            'endTime': endTime,
            'direction': 'NA',
            'contrastLevel': 'NA',
            'dotColor': 'NA',
            'usePersistentDots': 'NA',
            'isAnchor100': 'NA',
        })
        events[-1].update(self._okrTrialFields(blockIndex))
        self._stampOkrTrialDirection(events[-1], blockIndex)
        self._recordOkrSessionEvent(
            'Afternystagmus', startTime, endTime,
            blockOrEpochIndex=blockIndex,
        )
        self._sendOkrEyeLinkMessage(
            'OKR Afternystagmus after B{bi} {t0:.3f}-{t1:.3f}'.format(
                bi=blockIndex, t0=startTime, t1=endTime,
            ),
        )


    def _appendOkrFixation(self, events, counter, blockIndex, startTime, endTime):
        eventIndex = self._nextOkrEventIndex(counter)
        events.append({
            'eventIndex': eventIndex,
            'eventType': 'FixationITI',
            'contrastBlockIndex': blockIndex,
            'startTime': startTime,
            'endTime': endTime,
            'direction': 'NA',
            'contrastLevel': 'NA',
            'dotColor': 'NA',
            'usePersistentDots': 'NA',
            'isAnchor100': 'NA',
        })
        events[-1].update(self._okrTrialFields(blockIndex))
        self._stampOkrTrialDirection(events[-1], blockIndex)
        self._recordOkrSessionEvent(
            'FixationITI', startTime, endTime,
            blockOrEpochIndex=blockIndex,
        )
        if blockIndex is None or int(blockIndex) < 0:
            msg = 'OKR FixationITI lead-in {t0:.3f}-{t1:.3f}'.format(
                t0=startTime, t1=endTime,
            )
        else:
            msg = 'OKR FixationITI after B{bi} {t0:.3f}-{t1:.3f}'.format(
                bi=blockIndex, t0=startTime, t1=endTime,
            )
        self._sendOkrEyeLinkMessage(msg)


    def _appendOkrAttentionProbe(
        self, events, counter, blockIndex, contrast, probeNumber, startTime,
    ):
        '''Record one red attention dot in the OKR log. endTime is filled when it turns off.'''
        if events is None or counter is None:
            return
        eventIndex = self._nextOkrEventIndex(counter)
        row = {
            'eventIndex': eventIndex,
            'eventType': 'AttentionProbe',
            'contrastBlockIndex': blockIndex,
            'startTime': startTime,
            'endTime': startTime,
            'direction': 'NA',
            'contrastLevel': contrast,
            'dotColor': 'Red',
            'usePersistentDots': 'NA',
            'isAnchor100': 'NA',
            'probeNumber': probeNumber,
        }
        row.update(self._okrTrialFields(blockIndex))
        self._stampOkrTrialDirection(row, blockIndex)
        events.append(row)
        sessionRow = self._recordOkrSessionEvent(
            'AttentionProbe', startTime, startTime,
            contrastLevel=contrast,
            blockOrEpochIndex=blockIndex,
            sweep=probeNumber,
            dotColor='Red',
        )
        self._openOkrProbeEvent = row
        self._openOkrProbeSessionEvent = sessionRow


    def _closeOkrAttentionProbe(self, endTime):
        '''Set the open attention-probe row to the time the red dot turned off.'''
        row = getattr(self, '_openOkrProbeEvent', None)
        if row is not None:
            row['endTime'] = endTime
        sessionRow = getattr(self, '_openOkrProbeSessionEvent', None)
        if sessionRow is not None:
            sessionRow['protocolEndTime'] = endTime
            sessionZero = getattr(self, '_okrProtocolSessionZero', None)
            if sessionZero is not None:
                sessionRow['endTime'] = sessionZero + endTime
        self._openOkrProbeEvent = None
        self._openOkrProbeSessionEvent = None


    def _writeOkrLogFile(self, events):
        if not events:
            return None
        logDir = getattr(self, '_okrLogDir', None)
        if logDir is None:
            logDir = Path.cwd()
        else:
            logDir = Path(logDir)
        logDir.mkdir(parents=True, exist_ok=True)
        logPath = self._okrLogFilePath(logDir)
        directionPool = self._directionPool()
        directionText = ', '.join('{g:g}'.format(g=d) for d in directionPool)
        pairedAsc = self._pairedEyeLinkAscName()
        headerLines = [
            '# OKR Condition Log',
            '# StimulusName: Bassoon {name}'.format(name=self.protocolName),
        ]
        if pairedAsc:
            headerLines.append('# PairedEyeLinkAsc: {name}'.format(name=pairedAsc))
        contrastText = ', '.join('{g:g}'.format(g=float(c)) for c in self.contrasts)
        headerLines += [
            '# RecordingStarts: first fixation frame after Enter (EyeLink SYNCTIME). Waiting-room time is not in this file.',
            '# TimeBase: seconds from that frame. If tailTime is 0, t=0 is the start of the first dots trial, still after Enter.',
            '# DirectionsDeg: {dirs}'.format(dirs=directionText),
            '# Contrasts: {values}'.format(values=contrastText),
            '# SpeedDegPerSec: {v:g}'.format(v=float(self.speed)),
            '# StimTimeSec: {v:g}'.format(v=float(self.stimTime)),
            '# PreTimeSec: {v:g}'.format(v=float(self.preTime)),
            '# PostStimTimeSec: {v:g}'.format(v=float(getattr(self, 'postStimTime', 0.0))),
            '# TailTimeSec: {v:g}'.format(v=float(self.tailTime)),
            '# InterStimulusIntervalSec: {v:g}'.format(v=float(self.interStimulusInterval)),
            '# DotCount: {n}'.format(n=int(self.numberOfDots)),
            '# DotSizeDeg: {v:g}'.format(v=float(self.dotSizeDegrees)),
            '# DotLifetimeSec: {v:g}'.format(v=float(self.dotLifetime)),
            '# SpawnStaggerSec: {v:g}'.format(v=float(self.spawnStagger)),
            '# PersistentDots: {v}'.format(v=int(self._usePersistentDots())),
            '# MaxConsecutiveSameDirection: {n}'.format(n=int(self.maxConsecutiveSameDirection)),
            '# AttentionProbeEnabled: {v}'.format(v=int(bool(getattr(self, 'attentionProbe', False)))),
            '# AttentionProbeCount: {n}'.format(n=int(getattr(self, 'attentionProbeCount', 0))),
            '# AttentionProbeDiameterDeg: {v:g}'.format(v=float(getattr(self, 'attentionProbeDiameterDegrees', 0.0))),
            '# AttentionProbeSizeDeg: {v:g}'.format(v=float(self._attentionProbeSizeDegrees())),
            '# AttentionProbeResponseWindowSec: {v:g}'.format(
                v=float(getattr(self, 'attentionProbeResponseWindowSec', 1.0)),
            ),
            '# RandomSeed: {v}'.format(v=self.randomSeed),
            '# FrameRateHz: {v}'.format(v=getattr(self, '_FR', 'NA')),
            '# AttentionProbe: startTime is when the red dot appeared, endTime is when it turned off',
            '# direction is the motion direction of that trial (Right, Up, Left, Down). directionDeg is the angle.',
            '# hemifield is upper or lower for nasal-step and arcuate trials.',
            '# tunnelDiameterDeg is the clear-aperture diameter in degrees. On TunnelDiameter rows it is the size on screen for that time range.',
            '# condition names the variables for that trial.',
            'eventIndex\teventType\tcontrastBlockIndex\tstartTime\tendTime\tdirection\tcontrastLevel\tdotColor\tusePersistentDots\tisAnchor100\tprobeNumber\tdirectionDeg\themifield\ttunnelDiameterDeg\tcondition',
        ]
        rowLines = []
        for event in events:
            rowLines.append('\t'.join([
                str(event['eventIndex']),
                event['eventType'],
                str(event['contrastBlockIndex']),
                '{:.6f}'.format(event['startTime']),
                '{:.6f}'.format(event['endTime']),
                str(event['direction']),
                str(event['contrastLevel']),
                str(event['dotColor']),
                str(event['usePersistentDots']),
                str(event['isAnchor100']),
                str(event.get('probeNumber', 'NA')),
                str(event.get('directionDeg', 'NA')),
                str(event.get('hemifield', 'NA')),
                str(event.get('tunnelDiameterDeg', 'NA')),
                str(event.get('condition', 'NA')),
            ]))
        logPath.write_text('\n'.join(headerLines + rowLines) + '\n', encoding='utf-8')
        return logPath


    def respawnDot(self, win, dotRadiusPix, dot=None):
        '''Place a dot anywhere on screen (no edge inset; may clip at borders).'''
        halfW = win.size[0] / 2.0
        halfH = win.size[1] / 2.0
        xPos = random.uniform(-halfW, halfW)
        yPos = random.uniform(-halfH, halfH)
        if dot is not None:
            self.currentFrames[dot] = 0
            self.dotCoords[dot] = [xPos, yPos]
        return xPos, yPos


    def _wrapPersistentDotCoords(self, win, dotRadiusPix):
        '''Keep persistent dots on screen by wrapping at the window edges.'''
        xMin = -win.size[0] / 2.0
        xMax = win.size[0] / 2.0
        yMin = -win.size[1] / 2.0
        yMax = win.size[1] / 2.0
        xSpan = xMax - xMin
        ySpan = yMax - yMin
        self.dotCoords[:, 0] = xMin + ((self.dotCoords[:, 0] - xMin) % xSpan)
        self.dotCoords[:, 1] = yMin + ((self.dotCoords[:, 1] - yMin) % ySpan)


    def _afterDotMotion(self, win, dotRadiusPix, speedComponents):
        if self._usePersistentDots():
            self._wrapPersistentDotCoords(win, dotRadiusPix)


    def _onMotionFrame(self, frameIndex, epoch):
        '''Optional per-frame hook during motion (e.g. ramping tunnel diameter).'''
        pass


    def _onFixationFrame(self, frameIndex, nFrames):
        '''Optional per-frame hook during fixation cross (lead-in and ITI).'''
        pass


    def _markFixationOnset(self, trialClock):
        '''Align protocol t=0 and EyeLink SYNCTIME with the first fixation frame.'''
        if getattr(self, '_fixationOnsetMarked', False):
            return
        self._fixationOnsetMarked = True
        sessionClock = getattr(self, '_okrSessionClock', None)
        if sessionClock is not None:
            self._okrProtocolSessionZero = sessionClock.getTime()
        else:
            self._okrProtocolSessionZero = None
        trialClock.reset()
        sendMessage = getattr(self, '_sendEyeLinkMessage', None)
        if sendMessage is not None:
            sendMessage('SYNCTIME')


    def _afterProtocolComplete(self, win):
        '''Optional hook after all epochs finish successfully (subclasses may override).'''
        pass


    def _stimulusTitle(self):
        return 'Contrast Dots'


    def _initPerRunStimulus(self, win, pixPerDeg):
        '''Optional per-run setup hook for subclasses (e.g. gaze-contingent mask).

        pixPerDeg may be a scalar or (ppd_h, ppd_v) tuple.
        '''
        pass


    def _renderDotsFrame(self, win, dots):
        dots.draw()


    def _teardownPerRunStimulus(self):
        '''Optional per-run cleanup hook for subclasses.'''
        self._clearPrebuiltPresentation()


    def _presentationIsReady(self, win):
        return (
            getattr(self, '_prebuiltDots', None) is not None
            and getattr(self, '_prebuiltFixationCrossArms', None) is not None
            and getattr(self, '_prebuiltWin', None) is win
            and getattr(self, '_prebuiltWinSize', None) == self._winSizeKey(win)
        )


    def _clearPrebuiltPresentation(self):
        self._prebuiltDots = None
        self._prebuiltFixationCrossArms = None
        self._prebuiltWin = None
        self._prebuiltWinSize = None


    def prepareBeforeWaitingRoom(self, win):
        '''Upload the dot field, fixation cross, and probe sound before Enter.'''
        print('--> Preparing the dot field and fixation cross before the trial starts.')
        self._preparePresentation(win)


    def _preparePresentation(self, win):
        '''
        Create the stimuli that otherwise hitch between Enter and the first cross.

        Shader upload and the probe sound happen here. When this runs from the
        waiting room, Enter can flip straight to the fixation cross.
        '''
        self.getFR(win)
        if not hasattr(self, 'postStimTime'):
            self.postStimTime = 0.0
        self._interStimulusIntervalNumFrames = round(self._FR * self.interStimulusInterval)
        self._actualInterStimulusInterval = self._interStimulusIntervalNumFrames * (1 / self._FR)
        self._postStimTimeNumFrames = round(self._FR * self.postStimTime)
        self._actualPostStimTime = self._postStimTimeNumFrames * (1 / self._FR)

        ppd_h, ppd_v, dotDiameterPix, _probeDiameterPix, _dotRadiusPix = (
            self._stimulusScaleFromWin(win)
        )
        self._initPerRunStimulus(win, (ppd_h, ppd_v))
        self._initAttentionProbeSound()
        if self._presentationIsReady(win):
            return

        self.dotCoords = np.zeros((self.numberOfDots, 2))
        self._prebuiltDots = visual.ElementArrayStim(
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
        self._prebuiltFixationCrossArms = (
            visual.ShapeStim(
                win,
                units='pix',
                vertices=((-crossHalfX, 0.0), (crossHalfX, 0.0)),
                lineWidth=crossLineWidth,
                closeShape=False,
                lineColor=self.fixationCrossColor,
            ),
            visual.ShapeStim(
                win,
                units='pix',
                vertices=((0.0, -crossHalfY), (0.0, crossHalfY)),
                lineWidth=crossLineWidth,
                closeShape=False,
                lineColor=self.fixationCrossColor,
            ),
        )
        self._prebuiltWin = win
        self._prebuiltWinSize = self._winSizeKey(win)
        self._warmupPresentation(win)


    def _warmupPresentation(self, win):
        '''Draw once so the graphics driver compiles the stimuli, then cover them.'''
        try:
            self._prebuiltDots.draw()
            mask = getattr(self, '_gazeMask', None)
            if mask is not None:
                mask.draw()
            for arm in self._prebuiltFixationCrossArms:
                arm.draw()
        except Exception as err:
            print('*** Could not pre-draw the stimulus (' + str(err) + ').')
        try:
            visual.Rect(
                win,
                width=float(win.size[0]) + 4.0,
                height=float(win.size[1]) + 4.0,
                units='pix',
                pos=(0.0, 0.0),
                fillColor=self.backgroundColor,
                lineColor=None,
            ).draw()
        except Exception:
            if hasattr(win, 'clearBuffer'):
                win.clearBuffer()


    def run(self, win, informationWin):
        self._completed = 0
        self._informationWin = informationWin
        self._preparePresentation(win)

        random.seed(self.randomSeed)
        ppd_h, ppd_v, dotDiameterPix, probeDiameterPix, dotRadiusPix = (
            self._stimulusScaleFromWin(win)
        )

        if self.userInitiated:
            probeHint = ''
            if getattr(self, 'attentionProbe', False):
                probeHint = (
                    '\n{n} red dots ({size:g}\u00b0) flash near where you look — press {key} each time '
                    '(you will hear a ding)'.format(
                        n=int(self.attentionProbeCount),
                        size=self._attentionProbeSizeDegrees(),
                        key=getattr(self, 'attentionProbeKey', 'space'),
                    )
                )
            self.showInformationText(
                win,
                'Stimulus Information: {title}{probe}\nPress any key to begin'.format(
                    title=self._stimulusTitle(),
                    probe=probeHint,
                ),
            )
            event.waitKeys()

        win.color = self.backgroundColor
        # A lead-in cross replaces the waiting room on the next flip. Without
        # one, two gray flips clear the previous screen before the dots.
        if self._tailTimeNumFrames <= 0:
            win.flip()
            win.flip()

        dotLifetimeFrames = round(self._FR * self.dotLifetime)
        dots = self._prebuiltDots
        fixationCrossArms = self._prebuiltFixationCrossArms

        self.createEpochLog()
        totalEpochs = len(self._epochLog)
        # With a lead-in cross, SYNCTIME is sent on the frame the cross appears.
        self._fixationOnsetMarked = False
        if self._tailTimeNumFrames > 0:
            trialClock = core.Clock()
            self._okrProtocolSessionZero = None
        else:
            trialClock = self._startTrialClock()
        okrEvents = []
        okrEventCounter = [0]
        self._okrEvents = okrEvents
        self._okrEventCounter = okrEventCounter
        self._openOkrProbeEvent = None
        self._openOkrProbeSessionEvent = None
        self._attentionEvents = []
        probeEnabled = (
            getattr(self, 'attentionProbe', False)
            and int(getattr(self, 'attentionProbeCount', 0)) > 0
        )
        # Last ElementArrayStim index draws on top of neighboring field dots.
        probeIndex = self._attentionProbeIndex() if probeEnabled else None
        # Red flash length (visual) vs spacing so response windows do not overlap.
        probeFlashFrames = max(1, round(self._FR * float(self.dotLifetime)))
        probeSpacingFrames = max(
            probeFlashFrames,
            max(1, round(self._FR * float(getattr(self, 'attentionProbeResponseWindowSec', 1.0)))),
        )

        try:
            # Same fixation cross as inter-trial tailTime, shown before the first dots trial.
            if self._tailTimeNumFrames > 0:
                win.color = self.backgroundColor
                fixationStart = None
                for f in range(self._tailTimeNumFrames):
                    for arm in fixationCrossArms:
                        arm.draw()
                    if f == 0:
                        win.callOnFlip(self._markFixationOnset, trialClock)
                    win.flip()
                    self._onFixationFrame(f, self._tailTimeNumFrames)
                    if fixationStart is None:
                        if not self._fixationOnsetMarked:
                            self._markFixationOnset(trialClock)
                        fixationStart = 0.0
                    if self.checkQuitOrPause():
                        self._appendOkrFixation(
                            okrEvents, okrEventCounter, -1,
                            fixationStart, trialClock.getTime(),
                        )
                        return
                self._appendOkrFixation(
                    okrEvents, okrEventCounter, -1,
                    fixationStart, trialClock.getTime(),
                )

            for epochNum, epoch in enumerate(self._epochLog, start=1):
                # Re-scale if the window was resized (windowed demos / capture).
                ppd_h, ppd_v, dotDiameterPix, probeDiameterPix, dotRadiusPix = (
                    self._stimulusScaleFromWin(win)
                )
                self._syncFixationCrossGeometry(fixationCrossArms, ppd_h, ppd_v)
                contrast = float(epoch['contrast'])
                blockDirection = epoch['direction']
                blockIndex = epochNum - 1
                directionRad = math.radians(blockDirection)
                # Anisotropic deg→pix so speed is correct on both axes.
                speedComponents = np.array([
                    self.speed * ppd_h * (1 / self._FR) * math.cos(directionRad),
                    self.speed * ppd_v * (1 / self._FR) * math.sin(directionRad),
                ])
                infoExtra = self._epochDirectionExtra(epoch)
                if probeEnabled:
                    infoExtra += (
                        '\n{n} red probes ({size:g}\u00b0) near gaze — press {key} when you see one'.format(
                            n=int(self.attentionProbeCount),
                            size=self._attentionProbeSizeDegrees(),
                            key=getattr(self, 'attentionProbeKey', 'space'),
                        )
                    )
                if self._informationWin[0]:
                    self.showInformationText(
                        win,
                        'Running {title}\nContrast = {c}\nDirection = {d:g}\u00b0 ({label}){extra}\nEpoch {n} of {total}'.format(
                            title=self._stimulusTitle(),
                            c=self._epochContrastDisplay(epoch),
                            d=blockDirection,
                            label=self._directionLabel(blockDirection),
                            extra=infoExtra,
                            n=epochNum,
                            total=totalEpochs,
                        ),
                    )

                win.color = self.backgroundColor
                self.initDotPositions(win, dotRadiusPix)
                self._probeActive = False
                self._probeResponseOpen = False
                self._probeResponseUntil = None
                startContrast = self._motionContrast(epoch, 0)
                self._applyDotAppearance(
                    dots, startContrast, probeIndex, dotDiameterPix, probeDiameterPix,
                )
                dots.xys = self.dotCoords.tolist()
                event.clearEvents()
                for f in range(self._interStimulusIntervalNumFrames):
                    win.flip()
                    if self.checkQuitOrPause():
                        return

                self._stimulusStartLog.append(trialClock.getTime())
                self.sendTTL()
                self._numberOfEpochsStarted += 1

                for f in range(self._preTimeNumFrames):
                    self._applyDotAppearance(
                        dots, startContrast, probeIndex, dotDiameterPix, probeDiameterPix,
                    )
                    dots.xys = self.dotCoords.tolist()
                    self._renderDotsFrame(win, dots)
                    win.flip()
                    if self.checkQuitOrPause():
                        return

                motionStart = None
                if not self._usePersistentDots():
                    self.initDotSpawnStagger()
                probeOnsets = self._scheduleProbeOnsetFrames(
                    self._stimTimeNumFrames,
                    probeSpacingFrames,
                    int(self.attentionProbeCount) if probeEnabled else 0,
                )
                onsetSet = {frame: (i + 1) for i, frame in enumerate(probeOnsets)}
                self._probeActive = False
                self._probeResponseOpen = False

                for f in range(self._stimTimeNumFrames):
                    if self._winSizeKey(win) != getattr(self, '_lastWinSizeKey', None):
                        ppd_h, ppd_v, dotDiameterPix, probeDiameterPix, dotRadiusPix = (
                            self._stimulusScaleFromWin(win)
                        )
                        self._syncFixationCrossGeometry(fixationCrossArms, ppd_h, ppd_v)
                        speedComponents = np.array([
                            self.speed * ppd_h * (1 / self._FR) * math.cos(directionRad),
                            self.speed * ppd_v * (1 / self._FR) * math.sin(directionRad),
                        ])
                    frameContrast = self._motionContrast(epoch, f)
                    self._onMotionFrame(f, epoch)
                    if probeEnabled and f in onsetSet and probeIndex is not None:
                        self._startAttentionProbe(
                            onsetSet[f], probeIndex, ppd_h, ppd_v,
                            trialClock, blockIndex, frameContrast, f, win=win,
                        )
                    if (
                        probeEnabled
                        and getattr(self, '_probeActive', False)
                        and probeIndex is not None
                        and getattr(self, '_probeSpawnFrame', None) is not None
                        and f >= self._probeSpawnFrame + probeFlashFrames
                    ):
                        self._endAttentionProbe(
                            trialClock, blockIndex, frameContrast, ppd_h, ppd_v, probeIndex,
                        )
                    if probeEnabled:
                        self._maybeFinalizeAttentionProbeResponse(
                            trialClock, blockIndex, frameContrast, ppd_h, ppd_v, probeIndex,
                        )

                    if not self._usePersistentDots():
                        self.currentFrames += 1
                        for dot in range(self.numberOfDots):
                            if self.currentFrames[dot] >= dotLifetimeFrames:
                                # Keep the active probe in place for its full flash.
                                if (
                                    getattr(self, '_probeActive', False)
                                    and probeIndex is not None
                                    and dot == probeIndex
                                ):
                                    self.currentFrames[dot] = 0
                                    continue
                                self.respawnDot(win, dotRadiusPix, dot=dot)

                    self.dotCoords[:, 0] += speedComponents[0]
                    self.dotCoords[:, 1] += speedComponents[1]
                    self._afterDotMotion(win, dotRadiusPix, speedComponents)
                    self._applyDotAppearance(
                        dots, frameContrast, probeIndex, dotDiameterPix, probeDiameterPix,
                    )
                    dots.xys = self.dotCoords.tolist()
                    self._renderDotsFrame(win, dots)
                    win.flip()
                    if motionStart is None:
                        motionStart = trialClock.getTime()
                    self._onOkrMotionSample(trialClock, blockIndex, blockDirection)
                    if probeEnabled:
                        if self._handleMotionKeys(
                            trialClock, blockIndex, frameContrast, ppd_h, ppd_v, probeIndex,
                        ):
                            self._finalizeAttentionProbeResponse(
                                trialClock, blockIndex, frameContrast, ppd_h, ppd_v, probeIndex,
                            )
                            self._appendOkrContrastBlock(
                                okrEvents, okrEventCounter, blockIndex, contrast, blockDirection,
                                motionStart, trialClock.getTime(),
                                baseDirection=epoch.get('baseDirection', blockDirection),
                                directionOffset=epoch.get('directionOffset', 0.0),
                            )
                            return
                    elif self.checkQuitOrPause():
                        self._appendOkrContrastBlock(
                            okrEvents, okrEventCounter, blockIndex, contrast, blockDirection,
                            motionStart, trialClock.getTime(),
                            baseDirection=epoch.get('baseDirection', blockDirection),
                            directionOffset=epoch.get('directionOffset', 0.0),
                        )
                        return

                if probeEnabled:
                    self._finalizeAttentionProbeResponse(
                        trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex,
                    )

                motionEnd = trialClock.getTime()
                self._appendOkrContrastBlock(
                    okrEvents, okrEventCounter, blockIndex, contrast, blockDirection,
                    motionStart, motionEnd,
                    baseDirection=epoch.get('baseDirection', blockDirection),
                    directionOffset=epoch.get('directionOffset', 0.0),
                )

                # Gray blank for OKR afternystagmus (no dots, no fixation cross)
                afternystagmusStart = None
                for f in range(self._postStimTimeNumFrames):
                    win.flip()
                    if afternystagmusStart is None:
                        afternystagmusStart = trialClock.getTime()
                    if self.checkQuitOrPause():
                        if afternystagmusStart is not None:
                            self._appendOkrAfternystagmus(
                                okrEvents, okrEventCounter, blockIndex,
                                afternystagmusStart, trialClock.getTime(),
                            )
                        return

                if afternystagmusStart is not None:
                    self._appendOkrAfternystagmus(
                        okrEvents, okrEventCounter, blockIndex,
                        afternystagmusStart, trialClock.getTime(),
                    )

                fixationStart = None
                for f in range(self._tailTimeNumFrames):
                    for arm in fixationCrossArms:
                        arm.draw()
                    win.flip()
                    self._onFixationFrame(f, self._tailTimeNumFrames)
                    if fixationStart is None:
                        fixationStart = trialClock.getTime()
                    if self.checkQuitOrPause():
                        self._appendOkrFixation(
                            okrEvents, okrEventCounter, blockIndex,
                            fixationStart, trialClock.getTime(),
                        )
                        return

                fixationEnd = trialClock.getTime()
                self._appendOkrFixation(
                    okrEvents, okrEventCounter, blockIndex, fixationStart, fixationEnd,
                )

                self._stimulusEndLog.append(trialClock.getTime())
                self.sendTTL()
                win.flip()
                win.flip()
                self._numberOfEpochsCompleted += 1

            self._completed = 1
            self._afterProtocolComplete(win)
        finally:
            try:
                self._teardownPerRunStimulus()
            finally:
                # Audio players and celebration clips cannot be pickled with the experiment.
                self._attentionProbeSound = None
                self._celebrationSoundRefs = None
            self._okrEvents = None
            self._okrEventCounter = None
            self._openOkrProbeEvent = None
            self._openOkrProbeSessionEvent = None
            okrLogPath = self._writeOkrLogFile(okrEvents)
            if okrLogPath is not None:
                print('--> Wrote OKR condition log for slowphase-okr:', okrLogPath)
            if getattr(self, 'attentionProbe', False):
                attentionPath = self._writeAttentionProbeLog(getattr(self, '_attentionEvents', []))
                if attentionPath is not None:
                    print('--> Wrote attention probe log:', attentionPath)
