# -*- coding: utf-8 -*-
"""
Contrast Dots presents coherent moving dots on a gray background, then an
optional gray blank for afternystagmus, then a red fixation cross.

Optional attention probe: a set number of brief full-red dots appear at random
times in a disk around current gaze (each lasting one dotLifetime). Probe size is
settable (independent of field dots) and uses the last ElementArrayStim slot so it
draws on top of neighboring dots. The subject presses space when they notice each
one; a short sound plays on each press.
"""
from protocols.protocol import protocol
from psychopy import visual, event
from datetime import datetime
from pathlib import Path
import random, math
import time
import numpy as np


class ContrastDots(protocol):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDots'
        self.stimulusReps = 1 #number of repetitions through all contrast levels
        self.interStimulusInterval = 0.0 #seconds - wait time between epochs
        self.preTime = 0.0 #seconds - stationary period before dot motion
        self.stimTime = 20.0 #seconds - moving dots are shown for this duration
        self.postStimTime = 2.0 #seconds - gray background after dots end, before the red cross (for OKR afternystagmus)
        self.tailTime = 4.0 #seconds - red fixation cross after the post-stim gray blank
        self.backgroundColor = [0.0, 0.0, 0.0] #gray background (in RGB, -1 to 1)

        # Dot parameters
        self.numberOfDots = 200 #number of dots displayed at once
        self.dotColor = [1.0, 1.0, 1.0] #white dots at full contrast (in RGB, -1 to 1)
        self.contrasts = [1.0, 0.1, 0.05, -0.05, -0.1, -1.0] #list of contrast levels from -1 to 1. Total epochs = len(contrasts) * len(directions) * stimulusReps
        self.dotSizeDegrees = 1.0 #degrees - dot diameter
        self.dotLifetime = 0.15 #seconds - how long each dot stays visible before respawning; also the attention-probe flash duration
        self.spawnStagger = 0.15 #seconds - max random delay before each dot's first lifetime expiry (spreads respawns across time)
        self.direction = 90.0 #degrees - legacy single direction; ignored when directions has more than one entry
        self.directions = [90.0, 270.0] #degrees - motion directions per block (90 up, 270 down)
        self.maxConsecutiveSameDirection = 3 #maximum blocks in a row with the same direction before forcing a switch
        self.speed = 10.0 #degrees per second

        # Fixation cross shown during tail time
        self.fixationCrossColor = [1.0, -1.0, -1.0] #red (in RGB, -1 to 1)
        self.fixationCrossSizeDegrees = 1.0 #degrees - tip-to-tip span (same visual extent scale as dotSizeDegrees)

        # Attention probe: N discrete full-red flashes near gaze during each motion block.
        self.attentionProbe = True
        self.attentionProbeCount = 3 #how many red probes appear during each stimTime block
        self.attentionProbeDiameterDegrees = 0.0 #disk diameter around gaze (0 = place exactly at gaze)
        self.attentionProbeSizeDegrees = 2.0 #probe dot diameter (independent of field dotSizeDegrees)
        self.attentionProbeColor = [1.0, -1.0, -1.0] #full red (independent of grayscale contrast block)
        self.attentionProbeKey = 'space'
        self.attentionProbeSound = True #play a short ding on each spacebar press during motion


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
        if getattr(self, 'postStimTime', 2.0) < 0:
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
                if duration <= 0:
                    tf = False
                    errorMessage.append(
                        'Dot Lifetime must be greater than 0 seconds when attention probes are used '
                        '(probe duration equals dot lifetime).'
                    )
                elif float(self.stimTime) < int(self.attentionProbeCount) * duration:
                    tf = False
                    errorMessage.append(
                        'Stim Time must be at least Attention Probe Count × Dot Lifetime '
                        'so all probes can appear without overlap.'
                    )

        tfColors, colorErrorMessages = self.validateColorInput()
        tf = tf and tfColors
        errorMessage += colorErrorMessages
        return tf, errorMessage


    def estimateTime(self):
        postStimTime = getattr(self, 'postStimTime', 2.0)
        timePerEpoch = (
            self.preTime
            + self.stimTime
            + postStimTime
            + self.tailTime
            + self.interStimulusInterval
        )
        numberOfEpochs = self.stimulusReps * len(self.contrasts) * len(self._directionPool())
        self._estimatedTime = timePerEpoch * numberOfEpochs
        return self._estimatedTime


    def dotColorAtContrast(self, contrast):
        '''Linear contrast around background: 0 = background, +1 = dotColor, -1 = mirrored decrement.'''
        return [
            self.backgroundColor[i] + contrast * (self.dotColor[i] - self.backgroundColor[i])
            for i in range(3)
        ]


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
        '''Preload a short marimba-like strike so the first keypress is not delayed.'''
        self._attentionProbeSound = None
        if not getattr(self, 'attentionProbe', False):
            return
        if not getattr(self, 'attentionProbeSound', True):
            return
        try:
            from psychopy import sound
            sampleRate = 44100
            duration = 0.28
            t = np.linspace(0.0, duration, int(sampleRate * duration), endpoint=False)
            # Clean marimba key: mallet click + decaying bar partials (slightly stretched).
            f0 = 698.46  # F5 — bright, clear, not harsh
            partials = (
                (1.00, 1.00, 0.055),
                (2.00, 0.55, 0.035),
                (3.01, 0.28, 0.022),
                (4.02, 0.14, 0.015),
                (5.04, 0.07, 0.010),
            )
            wave = np.zeros_like(t)
            for ratio, amp, tau in partials:
                wave += amp * np.exp(-t / tau) * np.sin(2.0 * np.pi * f0 * ratio * t)
            # Soft mallet strike (brief noise burst, high-passed by differencing).
            rng = np.random.RandomState(7)
            noise = rng.randn(t.size).astype(np.float64)
            noise = np.concatenate([[0.0], np.diff(noise)])
            strikeEnv = np.exp(-t / 0.006)
            wave += 0.22 * noise * strikeEnv
            # Very short fade-in so the onset is clean, not clicky-DC.
            attackN = max(1, int(0.002 * sampleRate))
            wave[:attackN] *= np.linspace(0.0, 1.0, attackN)
            peak = float(np.max(np.abs(wave))) or 1.0
            wave = (0.40 * wave / peak).astype(np.float32)
            self._attentionProbeSound = sound.Sound(
                value=wave,
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
        origin = self._getAttentionProbeOriginPix(win) if win is not None else (0.0, 0.0)
        xPix, yPix, xDeg, yDeg = self._randomPointInProbeDisk(
            ppd_h, ppd_v, originPix=origin, win=win,
        )
        self.dotCoords[probeIndex] = [xPix, yPix]
        if getattr(self, 'currentFrames', None) is not None:
            self.currentFrames[probeIndex] = 0
        spawnTime = trialClock.getTime()
        self._probeActive = True
        self._probeNumber = probeNumber
        self._probeSpawnTime = spawnTime
        self._probeSpawnFrame = frameIndex
        self._probePressCount = 0
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
        self._sendOkrEyeLinkMessage(
            'OKR AttentionProbeSpawn B{bi} P{pn} contrast {c:g} t={t:.3f}'.format(
                bi=blockIndex, pn=probeNumber, c=contrast, t=spawnTime,
            )
        )


    def _endAttentionProbe(self, trialClock, blockIndex, contrast, ppd_h, ppd_v, probeIndex):
        if not getattr(self, '_probeActive', False):
            return
        endTime = trialClock.getTime()
        pressCount = int(getattr(self, '_probePressCount', 0))
        xPix, yPix = self.dotCoords[probeIndex]
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
            'OKR AttentionProbeEnd B{bi} P{pn} presses={n} t={t:.3f}'.format(
                bi=blockIndex,
                pn=getattr(self, '_probeNumber', 'NA'),
                n=pressCount,
                t=endTime,
            )
        )
        self._probeActive = False
        self._probeSpawnTime = None
        self._probeSpawnFrame = None


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
        if getattr(self, '_probeActive', False) and probeIndex is not None:
            self._probePressCount = int(getattr(self, '_probePressCount', 0)) + 1
            spawnTime = getattr(self, '_probeSpawnTime', None)
            rt = (responseTime - spawnTime) if spawnTime is not None else None
            xPix, yPix = self.dotCoords[probeIndex]
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


    def _appendOkrContrastBlock(self, events, counter, blockIndex, contrast, direction, startTime, endTime,
                                baseDirection=None, directionOffset=None):
        eventIndex = self._nextOkrEventIndex(counter)
        directionLabel = self._directionLabel(direction)
        dotColor = self._dotColorLabel()
        isAnchor100 = 1 if contrast >= 1.0 or contrast <= -1.0 else 0
        events.append({
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
        })
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
        self._recordOkrSessionEvent(
            'FixationITI', startTime, endTime,
            blockOrEpochIndex=blockIndex,
        )
        self._sendOkrEyeLinkMessage(
            'OKR FixationITI after B{bi} {t0:.3f}-{t1:.3f}'.format(
                bi=blockIndex, t0=startTime, t1=endTime,
            ),
        )


    def _writeOkrLogFile(self, events):
        if not events:
            return None
        logDir = getattr(self, '_okrLogDir', None)
        if logDir is None:
            logDir = Path.cwd()
        else:
            logDir = Path(logDir)
        logDir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logPath = logDir / ('OKR_Log_{name}_{stamp}.txt'.format(
            name=self.protocolName, stamp=stamp,
        ))
        directionPool = self._directionPool()
        directionText = ', '.join('{g:g}'.format(g=d) for d in directionPool)
        headerLines = [
            '# OKR Condition Log',
            '# StimulusName: Bassoon {name}'.format(name=self.protocolName),
            '# TimeBase: seconds from EyeLink SYNCTIME (sent when stimulus timing clock starts, after setup)',
            '# DirectionsDeg: {dirs}'.format(dirs=directionText),
            '# MaxConsecutiveSameDirection: {n}'.format(n=int(self.maxConsecutiveSameDirection)),
            'eventIndex\teventType\tcontrastBlockIndex\tstartTime\tendTime\tdirection\tcontrastLevel\tdotColor\tusePersistentDots\tisAnchor100',
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
            ]))
        logPath.write_text('\n'.join(headerLines + rowLines) + '\n', encoding='utf-8')
        return logPath


    def respawnDot(self, win, dotRadiusPix, dot=None):
        xPos = random.uniform(
            -win.size[0] / 2 + dotRadiusPix * 2,
            win.size[0] / 2 - dotRadiusPix * 2,
        )
        yPos = random.uniform(
            -win.size[1] / 2 + dotRadiusPix * 2,
            win.size[1] / 2 - dotRadiusPix * 2,
        )
        if dot is not None:
            self.currentFrames[dot] = 0
            self.dotCoords[dot] = [xPos, yPos]
        return xPos, yPos


    def _wrapPersistentDotCoords(self, win, dotRadiusPix):
        '''Keep persistent dots on screen by wrapping coordinates at the edges.'''
        margin = dotRadiusPix * 2
        xMin = -win.size[0] / 2 + margin
        xMax = win.size[0] / 2 - margin
        yMin = -win.size[1] / 2 + margin
        yMax = win.size[1] / 2 - margin
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
        pass


    def run(self, win, informationWin):
        self._completed = 0
        self._informationWin = informationWin
        self.getFR(win)

        if not hasattr(self, 'postStimTime'):
            self.postStimTime = 2.0

        self._interStimulusIntervalNumFrames = round(self._FR * self.interStimulusInterval)
        self._actualInterStimulusInterval = self._interStimulusIntervalNumFrames * (1 / self._FR)
        self._postStimTimeNumFrames = round(self._FR * self.postStimTime)
        self._actualPostStimTime = self._postStimTimeNumFrames * (1 / self._FR)

        random.seed(self.randomSeed)
        ppd_h, ppd_v = self.getPixPerDegXY(win.monitor)
        # ElementArrayStim size [w, h] so a N° diameter stays circular in degrees.
        dotDiameterPix = [
            float(self.dotSizeDegrees) * ppd_h,
            float(self.dotSizeDegrees) * ppd_v,
        ]
        probeDiameterPix = [
            self._attentionProbeSizeDegrees() * ppd_h,
            self._attentionProbeSizeDegrees() * ppd_v,
        ]
        # Respawn margin: use the larger axis so dots stay fully on-screen.
        dotRadiusPix = 0.5 * max(dotDiameterPix)

        self._initPerRunStimulus(win, (ppd_h, ppd_v))
        self._initAttentionProbeSound()

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
        win.flip()
        win.flip()

        dotLifetimeFrames = round(self._FR * self.dotLifetime)

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

        # ShapeStim arms: horizontal span uses ppd_h, vertical uses ppd_v.
        crossHalfX = 0.5 * float(self.fixationCrossSizeDegrees) * ppd_h
        crossHalfY = 0.5 * float(self.fixationCrossSizeDegrees) * ppd_v
        crossLineWidth = max(2.0, min(crossHalfX, crossHalfY) * 2.0 * 0.15)
        fixationCrossArms = (
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

        self.createEpochLog()
        totalEpochs = len(self._epochLog)
        trialClock = self._startTrialClock()
        okrEvents = []
        okrEventCounter = [0]
        self._attentionEvents = []
        probeEnabled = (
            getattr(self, 'attentionProbe', False)
            and int(getattr(self, 'attentionProbeCount', 0)) > 0
        )
        # Last ElementArrayStim index draws on top of neighboring field dots.
        probeIndex = self._attentionProbeIndex() if probeEnabled else None
        probeDurationFrames = max(1, round(self._FR * float(self.dotLifetime)))

        try:
            for epochNum, epoch in enumerate(self._epochLog, start=1):
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
                    probeDurationFrames,
                    int(self.attentionProbeCount) if probeEnabled else 0,
                )
                onsetSet = {frame: (i + 1) for i, frame in enumerate(probeOnsets)}
                self._probeActive = False

                for f in range(self._stimTimeNumFrames):
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
                        and f >= self._probeSpawnFrame + probeDurationFrames
                    ):
                        self._endAttentionProbe(
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
                    if probeEnabled:
                        if self._handleMotionKeys(
                            trialClock, blockIndex, frameContrast, ppd_h, ppd_v, probeIndex,
                        ):
                            if getattr(self, '_probeActive', False) and probeIndex is not None:
                                self._endAttentionProbe(
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

                if getattr(self, '_probeActive', False) and probeIndex is not None:
                    self._endAttentionProbe(
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
        finally:
            self._teardownPerRunStimulus()
            okrLogPath = self._writeOkrLogFile(okrEvents)
            if okrLogPath is not None:
                print('--> Wrote OKR condition log for slowphase-okr:', okrLogPath)
            if getattr(self, 'attentionProbe', False):
                attentionPath = self._writeAttentionProbeLog(getattr(self, '_attentionEvents', []))
                if attentionPath is not None:
                    print('--> Wrote attention probe log:', attentionPath)
