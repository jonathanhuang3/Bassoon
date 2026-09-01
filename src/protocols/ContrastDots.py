# -*- coding: utf-8 -*-
"""
Contrast Dots presents coherent moving dots on a gray background, followed by
a red fixation cross.
"""
from protocols.protocol import protocol
from psychopy import visual, event
from datetime import datetime
from pathlib import Path
import random, math
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
        self.tailTime = 4.0 #seconds - red fixation cross after dot motion
        self.backgroundColor = [0.0, 0.0, 0.0] #gray background (in RGB, -1 to 1)

        # Dot parameters
        self.numberOfDots = 100 #number of dots displayed at once
        self.dotColor = [1.0, 1.0, 1.0] #white dots at full contrast (in RGB, -1 to 1)
        self.contrasts = [1.0, 0.1, 0.05, -0.05, -0.1, -1.0] #list of contrast levels from -1 to 1. Total epochs = len(contrasts) * len(directions) * stimulusReps
        self.dotSizeDegrees = 1.0 #degrees - dot diameter
        self.dotLifetime = 0.15 #seconds - how long each dot stays visible before respawning
        self.spawnStagger = 0.15 #seconds - max random delay before each dot's first lifetime expiry (spreads respawns across time)
        self.direction = 90.0 #degrees - legacy single direction; ignored when directions has more than one entry
        self.directions = [90.0, 270.0] #degrees - motion directions per block (90 up, 270 down)
        self.maxConsecutiveSameDirection = 3 #maximum blocks in a row with the same direction before forcing a switch
        self.speed = 10.0 #degrees per second

        # Fixation cross shown during tail time
        self.fixationCrossColor = [1.0, -1.0, -1.0] #red (in RGB, -1 to 1)
        self.fixationCrossSizeDegrees = 1.0 #degrees - height of the fixation cross


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

        tfColors, colorErrorMessages = self.validateColorInput()
        tf = tf and tfColors
        errorMessage += colorErrorMessages
        return tf, errorMessage


    def estimateTime(self):
        timePerEpoch = self.preTime + self.stimTime + self.tailTime + self.interStimulusInterval
        numberOfEpochs = self.stimulusReps * len(self.contrasts) * len(self._directionPool())
        self._estimatedTime = timePerEpoch * numberOfEpochs
        return self._estimatedTime


    def dotColorAtContrast(self, contrast):
        '''Linear contrast around background: 0 = background, +1 = dotColor, -1 = mirrored decrement.'''
        return [
            self.backgroundColor[i] + contrast * (self.dotColor[i] - self.backgroundColor[i])
            for i in range(3)
        ]


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
        '''Map motion direction (degrees) to slowphase-okr direction names.'''
        angle = self.deg0to360(self.direction if direction is None else direction)
        if 45.0 <= angle < 135.0:
            return 'Up'
        if 135.0 <= angle < 225.0:
            return 'Down'
        if 225.0 <= angle < 315.0:
            return 'Left'
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


    def _appendOkrContrastBlock(self, events, counter, blockIndex, contrast, direction, startTime, endTime):
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
        self._sendOkrEyeLinkMessage(
            'OKR ContrastBlock B{bi} contrast {c:g} dir {d} {t0:.3f}-{t1:.3f}'.format(
                bi=blockIndex, c=contrast, d=directionLabel, t0=startTime, t1=endTime,
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


    def _stimulusTitle(self):
        return 'Contrast Dots'


    def _initPerRunStimulus(self, win, pixPerDeg):
        '''Optional per-run setup hook for subclasses (e.g. gaze-contingent mask).'''
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

        self._interStimulusIntervalNumFrames = round(self._FR * self.interStimulusInterval)
        self._actualInterStimulusInterval = self._interStimulusIntervalNumFrames * (1 / self._FR)

        random.seed(self.randomSeed)
        pixPerDeg = self.getPixPerDeg(win.monitor)
        dotRadiusPix = (self.dotSizeDegrees / 2.0) * pixPerDeg

        self._initPerRunStimulus(win, pixPerDeg)

        if self.userInitiated:
            self.showInformationText(
                win,
                'Stimulus Information: {title}\nPress any key to begin'.format(
                    title=self._stimulusTitle(),
                ),
            )
            event.waitKeys()

        win.color = self.backgroundColor
        win.flip()
        win.flip()

        dotLifetimeFrames = round(self._FR * self.dotLifetime)

        self.dotCoords = np.zeros((self.numberOfDots, 2))
        dotDiameterPix = 2 * dotRadiusPix

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

        fixationCross = visual.TextStim(
            win,
            text='+',
            color=self.fixationCrossColor,
            height=self.fixationCrossSizeDegrees * pixPerDeg,
            units='pix',
        )

        self.createEpochLog()
        totalEpochs = len(self._epochLog)
        trialClock = self._startTrialClock()
        okrEvents = []
        okrEventCounter = [0]

        try:
            for epochNum, epoch in enumerate(self._epochLog, start=1):
                contrast = epoch['contrast']
                blockDirection = epoch['direction']
                blockIndex = epochNum - 1
                pixPerFrame = self.speed * pixPerDeg * (1 / self._FR)
                directionRad = math.radians(blockDirection)
                speedComponents = np.array([
                    pixPerFrame * math.cos(directionRad),
                    pixPerFrame * math.sin(directionRad),
                ])
                if self._informationWin[0]:
                    self.showInformationText(
                        win,
                        'Running {title}\nContrast = {c}\nDirection = {d:g}\u00b0 ({label})\nEpoch {n} of {total}'.format(
                            title=self._stimulusTitle(),
                            c=contrast,
                            d=blockDirection,
                            label=self._directionLabel(blockDirection),
                            n=epochNum,
                            total=totalEpochs,
                        ),
                    )

                win.color = self.backgroundColor
                dots.colors = self.dotColorAtContrast(contrast)
                self.initDotPositions(win, dotRadiusPix)
                dots.xys = self.dotCoords.tolist()
                for f in range(self._interStimulusIntervalNumFrames):
                    win.flip()
                    if self.checkQuitOrPause():
                        return

                self._stimulusStartLog.append(trialClock.getTime())
                self.sendTTL()
                self._numberOfEpochsStarted += 1

                for f in range(self._preTimeNumFrames):
                    self._renderDotsFrame(win, dots)
                    win.flip()
                    if self.checkQuitOrPause():
                        return

                motionStart = None
                if not self._usePersistentDots():
                    self.initDotSpawnStagger()
                for f in range(self._stimTimeNumFrames):
                    if not self._usePersistentDots():
                        self.currentFrames += 1
                        for dot in range(self.numberOfDots):
                            if self.currentFrames[dot] >= dotLifetimeFrames:
                                self.respawnDot(win, dotRadiusPix, dot=dot)
                    self.dotCoords[:, 0] += speedComponents[0]
                    self.dotCoords[:, 1] += speedComponents[1]
                    self._afterDotMotion(win, dotRadiusPix, speedComponents)
                    dots.xys = self.dotCoords.tolist()
                    self._renderDotsFrame(win, dots)
                    win.flip()
                    if motionStart is None:
                        motionStart = trialClock.getTime()
                    if self.checkQuitOrPause():
                        self._appendOkrContrastBlock(
                            okrEvents, okrEventCounter, blockIndex, contrast, blockDirection,
                            motionStart, trialClock.getTime(),
                        )
                        return

                motionEnd = trialClock.getTime()
                self._appendOkrContrastBlock(
                    okrEvents, okrEventCounter, blockIndex, contrast, blockDirection,
                    motionStart, motionEnd,
                )

                fixationStart = None
                for f in range(self._tailTimeNumFrames):
                    fixationCross.draw()
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
