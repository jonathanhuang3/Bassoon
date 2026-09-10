# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:09:17 2021

@author: mrsco
"""

from protocols.protocol import protocol
from psychopy import core, visual, data, event, monitors
from psychopy.hardware import keyboard
from datetime import datetime
from pathlib import Path
import numpy as np
import serial, random, math, time, asyncio, os

# PyAV (`av`) module in `g3pylib` isn't able to access or read FFmpeg DLL files, so I manually specified it below
# os.add_dll_directory(r"C:\path\to\your\ffmpeg\bin") # EXAMPLE
try:
    os.add_dll_directory(r"C:\Users\dunnl\miniconda3\envs\Bassoon\share\ffpyplayer\ffmpeg\bin")
    from g3pylib import connect_to_glasses
except:
    print("g3pylib couldn't be imported! Tobii Glasses3 eye-tracking data will not be synced to stimulus events")

class VisualAcuityDots(protocol):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'Visual Acuity Dots' #The VisualAcuityDots protocol presents an evenly spaced array of moving dots that increase and decrease in size over time.
        self.backgroundColor = [0.0, 0.0, 0.0] #background color of the screen before/after the flash  (in RGB). -1.0 equates to 0 and 1.0 equates to 255 for 8 bit colors.
        self.stimulusReps = 5 #number of repetitions
        self.preTime = 0.75 #seconds - the amount of time before the flash on each epoch, during which the background is shown
        self.stimTime = 1 # unused for this stimulus - seconds - the amount of time that the flash lasts for
        self.tailTime = 0.75 #seconds - the amount of time after the flash on each epoch, during which the background is shown
        self.interStimulusInterval = 6.0 #seconds - fixation cross shown in the center of the screen
        
        self.speed = 5.0 # speed of dots in deg/sec
        self.spacing = 0.5 # degrees - distance between dots
        self.orientations = [90.0] # direction in degrees that dots will travel (e.g. 225 == southwest).
        self.lowerLogMAR = 0.0 # smallest logMAR value, diameter of dots corresponds to minimum angle of resolution (MAR)
        self.upperLogMAR = 1.0 # greatest logMAR value, diameter of dots corresponds to minimum angle of resolution (MAR)
        self.stepSize = 0.1 # how far values in logMAR range are from each other
        self.stepTime = 2.0 # time in seconds between each step
        self.ratio = 2/1 # disk surround:center diameter ratio (i.e. how many times larger should the surround diameter be compared to the center)
        self.centerContrast = 0.4
        self.surroundContrast = -0.19

        self.fixationCrossSize = 0.5 # size of fixation cross in degrees

        self._glasses = None # `Glasses3` object
        
    def estimateTime(self):
        '''
        Estimate the total amount of time that this protocol will take to run
        given the current parameters
        
        Value is stored as total time in seconds in the property 'self.estimatedTime'
        which is initialized by the protocol superclass.
        
        returns: estimated time in seconds
        '''
        self.stimTime = self.stepTime * ((self.upperLogMAR + self.stepSize - self.lowerLogMAR) / self.stepSize) * 2
        timePerEpoch = self.preTime + self.stimTime + self.tailTime + self.interStimulusInterval
        numberOfEpochs = self.stimulusReps
        self._estimatedTime = timePerEpoch * numberOfEpochs #return estimated time for the total stimulus in seconds
        
        return self._estimatedTime
    
    def createOrientationLog(self):
        '''
        Generate a random sequence of orientations given the desired orientations

        Desired orientations are specified as a list in self.orientations

        creates self._orientationLog, a list which specifies the orienation
        to use for each epoch
        '''
        orientations = self.orientations
        self._orientationLog = []
        #random.seed(self.randomSeed) #reinitialize the random seed

        for n in range(self.stimulusReps):
            self._orientationLog += random.sample(orientations, len(orientations))
        
        return
    
    def deg0to360(self,angle):
        '''
        Converts any angle to a value between 0 and 360 degrees
        '''
        factor = abs(int(angle/360.0))
        if angle < 0:
            angle += 360.0*(factor+1)
        if angle >= 360.0:
            angle -= 360.0*factor
        return angle

    def _directionLabel(self, direction=None):
        '''Map motion direction (degrees) to slowphase-okr direction names.'''
        angle = self.deg0to360(self.orientations[0] if direction is None else direction)
        if 45.0 <= angle < 135.0:
            return 'Up'
        if 135.0 <= angle < 225.0:
            return 'Down'
        if 225.0 <= angle < 315.0:
            return 'Left'
        return 'Right'

    def _nextOkrEventIndex(self, counter):
        counter[0] += 1
        return counter[0]

    def _sendOkrEyeLinkMessage(self, text):
        sendMessage = getattr(self, '_sendEyeLinkMessage', None)
        if sendMessage is not None:
            sendMessage(text)

    def _appendOkrLogMARStep(self, events, counter, epochIndex, sweepLabel, logMAR, direction, startTime, endTime):
        eventIndex = self._nextOkrEventIndex(counter)
        directionLabel = self._directionLabel(direction)
        events.append({
            'eventIndex': eventIndex,
            'eventType': 'LogMARStep',
            'epochIndex': epochIndex,
            'sweep': sweepLabel,
            'startTime': startTime,
            'endTime': endTime,
            'direction': directionLabel,
            'logMAR': logMAR,
        })
        self._recordOkrSessionEvent(
            'LogMARStep', startTime, endTime,
            direction=directionLabel,
            logMAR=logMAR,
            blockOrEpochIndex=epochIndex,
            sweep=sweepLabel,
        )
        self._sendOkrEyeLinkMessage(
            'OKR LogMARStep E{ei} {sw} logMAR {m:g} dir {d} {t0:.3f}-{t1:.3f}'.format(
                ei=epochIndex, sw=sweepLabel, m=logMAR, d=directionLabel,
                t0=startTime, t1=endTime,
            ),
        )

    def _appendOkrFixation(self, events, counter, epochIndex, sweepLabel, startTime, endTime):
        eventIndex = self._nextOkrEventIndex(counter)
        events.append({
            'eventIndex': eventIndex,
            'eventType': 'FixationITI',
            'epochIndex': epochIndex,
            'sweep': sweepLabel,
            'startTime': startTime,
            'endTime': endTime,
            'direction': 'NA',
            'logMAR': 'NA',
        })
        self._recordOkrSessionEvent(
            'FixationITI', startTime, endTime,
            blockOrEpochIndex=epochIndex,
            sweep=sweepLabel,
        )
        self._sendOkrEyeLinkMessage(
            'OKR FixationITI after E{ei} {sw} {t0:.3f}-{t1:.3f}'.format(
                ei=epochIndex, sw=sweepLabel, t0=startTime, t1=endTime,
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
        safeName = str(self.protocolName).replace(' ', '_')
        logPath = logDir / ('OKR_Log_{name}_{stamp}.txt'.format(
            name=safeName, stamp=stamp,
        ))
        directionText = ', '.join('{g:g}'.format(g=d) for d in self.orientations)
        headerLines = [
            '# OKR Condition Log',
            '# StimulusName: Bassoon {name}'.format(name=self.protocolName),
            '# TimeBase: seconds from EyeLink SYNCTIME (sent when stimulus timing clock starts, after setup)',
            '# DirectionsDeg: {dirs}'.format(dirs=directionText),
            '# LowerLogMAR: {v:g}'.format(v=self.lowerLogMAR),
            '# UpperLogMAR: {v:g}'.format(v=self.upperLogMAR),
            '# StepSize: {v:g}'.format(v=self.stepSize),
            '# StepTime: {v:g}'.format(v=self.stepTime),
            'eventIndex\teventType\tepochIndex\tsweep\tstartTime\tendTime\tdirection\tlogMAR',
        ]
        rowLines = []
        for event in events:
            rowLines.append('\t'.join([
                str(event['eventIndex']),
                event['eventType'],
                str(event['epochIndex']),
                str(event['sweep']),
                '{:.6f}'.format(event['startTime']),
                '{:.6f}'.format(event['endTime']),
                str(event['direction']),
                str(event['logMAR']),
            ]))
        logPath.write_text('\n'.join(headerLines + rowLines) + '\n', encoding='utf-8')
        return logPath

    def logMAR2Pix(self, logMAR, pixPerDeg):
        # pixW = 28 # arbitrary diameter of circle when logMAR = 1.0
        # pix = pixW * 10 ** (logMAR - 1)
        minimumAngleOfResolution = (10**logMAR) / 60.0
        rawPix = minimumAngleOfResolution * pixPerDeg
        # if round(rawPix) < 2: # if diameter smaller than smallest visible diameter
        #     pix = 2 # smallest visible diameter in pixels
        # else:
        #     pix = rawPix
        pix = rawPix
        # pix = round(rawPix)
        print("pix", pix, "pixPerDeg", pixPerDeg, "logMAR", logMAR)
        return pix
    
    #Coroutines
    async def connectGlasses(self):
        '''Connect to Tobii Glasses 3 if available'''
        try:
            self._glasses = await connect_to_glasses.with_hostname("tg03b-080204223211")
        except Exception:
            self._glasses = None

    async def startRecording(self):
        '''Start Tobii Glasses 3 recording if connected'''
        if self._glasses is not None:
            try:
                await self._glasses.recorder.start()
            except Exception:
                self._glasses = None

    async def stopRecording(self):
        '''Stop Tobii Glasses 3 recording'''
        if self._glasses is not None:
            try:
                await self._glasses.recorder.stop()
            except Exception:
                pass
            try:
                await self._glasses.close()
            except Exception:
                pass
            self._glasses = None

    async def sendEventData(self, event, data):
        '''Send event data to Tobii Glasses 3 if connected'''
        if self._glasses is None:
            return

        try:
            await self._glasses.recorder.send_event(event, data)
        except Exception:
            pass

    async def run_async(self, win, informationWin):
        self._completed = 0 #started but not completed
        self._informationWin = informationWin #tuple, save here so you don't have to pass this as a function parameter every time you use it

        self.getFR(win)
        self._interStimulusIntervalNumFrames = round(self._FR * self.interStimulusInterval)
        self._actualInterStimulusInterval = self._interStimulusIntervalNumFrames * 1/self._FR
        self._sweepTimeNumFrames = round((self.stimTime / 2) * self._FR)
        self._actualSweepTime = self._sweepTimeNumFrames * 1/self._FR
        self._stepTimeNumFrames = round(self._FR * self.stepTime)
        self._actualStepTime = self._stepTimeNumFrames * 1/self._FR

        stimMonitor = win.monitor
        pixPerDeg = self.getPixPerDeg(stimMonitor)
        pixPerFrame = self.speed * pixPerDeg * (1/self._FR) #in units: deg/s * pix/deg * s/frame = pixPerFrame 
        spacingPix = self.spacing * pixPerDeg
        #Pause for keystroke if the user wants to manually initiate
        if self.userInitiated:
            self.showInformationText(win, 'Stimulus Information: Flash\nPress any key to begin')
            event.waitKeys() #wait for key press 

        nx = int(win.size[0]/spacingPix) + 1 # number of dots horizontally
        ny = int(win.size[1]/spacingPix) + 1 # number of dots vertically
        numDots = nx * ny
        x = np.linspace(0, win.size[0], nx)
        y = np.linspace(0, win.size[1], ny)
        xPos, yPos = np.meshgrid(x,y)
        positions = np.column_stack((xPos.ravel(), yPos.ravel())) - win.size / 2
        diameters = np.full(numDots, self.logMAR2Pix(self.lowerLogMAR, pixPerDeg))

        dots = visual.ElementArrayStim(
            win,
            units = 'pix',
            nElements = numDots,
            elementMask ="circle",
            elementTex = None,
            xys = positions,
            sizes = diameters,
            contrs = self.centerContrast
            )

        surroundDots = visual.ElementArrayStim(
            win,
            units = 'pix',
            nElements = numDots,
            elementMask ="circle",
            elementTex = None,
            xys = positions,
            sizes = diameters * self.ratio,
            contrs = self.surroundContrast
            )

        fixationCross = visual.TextStim(
            win,
            text="+",
            pos=(0,0),
            units='pix',
        )
        fixationCross.size = self.fixationCrossSize * pixPerDeg
        logMARs = np.arange(self.lowerLogMAR, self.upperLogMAR + self.stepSize, self.stepSize)

        debugDot = visual.Circle(
            win,
            radius=5,
            units='pix',
            color=[1.0,-1.0,-1.0],
            pos= (-win.size[0]/2, 0)#(0,-win.size[1]/2)
        )

        debugLineLen = 2*math.degrees(math.atan((0.05/2)/0.75)) * pixPerDeg
        # print("line len", debugLineLen)
        debugLine = visual.Line(win=win, start=(-debugLineLen/2, 0), end=(debugLineLen/2, 0), units="pix")

        self.createOrientationLog()
        
        epochNum = 0
        kb = keyboard.Keyboard()
        okrEvents = []
        okrEventCounter = [0]

        await self.connectGlasses()
        await self.startRecording()
        trialClock = self._startTrialClock()
        try:
            for stim in range(self.stimulusReps):
                epochNum += 1
                ori = self._orientationLog[stim]
                
                #show information if necessary
                if self._informationWin[0]:
                    self.showInformationText(win, 'Running Flash\n Epoch ' + str(epochNum) + ' of ' + str(self.stimulusReps))
                
                directionRad = math.radians(self.deg0to360(ori)) # radians - direction of dot movement
                speedComponents = np.array([pixPerFrame*math.cos(directionRad), pixPerFrame*math.sin(directionRad)])
                
                index = logMAR = 0

                win.color = self.backgroundColor
                for sweep in range(2):
                    sweepLabel = 'Ascending' if sweep == 0 else 'Descending'
                    #pretime... nothing happens
                    self._stimulusStartLog.append(trialClock.getTime())
                    self.sendTTL()
                    self._numberOfEpochsStarted += 1
                    await self.sendEventData("Pre time", {})
                    for f in range(self._preTimeNumFrames):
                        surroundDots.draw()
                        dots.draw()
                        win.flip()
                        if self.checkQuitOrPause():
                            return
                    
                    #stim time - dots move
                    count = 0
                    if sweep == 0:
                        await self.sendEventData("Ascending sweep", {})
                    else:
                        await self.sendEventData("Descending sweep", {})

                    stepStart = None
                    currentLogMAR = logMARs[index]
                    for f in range(self._sweepTimeNumFrames):
                        count += 1
                        if count >= self._stepTimeNumFrames:
                            count = 0
                            canStep = (
                                (sweep == 0 and index < logMARs.size - 1)
                                or (sweep == 1 and index > 0)
                            )
                            if canStep:
                                if stepStart is not None:
                                    self._appendOkrLogMARStep(
                                        okrEvents, okrEventCounter, epochNum, sweepLabel,
                                        currentLogMAR, ori, stepStart, trialClock.getTime(),
                                    )
                                    stepStart = None
                                if sweep == 0:
                                    index += 1
                                else:
                                    index -= 1
                                logMAR = logMARs[index]
                                currentLogMAR = logMAR
                                diameters[:] = self.logMAR2Pix(logMAR, pixPerDeg)
                                dots.sizes = diameters
                                surroundDots.sizes = diameters * self.ratio

                        keyPressed = kb.getKeys()
                        if keyPressed:
                            print("key press", logMAR)

                        dots.xys = surroundDots.xys = (dots.xys + speedComponents + win.size/2) % win.size - (win.size / 2)
                        debugDot.pos += speedComponents[::-1]
                        # debugDot.draw()
                        # debugLine.draw()
                        surroundDots.draw()
                        dots.draw()
                        win.flip()
                        if stepStart is None:
                            stepStart = trialClock.getTime()

                        if self.checkQuitOrPause():
                            if stepStart is not None:
                                self._appendOkrLogMARStep(
                                    okrEvents, okrEventCounter, epochNum, sweepLabel,
                                    currentLogMAR, ori, stepStart, trialClock.getTime(),
                                )
                            return
                    
                    if stepStart is not None:
                        self._appendOkrLogMARStep(
                            okrEvents, okrEventCounter, epochNum, sweepLabel,
                            currentLogMAR, ori, stepStart, trialClock.getTime(),
                        )
                    
                    #tail time
                    await self.sendEventData("Tail time", {})
                    for f in range(self._tailTimeNumFrames):
                        surroundDots.draw()
                        dots.draw()
                        win.flip()
                        if self.checkQuitOrPause():
                            return
                    
                    #pause for inter stimulus interval
                    await self.sendEventData("Fixation interval", {})
                    fixationStart = None
                    for f in range(self._interStimulusIntervalNumFrames):
                        fixationCross.draw()
                        win.flip()
                        if fixationStart is None:
                            fixationStart = trialClock.getTime()
                        if self.checkQuitOrPause():
                            if fixationStart is not None:
                                self._appendOkrFixation(
                                    okrEvents, okrEventCounter, epochNum, sweepLabel,
                                    fixationStart, trialClock.getTime(),
                                )
                            return

                    if fixationStart is not None:
                        self._appendOkrFixation(
                            okrEvents, okrEventCounter, epochNum, sweepLabel,
                            fixationStart, trialClock.getTime(),
                        )
            
                self._stimulusEndLog.append(trialClock.getTime())
                self.sendTTL()
                
                self._numberOfEpochsCompleted += 1
        finally:
            await self.stopRecording()
            okrLogPath = self._writeOkrLogFile(okrEvents)
            if okrLogPath is not None:
                print('--> Wrote OKR condition log for slowphase-okr:', okrLogPath)

        self._completed = 1

    def run(self, win, informationWin):
        '''
        Executes the Visual Acuity Dot stimulus
        '''
        asyncio.run(self.run_async(win, informationWin))
