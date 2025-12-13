# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:39:25 2023

@author: mrsco
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:46:55 2021

@author: mrsco
"""
from protocols.protocol import protocol
from psychopy import core, visual, data, event, monitors
import serial, random, math, time
import numpy as np

class DirectionalDots(protocol):
    def __init__(self):
        super().__init__()
        # Protocol parameters
        self.protocolName = 'DirectionalDots' # Directional Dots presents moving dots in a specified direction with a proportion of dots moving in an aberrant direction.
        self.stimulusReps = 3 #number of repetitions of the stimulus. The total number of epochs is equal to the number of orientations times the number of stimulus reps.   
        self.interStimulusInterval = 1.0 #seconds - the wait time between each epoch. The background color is displayed during this time
        self.preTime = 3.0 #20.0 #seconds
        self.stimTime = 0.0 #seconds - the total time that the grating moves for. This number is calculated during run time and should not be edited directly. It is listed here as a dummy variable because it is required by the getFR method in protocol.py
        self.tailTime = 3.0 #20.0 #seconds
        self.backgroundColor = [0.0, 0.0, 0.0] # Background color of the screen (in RGB). -1.0 equates to 0 and 1.0 equates to 255 for 8 bit colors.
        
        # Scotoma parameters
        self.scotomaStartFraction = 0.0 #Fraction of pixels that start as a scotoma. This number can be between 0.0 and 1.0.
        self.scotomaEndFraction = 1.0 #Fraction of pixels that end as a scotoma after one growth/decay period. This number can be between 0.0 and 1.0
        self.scotomaOpacity = 1.0 #The opacity of the scotomas. 1.0 is fully opaque, 0.0 is fully transparent
        self.scotomaReverse = True #bool - True means that the scotomas will go from start to end and then end to start again. False means they will only go from start to end (press the information button on protocolName for more insight).
        self.scotomaGrowthTime = 10.0 #80.0 #seconds - the amount of time that one growth/decay period takes.
        self.scotomaBookendTime = 5.0 #10.0 #seconds - the amount of time that each bookend takes. During the bookends, the grating moves but the scotomas do not change. All bookends must be the same length. There are a maximum of three bookends per epoch: Once immediately after the pretime, once immediately before the tail time, and (if scotomaReverse is set to True) once between the first and second growth/decay times.
        self.scotomaGrowth = 'lin' #Sets the pattern of scotoma growth. Currently only 'lin' is supported for linear growth
        self.scotomaSize = 0.25 #degrees - The size of the scotomas (height and width). Scotomas will be square, such that height == width
        self.scotomaColor = [0.0, 0.0, 0.0] #The color of the scotomas (in RGB).-1.0 equates to 0 and 1.0 equates to 255 for 8 bit colors.
                
        # Dot parameters
        self.numberOfDots = 100 # Number of dots to be displayed
        self.proportionAberrant = 0.0 # Proportion of dots that are moving in a different direction
        self.dotColor = [1.0, 1.0, 1.0] #Color of the dot (in RGB). -1.0 equates to 0 and 1.0 equates to 255 for 8 bit colors.
        self.opacity = 1.0 # Opacity of the aberrant dots (0.0 to 1.0)
        self.dotRadius_degrees = 0.5 #degrees - radius of the smallest optotype
        self.dotLifetime = 0.1 # seconds
        self.lifetimeOffset = 0.025 # seconds - maximum possible offset so that each dot's lifetime ends at slightly different times
        self.orientations = [90.0] # direction in degrees that dots will travel (e.g. 225 == southwest).
        self.directionAberrant = 0.0 # direction in degrees that aberrant dots will travel (e.g. 225 == southwest).
        self.speed = 6.0 # degrees/second - how fast the dots will move in user-specified direction                                                                                                                                                                                                                                                                                                                                                                                                                                              


    
    def internalValidation(self):
        '''
        Checks that direction is between 0 to 360 and that 
        '''
        tf = True
        errorMessage = []
        # Check that proportion is a valid proportion
        
        if self.proportionAberrant < 0 or self.proportionAberrant > 1:
            tf = False
            errorMessage.append("Value in Proportion Aberrant must be between 0 and 1.")
        
        if self.scotomaStartFraction > 1 or self.scotomaStartFraction < 0 or self.scotomaEndFraction > 1 or self.scotomaEndFraction < 0:
            tf = False
            errorMessage.append(
            'The scotoma start and end fractions must be a value between 0 and 1. If set to 1, all pixels will be blanked. ' \
            'If set to 0, no pixels will be blanked. If set to 0.5, half of the pixels will be blanked.'
            )

        if self.scotomaGrowth != 'lin':
            tf = False
            errorMessage.append('The Scotoma Growth parameter must be set to "lin". Other growth options will be supported in future versions.')
                
        if self.scotomaOpacity > 1 or self.scotomaOpacity < 0:
            tf = False
            errorMessage.append('Scotoma Opacity must be between 0.0 and 1.0. 1 is fully opaque, 0 is transparent.')
        
        if self.stimTime != 0:
            self.stimTime = 0
            print('\nNOTE: Stim Time was reset to 0. It will be updated on the fly during run time. Users should not manually change this parameter for the Scotoma Moving Grating stimulus.')
        
        if self.scotomaSize < 0.5:
            print('\nNOTE: Scotoma Size is small. This may slow down the frame rate. If this occurs, increase scotoma size to improve the frame rate')
                                    
        # Checks color values
        tf, colorErrorMessages = self.validateColorInput()
        errorMessage += colorErrorMessages

        return tf, errorMessage

    def estimateTime(self):
        '''
        Estimate the total amount of time that this protocol will take to run
        given the current parameters
        
        Value is stored as total time in seconds in the property 'self.estimatedTime'
        which is initialized by the protocol superclass.
        
        returns: estimated time in seconds
        '''
        if self.scotomaReverse:
            st = 3*self.scotomaBookendTime + 2*self.scotomaGrowthTime
        else:
            st = 2*self.scotomaBookendTime + self.scotomaGrowthTime
            
        timePerEpoch = self.preTime + st + self.tailTime + self.interStimulusInterval
        numberOfEpochs = self.stimulusReps * len(self.orientations)
        self._estimatedTime = timePerEpoch * numberOfEpochs #return estimated time for the total stimulus in seconds

        return self._estimatedTime
    
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
    
    def lifetime(self, win, dot=None):
        '''
        Restarts lifetime timer and randomizes dot position 
        Returns: new randomized (x,y) position
        '''
        xPos = yPos = None
        xPos = random.uniform(
            -win.size[0]/2 + self.dotRadius_pix*2, 
            win.size[0]/2 - self.dotRadius_pix*2
            )
        yPos = random.uniform(
            -win.size[1]/2 + self.dotRadius_pix*2,
            win.size[1]/2 - self.dotRadius_pix*2
            ) 
        if dot:
            self.currentFrames[dot] = 0 # Start lifetime timer for this dot      
            self.dotCoords[dot] = [xPos, yPos] # Randomize dot position 
        
        return xPos, yPos
    
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

    def createScotomaGrowthSequence(self, numScotomasToAdd, numTotalScotomas, scotomaIndices):
        '''
        Builds a sequence of scotoma indices to add or remove from the mask
        
        Inputs:
            - numScotomasToAdd: the total number of scotomas you need to add by the end of the growth sequence. Can be a positive or negative integer. If this number is negative, you'll be taking away scotomas from the mask rather than adding them
            - scotomaIndices: list 1d indices where scotomas have already been filled
            
        Returns:
            - no explicit returns, but creates variables self._newScotomasPerFrame (number of new scotomas to add on each frame) and self._scotomaSequence (list of indices that indicate where to add or take away scotomas in sequence)
        '''
        
        if self.scotomaGrowth == 'lin':
            #First figure out an integer number of scotomas to add on each frame. Consider that you need to add an integer number on each frame, and the frame rate may not divide the growth rate evenly
            meanScotomasAddedPerFrame = numScotomasToAdd / self._numFramesGrowth
            cumulativeScotomasAdded = np.round(np.cumsum([meanScotomasAddedPerFrame for f in range(self._numFramesGrowth)]))
            self._newScotomasPerFrame = np.diff(cumulativeScotomasAdded) #new scotomas per frame is the number of new additions to make on every sequential frame.
            self._newScotomasPerFrame = [int(el) for el in self._newScotomasPerFrame] #convert to list of integers
            self._newScotomasPerFrame = [round(meanScotomasAddedPerFrame)] + self._newScotomasPerFrame
            
        elif self.scotomaGrowth == 'exp':
            #   --- enter code here for exponential growth pattern --- #
            pass
            
        #if adding scotomas:
        if numScotomasToAdd > 0:
            noScotomaIndices = list(set([i for i in range(numTotalScotomas)]) - set(scotomaIndices))
            self._scotomaSequence = np.array(random.sample(noScotomaIndices, numScotomasToAdd)) 

        #if subtracting scotomas:
        if numScotomasToAdd < 0:
            self._scotomaSequence = np.array(random.sample(scotomaIndices, -numScotomasToAdd))
            self._newScotomasPerFrame = [-x for x in self._newScotomasPerFrame]

        if numScotomasToAdd == 0:
            self._scotomaSequence = np.array([0])
        
        self._scotomaSequence = self._scotomaSequence.tolist() # make it a list b/c ndarrays have trouble saving
        
        return
    
    def run(self, win, informationWin):
        '''
        Executes the Directional Dots stimulus
        '''
        self._completed = 0 #started but not completed
        self._informationWin = informationWin #tuple, save here so you don't have to pass this as a function parameter every time you use it
        self.getFR(win)

        random.seed(self.randomSeed) #reinitialize the random seed        

        #update the stim time
        if self.scotomaReverse:
            self.stimTime = 3*self.scotomaBookendTime + 2*self.scotomaGrowthTime
        else:
            self.stimTime = 2*self.scotomaBookendTime + self.scotomaGrowthTime
                    
        self._interStimulusIntervalNumFrames = round(self._FR * self.interStimulusInterval)
        self._actualInterStimulusInterval = self._interStimulusIntervalNumFrames * 1/self._FR        
        
        stimMonitor = win.monitor

        pixPerDeg = self.getPixPerDeg(stimMonitor)
        pixPerFrame = self.speed * pixPerDeg * (1/self._FR) #in units: deg/s * pix/deg * s/frame = pixPerFrame 
        
        scotomaSizePix = int(self.scotomaSize*pixPerDeg) #maybe a slight rounding error here by using int
        numAberrantDots = round(self.numberOfDots * self.proportionAberrant)

        # Specify the x and y center coordinates for each check
        xCoordinates = [x - win.size[0]/2 for x in range(-scotomaSizePix, win.size[0]+scotomaSizePix, scotomaSizePix)]
        yCoordinates = [y - win.size[1]/2 for y in range(-scotomaSizePix, win.size[1]+scotomaSizePix, scotomaSizePix)]
        numTotalScotomas = len(xCoordinates) * len(yCoordinates)
        sizes = [(scotomaSizePix, scotomaSizePix) for i in range(numTotalScotomas)]
        
        self._scotomaCoordinates = []
        colors = []
        for i in range(len(xCoordinates)):
            for j in range(len(yCoordinates)):
                self._scotomaCoordinates.append([xCoordinates[i], yCoordinates[j]])
                
        scotomaMask = visual.ElementArrayStim(
            win,
            nElements = numTotalScotomas,
            elementMask="None",
            elementTex = None,
            xys = self._scotomaCoordinates,
            sizes = sizes,
            colors = self.scotomaColor
            )
        
        mask = np.zeros((numTotalScotomas, 1)) #1 is fully transparent, -1 is fully opaque. Start with a fully transparent mask.    
        
        #fill the mask with the number of scotomas needed at the start of the stimulus
        numScotomasStart = round(numTotalScotomas*self.scotomaStartFraction)
        scotomaIndices = random.sample([i for i in range(numTotalScotomas)], numScotomasStart)
        mask[scotomaIndices] = self.scotomaOpacity            
        scotomaMask.opacities = mask #set the first mask 

        #Now build up a list of which mask locations you want to update on each frame 
        numScotomasEnd = round(numTotalScotomas*self.scotomaEndFraction)
        numScotomasToAdd = numScotomasEnd - numScotomasStart #note, this value can be positive or negative.
            
        self._numFramesGrowth = round(self._FR * self.scotomaGrowthTime) #number of frames overwhich the scotoma will grow
        self._actualScotomaGrowthTime = self._numFramesGrowth * 1/self._FR
        
        self._numFramesBookend = round(self._FR * self.scotomaBookendTime)
        self._actualBookendTime = self._numFramesBookend * 1/self._FR
        
        #create self._scotomaSequence and self._newScotomasPerFrame
        self.createScotomaGrowthSequence(numScotomasToAdd, numTotalScotomas, scotomaIndices)
            
        #create flipped copies of self._scotomaSequence and self._newScotomasPerFrame if you will also be doing a reverse
        if self.scotomaReverse:
            scotomaSequenceReverse = np.flip(self._scotomaSequence)
            newScotomasPerFrameReverse = np.flip(self._newScotomasPerFrame)        
        
        # Specify dot parameters
        self.dotRadius_pix = self.dotRadius_degrees * pixPerDeg

        #Pause for keystroke if the user wants to manually initiate
        if self.userInitiated:
            self.showInformationText(win, 'Stimulus Information: Directional Dots \nPress any key to begin')
            event.waitKeys() #wait for key press  
                
        win.color = self.backgroundColor
        win.flip()
        win.flip()
        
        # Create all dots as an elementArrayStim object (see Psychopy documentation for more info)
        self.dotCoords = np.zeros((self.numberOfDots, 2))
        diameters = []
        aberrantDots = []
        dotIntervalFrames = []
        for d in range(self.numberOfDots):
            # Assign random [x,y] coordinate lists to a list
            xPos, yPos = self.lifetime(win)
            self.dotCoords[d][0] = xPos
            self.dotCoords[d][1] = yPos
            # Add list of dot diameters
            diameters.append(2*self.dotRadius_pix)
            # Add user-specified number of aberrant dots 
            aberrantTF = True if numAberrantDots > d else False
            aberrantDots.append(aberrantTF)
            # Assign a dot lifetime (in frames) to each dot
            startTime = random.uniform(0, self.lifetimeOffset) # seconds - 
            interval = self.dotLifetime - startTime # seconds
            intervalFrames = self._FR * interval # num of frames dot will move before repositioning
            dotIntervalFrames.append(intervalFrames)
        aberrant_mask = np.array(aberrantDots) # Logical array: aberrant dot = True, normal dot= False
        normal_mask = ~np.array(aberrantDots) # Logical array: inverse of above
        
        dots = visual.ElementArrayStim(
            win,
            units = 'pix',
            nElements = self.numberOfDots,
            elementMask="circle",
            elementTex = None,
            xys = self.dotCoords.tolist(), 
            sizes = diameters, 
            colors = self.dotColor
            )
        
        self.createOrientationLog()

        #assign parameters
        totalEpochs = len(self._orientationLog)
        epochNum = 0
        trialClock = core.Clock() #this will reset every trial  

        # --- Stimulus Loop --- #
        for ori in self._orientationLog:
            epochNum += 1
            #show information if necessary
            if self._informationWin[0]:
                self.showInformationText(win, 'Running Directional Dots. Current orientation = ' + \
                                         str(ori) + '\n Epoch ' + str(epochNum) + ' of ' + str(totalEpochs))
            
            # Specify dot movement parameters
            directionRad = math.radians(self.deg0to360(ori)) # radians - direction of dot movement 
            aberrantDirRad = math.radians(self.deg0to360(self.directionAberrant)) # radians - direction of aberrant dot movement 
            speedComponents = np.array([pixPerFrame*math.cos(directionRad), pixPerFrame*math.sin(directionRad)])
            aberrantComponents = np.array([pixPerFrame*math.cos(aberrantDirRad), 2*pixPerFrame*math.sin(aberrantDirRad)])
            
            ## Inter-stimulus interval ##
            win.color = self.backgroundColor
            for f in range(self._interStimulusIntervalNumFrames):
                win.flip()
                if self.checkQuitOrPause():
                    return
            
            ## Pre-time ## (stationary dots)
            self._stimulusStartLog.append(trialClock.getTime())
            self.sendTTL()
            self._numberOfEpochsStarted += 1
            for f in range(self._preTimeNumFrames):
                dots.draw()
                scotomaMask.draw()
                win.flip()
                if self.checkQuitOrPause():
                    return
            
            ## Stim-time ##
            self.spawnTimes = [0 for d in range(self.numberOfDots)]
            self.currentFrames = np.zeros(self.numberOfDots)
            # First bookend
            for f in range(self._numFramesBookend):
                self.currentFrames += 1
                for dot in range(self.numberOfDots):
                    if self.currentFrames[dot] >= dotIntervalFrames[dot]:
                        self.lifetime(win, dot=dot) # Reposition dot at another random location
                self.dotCoords = np.array(self.dotCoords) # list --> np array
                # Move normal dots
                if self.dotCoords[normal_mask][:].size != 0:
                    self.dotCoords[normal_mask, 0] += speedComponents[0]
                    self.dotCoords[normal_mask, 1] += speedComponents[1]
                # Move aberrant dots 
                if self.dotCoords[aberrant_mask][:].size != 0:
                    self.dotCoords[aberrant_mask, 0] += aberrantComponents[0]
                    self.dotCoords[aberrant_mask, 1] += aberrantComponents[1]
                dots.xys = self.dotCoords.tolist() # Move the dots to their new positions
                dots.draw()               
                scotomaMask.draw()
                win.flip()
                if self.checkQuitOrPause():
                    return
                
            #first check whether you will be adding or taking away scotomas. Assign the addColor accordingly so that when you update the mask it either places a scotoma or sets the value to transparent
            addColor = self.scotomaOpacity if numScotomasToAdd > 0 else 0
            
            #scotoma growth starts here
            count = 0 
            for f in range(self._numFramesGrowth):
                scotomasToChangeThisFrame = self._scotomaSequence[count:count+self._newScotomasPerFrame[f]]
                count += self._newScotomasPerFrame[f]
                mask[scotomasToChangeThisFrame] = addColor
                scotomaMask.opacities = mask
                self.currentFrames += 1
                for dot in range(self.numberOfDots):
                    if self.currentFrames[dot] >= dotIntervalFrames[dot]:
                        self.lifetime(win, dot=dot) # Reposition dot at another random location
                # Move normal dots
                self.dotCoords = np.array(self.dotCoords) # list --> np array
                if self.dotCoords[normal_mask][:].size != 0:
                    self.dotCoords[normal_mask, 0] += speedComponents[0]
                    self.dotCoords[normal_mask, 1] += speedComponents[1]
                # Move aberrant dots 
                if self.dotCoords[aberrant_mask][:].size != 0:
                    self.dotCoords[aberrant_mask, 0] += aberrantComponents[0]
                    self.dotCoords[aberrant_mask, 1] += aberrantComponents[1]
                dots.xys = self.dotCoords.tolist() # Move the dots to their new positions                
                dots.draw()                
                scotomaMask.draw()
                win.flip()
                if self.checkQuitOrPause():
                    return
            
            # Middle bookend (if applicable)
            if self.scotomaReverse:
                #pause time before reversal
                for f in range(self._numFramesBookend):
                    self.currentFrames += 1
                    for dot in range(self.numberOfDots):
                        if self.currentFrames[dot] >= dotIntervalFrames[dot]:
                            self.lifetime(win, dot=dot) # Reposition dot at another random location
                    self.dotCoords = np.array(self.dotCoords) # list --> np array
                    # Move normal dots
                    if self.dotCoords[normal_mask][:].size != 0:
                        self.dotCoords[normal_mask, 0] += speedComponents[0]
                        self.dotCoords[normal_mask, 1] += speedComponents[1]
                    # Move aberrant dots 
                    if self.dotCoords[aberrant_mask][:].size != 0:
                        self.dotCoords[aberrant_mask, 0] += aberrantComponents[0]
                        self.dotCoords[aberrant_mask, 1] += aberrantComponents[1]
                    dots.xys = self.dotCoords # Move the dots to their new positions                    
                    dots.draw()                   
                    scotomaMask.draw()
                    win.flip()
                if self.checkQuitOrPause():
                    return
                
                #first check whether you will be adding or taking away scotomas. Assign the addColor accordingly so that when you update the mask it either places a scotoma or sets the value to transparent
                addColor = self.scotomaOpacity if numScotomasToAdd < 0 else 0
                #flip the scotoma sequence and scotomas to change this frame lists
                count = 0
                for f in range(self._numFramesGrowth): 
                    scotomasToChangeThisFrame = scotomaSequenceReverse[count:count+newScotomasPerFrameReverse[f]]
                    count += newScotomasPerFrameReverse[f]
                    mask[scotomasToChangeThisFrame] = addColor
                    scotomaMask.opacities = mask
                    self.currentFrames += 1
                    for dot in range(self.numberOfDots):
                        if self.currentFrames[dot] >= dotIntervalFrames[dot]:
                            self.lifetime(win, dot=dot) # Reposition dot at another random location
                    self.dotCoords = np.array(self.dotCoords) # list --> np array
                    # Move normal dots
                    if self.dotCoords[normal_mask][:].size != 0:
                        self.dotCoords[normal_mask, 0] += speedComponents[0]
                        self.dotCoords[normal_mask, 1] += speedComponents[1]
                    # Move aberrant dots 
                    if self.dotCoords[aberrant_mask][:].size != 0:
                        self.dotCoords[aberrant_mask, 0] += aberrantComponents[0]
                        self.dotCoords[aberrant_mask, 1] += aberrantComponents[1]
                    dots.xys = self.dotCoords # Move the dots to their new positions                    
                    dots.draw()                    
                    scotomaMask.draw()
                    win.flip()
                    if self.checkQuitOrPause():
                        return                
            
            # Last bookend
            for f in range(self._numFramesBookend):  
                self.currentFrames += 1
                for dot in range(self.numberOfDots):
                    if self.currentFrames[dot] >= dotIntervalFrames[dot]:
                        self.lifetime(win, dot=dot) # Reposition dot at another random location
                self.dotCoords = np.array(self.dotCoords) # list --> np array
                # Move normal dots
                if self.dotCoords[normal_mask][:].size != 0:
                    self.dotCoords[normal_mask, 0] += speedComponents[0]
                    self.dotCoords[normal_mask, 1] += speedComponents[1]
                # Move aberrant dots 
                if self.dotCoords[aberrant_mask][:].size != 0:
                    self.dotCoords[aberrant_mask, 0] += aberrantComponents[0]
                    self.dotCoords[aberrant_mask, 1] += aberrantComponents[1]
                dots.xys = self.dotCoords # Move the dots to their new positions                    
                dots.draw()                  
                scotomaMask.draw()
                win.flip()
                if self.checkQuitOrPause():
                    return
            
            ## Tail time
            for f in range(self._tailTimeNumFrames):
                dots.draw()
                scotomaMask.draw()
                win.flip()
                if self.checkQuitOrPause():
                    return            
            
            self._stimulusEndLog.append(trialClock.getTime())
            self.sendTTL()
            win.flip();win.flip() #two flips to allow for a pause for TTL writing

            self._numberOfEpochsCompleted += 1            
        self._completed = 1