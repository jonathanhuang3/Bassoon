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
from psychopy.hardware import keyboard
import numpy as np

class GlassPatterns(protocol):
    
    def __init__(self):
        super().__init__()
        #protocol parameters
        self.protocolName = 'GlassPatterns' #Glass Patterns presents a pseudorandom series of squares and circles to a subject and asks for dynamic feedback about the identity of each stimulus. The subject presses 's' to indicate a square and 'c' to indicate a circle.
        self.totalEpochs = 250 #total number of epochs in the stimulus
        self.safety = 165 #if the deltaAngle is greater than this value, then the true angle is likely close to 180 degrees and the subject input the correct angle.
        #optotype parameters
        self.numberOfOptotypes = 20 #number of different optotype levels/sizes to draw from
        self.optotypeColor = [1.0, 1.0, 1.0] #Color of the optotype (in RGB). -1.0 equates to 0 and 1.0 equates to 255 for 8 bit colors.
        self.backgroundColor = [0.0, 0.0, 0.0] #background color of the screen (in RGB). -1.0 equates to 0 and 1.0 equates to 255 for 8 bit colors.
        self.circleRadius_degrees = 0.25 #degrees - radius of the smallest optotype                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        #scotoma parameters
        self.scotomaOpacity = 1.0 #The opacity of the scotomas. 1.0 is fully opaque, 0.0 is fully transparent
        self.scotomaSize = 0.1 #degrees - side lengths of the scotomas, which are squares
        self.scotomaColor = [0.0, 0.0, 0.0] #color of the scotomas (in RGB). -1.0 equates to 0 and 1.0 equates to 255 for 8 bit colors.
        self.scotomaRange = [0.0, 1.0] #scotoma proportion range from lower range to upper range
        #input line parameters
        self.lineSize = [2, 80] #[width, length] of input line in pixels
        self.linePos = [280, -190] #coordinate position of the left end of the line. (0,0) is the center and each quadrant is a 400x300 rectangle
        self.lineColor = [-1.0, -1.0, -1.0] #Color of the input line (in RGB). -1.0 equates to 0 and 1.0 equates to 255 for 8 bit colors.
        self.initialLineOrientation = 0 #degrees - orientation of the input line corresponds to degrees in unit circle. Can range from 0 to 180.
        self.inputBackground = [0.0, 0.0, 0.0] #Color of the square background for the input line (in RGB). -1.0 equates to 0 and 1.0 equates to 255 for 8 bit colors.
        self.backgroundOpacity = 0 #opacity of the background for the input line. 1.0 is fully opaque, 0.0 is fully transparent
        
        self.preTime = 1.0 #seconds - unused for this stimulus
        self.stimTime = 1.0 #seconds - unused for this stimulus
        self.tailTime = 1.0 #seconds - unused for this stimulus
    
    def internalValidation(self):
        '''
        Checks that color values are valid and that scotomaRange is valid.
        '''
        tf = True
        errorMessage = []
        
        #check that scotomaRange is a valid range of proportions
        if self.scotomaRange[0] > 1 or self.scotomaRange[1] > 1:
            tf = False
            errorMessage.append("Value(s) in self.scotomaRange was/are greater than 1. self.scotomaRange is a range of the proportion of the window covered by scotomas and must only contain values between 0 and 1.")
        if self.scotomaRange[0] > self.scotomaRange[1]:
            tf = False
            errorMessage.append("First listed proportion in scotomaRange cannot be greater than the second listed proportion")
            
        #checks color values
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
        self._estimatedTime = self.totalEpochs * 4 #return estimated time for the total stimulus in seconds
        
        return self._estimatedTime
                
    def scatter(self, win, mask, overlapping):
        '''
        This function has two purposes:
            1. randomly select coodinates for each pair and put them into this numpy array: self.optotypePositions
            2. randomly select scotomas on screen to make visible
        '''
        #random.seed(self.randomSeed) #reinitialize the random seed
        optotypePositions = np.zeros((self.numberOfOptotypes, 2))
        for pair in range(self.numberOfOptotypes):
            self.xPos = random.uniform(-win.size[0]/2 + self.circleRadius_pix*2, win.size[0]/2 - self.circleRadius_pix*2)
            self.yPos = random.uniform(-win.size[1]/2 + self.circleRadius_pix*2, win.size[1]/2 - self.circleRadius_pix*2)
            optotypePositions[pair] = [self.xPos, self.yPos]
        
        #check if circles overlap
        protection = 0
        while overlapping and protection < 1000:
            c = 0
            for i, c1 in enumerate(optotypePositions):
                for j, c2 in enumerate(optotypePositions):
                    distance = math.sqrt((c2[0]- c1[0])**2 + (c2[1] - c1[1])**2) #distance between the junction point of one pair and the junction point of another pair
                    if distance < 4*self.circleRadius_pix and i != j: #if the pairs overlap, pick a new random coordinate
                        c += 1  
                        self.xPos = random.uniform(-win.size[0]/2 + self.circleRadius_pix*2, win.size[0]/2 - self.circleRadius_pix*2)
                        self.yPos = random.uniform(-win.size[1]/2 + self.circleRadius_pix*2, win.size[1]/2 - self.circleRadius_pix*2)
                        optotypePositions[j] = [self.xPos, self.yPos]
                    else:
                        continue
            protection += 1
            if c == 0:
                overlapping = False
                for index, coordinate in enumerate(optotypePositions): #the random x and y positions were based of the junction point for each pair. Since pairs are drawn at the center of one circle and not the junction point, this calculates the center coordinates of one circle based on the randomly chosen coordinate for the junction point.
                    self.xPos = optotypePositions[index][0] - (self.circleRadius_pix * math.cos(self._arcAngle))
                    self.yPos = optotypePositions[index][1] - (self.circleRadius_pix * math.cos(self._arcAngle))
                
            if protection == 1000:
                print("Pairs may overlap. Adjust the size and/or density of the pairs.")

        
        self.scotomaDensity = random.randint(int(self.scotomaRange[0]*self.numTotalScotomas), int(self.scotomaRange[1]*self.numTotalScotomas))
        scotomaIndices = random.sample([i for i in range(self.numTotalScotomas)], int(self.scotomaDensity))
        mask[scotomaIndices] = self.scotomaOpacity
        self._scotomaCoverage.append(self.scotomaDensity/self.numTotalScotomas)
        
        return optotypePositions, mask
        
    def run(self, win, informationWin):
        '''
        Executes the Glass Patterns stimulus
        '''
        self._completed = 0 #started but not completed
        self._informationWin = informationWin #tuple, save here so you don't have to pass this as a function parameter every time you use it
        userAngle = None
        self.getFR(win)
        
        stimMonitor = win.monitor

        self.pixPerDegree = self.getPixPerDeg(stimMonitor)
        
        self.circleRadius_pix = self.circleRadius_degrees * self.pixPerDegree
        self.scotomaSize_pix = int(self.scotomaSize*self.pixPerDegree) #maybe a slight rounding error here by using int
        
        #Pause for keystroke if the user wants to manually initiate
        if self.userInitiated:
            self.showInformationText(win, 'Stimulus Information: Glass Patterns \nPress any key to begin')
            event.waitKeys() #wait for key press  
        
        #Data for the stimulus
        self._deltaAngles = [] #the change between the userAngle and the arcAngle
        self._scotomaCoverage = [] #fraction of the screen covered by scotomas. If the scotomas are significantly smaller than the pairs, then the fractional coverage of the window will be approximately equal to the average coverage for the pairs.
        
        instructions = 'Glass Patterns Test for Visual Acuity: \nPress left and right arrows keys to match the line with the angle of the pairs \n\nPress any key to begin'
        self.showInformationText(win, instructions)
        event.waitKeys() #wait for key press
        
        kb = keyboard.Keyboard()
        
        win.color = self.backgroundColor
        win.flip()
        win.flip()
        
        #specify the x and y center coordinates for each check
        xCoordinates = [x - win.size[0]/2 for x in range(-self.scotomaSize_pix, win.size[0]+self.scotomaSize_pix, self.scotomaSize_pix)]
        yCoordinates = [y - win.size[1]/2 for y in range(-self.scotomaSize_pix, win.size[1]+self.scotomaSize_pix, self.scotomaSize_pix)]
        self.numTotalScotomas = len(xCoordinates) * len(yCoordinates)

        sizes = [(self.scotomaSize_pix, self.scotomaSize_pix) for i in range(self.numTotalScotomas)]
        
        scotomaCoordinates = []
        for i in range(len(xCoordinates)):
            for j in range(len(yCoordinates)):
                scotomaCoordinates.append([xCoordinates[i], yCoordinates[j]])

        scotomaMask = visual.ElementArrayStim(
            win,
            nElements = self.numTotalScotomas,
            elementMask="None",
            elementTex = None,
            xys = scotomaCoordinates,
            sizes = sizes,
            colors = self.scotomaColor
            )
        
        random.seed(self.randomSeed) #reinitialize the random seed
        mask = np.zeros((self.numTotalScotomas, 1)) #1 is fully transparent, -1 is fully opaque. Start with a fully transparent mask.
        scotomaMask.opacities = mask #set the first mask
        
        #initialize angle input line
        inputLine = visual.Line(
            win = win,
            units = 'pix',
            lineColor = self.lineColor,
            pos = self.linePos,
            size = self.lineSize,
            ori = 90 - self.initialLineOrientation #psychopy orientation starts at a 90 degree angle like a clock 
            )
        
        inputBackground = visual.Rect(
            win = win,
            fillColor = self.inputBackground,
            pos = [self.linePos[0], self.linePos[1]],
            size = [self.lineSize[1], self.lineSize[1]],
            opacity = self.backgroundOpacity
            )
        
        #assign parameters
        self._arcAngle = math.radians(random.randint(0,180)) #radians -- angle of optotype pairs.
        self._arcAngle_deg = math.degrees(self._arcAngle) #degrees
        optotypePositions, mask = self.scatter(win, mask, overlapping=True)
        scotomaMask.opacities = mask
        epochNum = 0
        testComplete = False
        trialClock = core.Clock() #this will reset every trial
        self._reactionTimes = []
        reactionTimeStart = time.perf_counter()
        newEpochFlag = True

        while not testComplete:
            if newEpochFlag:
                circleOnes = []
                circleTwos = []
                for pair in range(self.numberOfOptotypes):
                    self.xPos = optotypePositions[pair][0]
                    self.yPos = optotypePositions[pair][1]
                    #rotate optotype
                    xChange = 2 * self.circleRadius_pix * math.cos(self._arcAngle) # x' =  R * cos(theta)
                    yChange = 2 * self.circleRadius_pix * math.sin(self._arcAngle) # y' = R * sin(theta)
                    self.xNew = self.xPos + xChange
                    self.yNew = self.yPos + yChange
                    
                    #initialize a pair
                    optotypeCircleOne = visual.Circle(
                        win = win,
                        radius = self.circleRadius_pix,
                        units = 'pix',
                        lineColor = self.optotypeColor,
                        lineWidth = 2,
                        fillColor = self.optotypeColor,
                        pos = (self.xPos, self.yPos)
                        )
                    
                    optotypeCircleTwo = visual.Circle(
                        win = win,
                        radius = self.circleRadius_pix,
                        units = 'pix',
                        lineColor = self.optotypeColor,
                        lineWidth = 2,
                        fillColor = self.optotypeColor,
                        pos = (self.xNew, self.yNew)
                        )
                    
                    circleOnes.append(optotypeCircleOne)
                    circleTwos.append(optotypeCircleTwo)
            
               
                newEpochFlag = False
            
            [x.draw() for x in circleOnes]
            [x.draw() for x in circleTwos]

            
            scotomaMask.draw()
            inputBackground.draw()
            inputLine.draw()
            win.flip()
            
            keyPressed = kb.waitKeys(keyList=['right', 'left', 'q', 'return'], waitRelease=False, clear=False)
            if keyPressed:
                key = [key.value for key in keyPressed]
            if key:
                key = key[0]
            
            if key == 'q':
                break
            elif key == 'right':
                inputLine.ori += 1
                inputLine.draw()
            elif key == 'left':
                inputLine.ori -= 1
                inputLine.draw()
            elif key == 'return': #this key must be pressed only ONCE when the user wants to submit an answer. Holding the enter key will continually submit angles.
                self._reactionTimes.append(time.perf_counter() - reactionTimeStart)
                reactionTimeStart = time.perf_counter()
                userAngle = 90 - inputLine.ori
                
                #correct userAngle to the accurate angle between 0 and 180 degrees
                if userAngle > 180:
                    intermediate = userAngle - int(userAngle/360)*360
                    if intermediate > 180:
                        userAngle = intermediate - 180
                    else:
                        userAngle = intermediate
                elif userAngle < 0:
                    if userAngle < -360:
                        intermediate = userAngle - int(userAngle/360)*360
                        if intermediate < -180:
                            userAngle = intermediate + 360
                        else:
                            userAngle = intermediate + 180
                    else:
                        if userAngle < -180:
                            userAngle = userAngle + 360
                        else:
                            userAngle = userAngle + 180
                
                deltaAngle = abs(self._arcAngle_deg - userAngle)
                
                #the subject might inadvertently move the line to a 0 degree angle if pairs are at a 180 degree angle, leading to an inaccurately large deltaAngle.
                if deltaAngle >= self.safety: 
                    if userAngle < self._arcAngle_deg:
                        deltaAngle = abs((userAngle + 180) - self._arcAngle_deg)
                    elif userAngle > self._arcAngle_deg:
                        deltaAngle = abs((self._arcAngle_deg + 180) - userAngle)
                
                self._deltaAngles.append(deltaAngle)
                
                #reassign parameters
                epochNum += 1
                mask = np.zeros((self.numTotalScotomas, 1)) #1 is fully transparent, -1 is fully opaque. Start with a fully transparent mask.
                self._arcAngle = math.radians(random.randint(0,180)) #radians -- angle of optotype pairs.
                self._arcAngle_deg = math.degrees(self._arcAngle) #degrees
                optotypePositions, mask = self.scatter(win, mask, overlapping=True)
                scotomaMask.opacities = mask
                
                newEpochFlag = True
                

            if epochNum == self.totalEpochs:
                testComplete = 1
            else:
                testComplete = 0
        
        self._completed = 1