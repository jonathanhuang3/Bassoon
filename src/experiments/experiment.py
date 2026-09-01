# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:24:12 2021

@author: mrsco
"""
from psychopy import core, visual, data, event, monitors
from psychopy.visual.windowwarp import Warper
import os
import math
import serial
import json
import re
import platform
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

from bassoonMonitors import save_monitor_gamma

_CALIBRATION_FISH_PATH = Path(__file__).resolve().parent.parent / 'assets' / 'calibration_fish.png'


def _loadEyeLinkCoreGraphics():
    try:
        from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
    except ImportError:
        from psychopy_eyelink_coregraphics import EyeLinkCoreGraphicsPsychoPy
    return EyeLinkCoreGraphicsPsychoPy


class _BassoonEyeLinkGraphics(_loadEyeLinkCoreGraphics()):
    '''EyeLink graphics with visible defaults and GUI event pumping for Bassoon.'''

    def __init__(
        self,
        tracker,
        win,
        gui_root=None,
        cal_target_mode='standard',
        fish_image_path=None,
        fish_width_deg=1.0,
        pix_per_deg=None,
    ):
        self._cal_target_mode = cal_target_mode
        self._fish_image_path = fish_image_path
        self._fish_width_deg = fish_width_deg
        self._pix_per_deg = pix_per_deg
        super().__init__(tracker, win, disableAudio=True)
        self._gui_root = gui_root
        # Built-in defaults are black-on-black until the Host sends colors.
        bg = list(win.color) if hasattr(win.color, '__len__') else [-1, -1, -1]
        self.setCalibrationColors([1, 1, 1], bg)
        if self._cal_target_mode == 'fish':
            fish_path = Path(self._fish_image_path) if self._fish_image_path else None
            if fish_path is None or not fish_path.is_file():
                print('*** Fish calibration image not found; using standard target.')
                self._cal_target_mode = 'standard'
            else:
                self.setTargetType('picture')
                self.setPictureTarget(str(fish_path))
        self.update_cal_target()

    def _pumpGuiEvents(self):
        if self._gui_root is not None:
            try:
                self._gui_root.update_idletasks()
                self._gui_root.update()
            except Exception:
                pass
        try:
            core.wait(0.001, hogCPUInterval=0.001)
        except Exception:
            pass

    def get_input_key(self):
        self._pumpGuiEvents()
        return super().get_input_key()

    def update_cal_target(self):
        super().update_cal_target()
        if self._cal_target_mode == 'fish' and self._calibTar is not None and self._pix_per_deg:
            width_px = self._fish_width_deg * self._pix_per_deg
            nat_w, nat_h = self._calibTar.size
            if nat_h > 0:
                self._calibTar.size = (width_px, width_px * (nat_h / nat_w))

    def draw_cal_target(self, x, y):
        fg = self.getForegroundColor()
        if fg in ('black', 'Black', [-1, -1, -1], (0, 0, 0)):
            self.setCalibrationColors([1, 1, 1], self.getBackgroundColor())
            self.update_cal_target()
        print('--> EyeLink calibration target at ({x}, {y})'.format(
            x=int(x), y=int(y)))
        super().draw_cal_target(x, y)


class experiment():
    def __init__(self):
        self.protocolList = []

        self.experimentDate = datetime.now().strftime("%D %H:%M:%S")
        self.activated = False

        self.allowGUI = True
        self.screen = 0
        self.fullscr = False
        self.backgroundColor = [-1, -1, -1] #doesn't do much, more or less obsolete because it's hardly seen
        self.units = 'pix'
        self.allowStencil = True
        self.stimMonitor = 'testMonitor'
        self.gamma = 2.0 #Default gamma value, will be updated from calibration.

        self.useInformationMonitor = False
        self.informationMonitor = 'testMonitor'
        self.informationWin = None #will become a process. Must destroy to pickle the experiment object
        self.informationScreen = 0
        self.informationFullScreen = False

        self.estimatedTotalTime = 0

        self.loggedStimuli = []

        self.userInitiated = False #If True, the user will have to manually start each stimulus. Can also set this property manually for each stimulus
        self.angleOffset = 0.0 #deg - offset for directional stimuli

        self.writeTTL = 'None' #can be 'None', 'Pulse', 'Sustained'
        self.ttlBookmarks = False #used for sustained mode only to send stereotyped bookmark patterns before each stimulus
        self.ttlPort = ''
        self.ttlPortOpen = False #tracks whether the TTL port is open or not (not whether it's ON or OFF, but if the port itself is open and ready for commands)

        self.warpFileName = 'Warp File Location' #must be .data
        self.useFBO = False

        self.FR = 0 #frame rate of the stimulus window
        
        self.recompileExperiment = False  # option that is used by self.saveExperiment()

        self.timingReport = False

        # EyeLink (optional). Off by default so existing experiments are unchanged.
        self.useEyeLink = False 
        self.eyeLinkDummy = False # True = no Host PC; pylink dummy tracker
        self.eyeLinkIP = '100.1.1.1' # Host PC address on the dedicated Ethernet link
        self.eyeLinkEDF = 'BASS.EDF' # Host filename, 8 chars + .EDF
        self.eyeLinkEDFDir = '' # folder on the Bassoon PC for downloaded EDFs; empty means current working directory
        self.eyeLinkCalTarget = 'standard' # 'standard' (circles) or 'fish' (1 deg fish icon)
        self.eyeLinkEdf2AscPath = '' # optional full path to edf2asc.exe; empty means auto-detect
        self.eyeLinkWriteAsc = True # convert downloaded EDF to ASC after each EyeLink session
        self._elTracker = None
        self._eyeLinkSessionStamp = None
        self._eyeLinkCalibrationJsonPath = None
        
        #Load previously saved experimental settings from configOptions.json
        if Path('configOptions.json').is_file():
            with open('configOptions.json') as f:
                try:
                    configOptions = json.load(f)
                    #stimWindow
                    self.screen = configOptions['stimWindow']['screen']
                    self.fullscr = configOptions['stimWindow']['fullscr']
                    self.stimMonitor = configOptions['stimWindow']['stimMonitor']
                    try:
                        self.gamma = configOptions['stimWindow']['gamma']
                    except Exception as e:
                        print('*** Could not load gamma from config. Please calibrate to set new gamma.')
                    #infoWindow
                    self.useInformationMonitor = configOptions['infoWindow']['useInformationMonitor']
                    self.informationMonitor = configOptions['infoWindow']['informationMonitor']
                    self.informationFullScreen = configOptions['infoWindow']['informationFullScreen']
                    self.informationScreen = configOptions['infoWindow']['informationScreen']
                    #experiment
                    self.userInitiated = configOptions['experiment']['userInitiated']
                    self.angleOffset = float(configOptions['experiment']['angleOffset'])
                    self.writeTTL = configOptions['experiment']['writeTTL']
                    if isinstance(self.writeTTL, bool):
                        self.writeTTL = "None"
                    portInfo = configOptions['experiment']['ttlPort']
                    if self.writeTTL != "None":
                        self.establishPort(portInfo)
                    
                    self.useFBO = configOptions['experiment']['useFBO']
                    self.warpFileName = configOptions['experiment']['warpFileName']
                    
                    #add new options here so that they don't mess up old file formats
                    self.ttlBookmarks = configOptions['experiment']['ttlBookmarks']
                    self.timingReport = configOptions['experiment']['timingReport']
                    self.recompileExperiment = configOptions['experiment']['recompileExperiment']
                    self.useEyeLink = configOptions['experiment'].get('useEyeLink', False)
                    self.eyeLinkDummy = configOptions['experiment'].get('eyeLinkDummy', False)
                    self.eyeLinkIP = configOptions['experiment'].get('eyeLinkIP', '100.1.1.1')
                    self.eyeLinkEDF = configOptions['experiment'].get('eyeLinkEDF', 'BASS.EDF')
                    self.eyeLinkEDFDir = configOptions['experiment'].get('eyeLinkEDFDir', '')
                    self.eyeLinkCalTarget = configOptions['experiment'].get('eyeLinkCalTarget', 'standard')
                    self.eyeLinkEdf2AscPath = configOptions['experiment'].get('eyeLinkEdf2AscPath', '')
                    self.eyeLinkWriteAsc = configOptions['experiment'].get('eyeLinkWriteAsc', True)
                except:
                    print('*** Could not load all configuration settings from src/configOptions.json. Manually apply settings in the Options menu.')

        self.ensureEyeLinkDefaults()

    def ensureEyeLinkDefaults(self):
        '''Back-fill EyeLink settings on older experiment objects or configs.'''
        defaults = {
            'useEyeLink': False,
            'eyeLinkDummy': False,
            'eyeLinkIP': '100.1.1.1',
            'eyeLinkEDF': 'BASS.EDF',
            'eyeLinkEDFDir': '',
            'eyeLinkCalTarget': 'standard',
            'eyeLinkEdf2AscPath': '',
            'eyeLinkWriteAsc': True,
        }
        for key, value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        if getattr(self, 'eyeLinkCalTarget', 'standard') not in ('standard', 'fish'):
            self.eyeLinkCalTarget = 'standard'

    def getPixPerDeg(self):
        '''Pixels per visual degree for the stimulus monitor.'''
        mon = monitors.Monitor(self.stimMonitor)
        eyeDistance = mon.getDistance()
        numPixelsWide = mon.currentCalib['sizePix'][0]
        cmWide = mon.currentCalib['width']
        totalVisualDegrees = 2 * math.degrees(math.atan((cmWide / 2) / eyeDistance))
        return numPixelsWide / totalVisualDegrees

    def addProtocol(self, newProtocol):
        '''
        Add a protocol to the experiment
        '''
        self.protocolList.append((newProtocol.protocolName, newProtocol))

        estimatedTotalTime = 0
        for tup in self.protocolList:
            estimatedTime_protocol = tup[1].estimateTime() #each protocol object should have a method called self.estimateTime
            estimatedTotalTime += estimatedTime_protocol

        self.estimatedTotalTime = estimatedTotalTime #store total estimated time in self.estimatedTotalTime


    def establishPort(self, portInfo, fromSave = False):
        '''
        This function is used to open the COM/USB/serial/TTL port that is used for timing signals. It is critical that this port persists/remains open so long as the app is running/a port has been set. If the experiment object is deleted, the port attribute is deleted, or the port itself is closed, the voltage will revert back to its default state, making it difficult to control. This messes up timing protocols, so instead, keep the same port open for the duration of the experiment(s). Note: this scheme of keeping the port continuously open is an update as of 6/7/2024

        The port is purposefully closed when an experiment is saved, however, because it cannot be serialized (I think). This function is then called again after saving to re-establish the port.
        
        Inputs:
            - portInfo = the information about the selected port that is returned by serial.tools.list_ports.comports()
            - fromSave = boolean value that indicates whether this function is being called from the saveExperiment() function in main.py
        '''
        
        if self.ttlPortOpen: #check if there is an open port. If so, close it so that you can reconnect or connect to a different port
            print('--> Closing old TTL port')
            self.ttlPortOpen = False
            self.portObj.close() #close the open port if one is open that has a DIFFERENT name than the new one
        
        if self.writeTTL != ('Sustained' or 'Pulse'):
            return #just in case, make sure this function is only called when the writeTTL option is set to Sustained
        
        if portInfo == 'No Available Ports' or portInfo == '':
            return #check to make sure a real port has been selected
        
        #get port name
        if fromSave:
            portName = portInfo
        else:   
            space_index = portInfo.find(' ')
            if space_index == -1:
                portName = portInfo
            else:    
                portName = portInfo[:portInfo.find(' ')] #PARSING FOR HOW PORT NAME IS DETERMINED - may need to be manually adjusted based on operating system
            
        self.ttlPort = portName
        
        #if self.ttlPort is blank, then self.writeTTL must be None
        if self.ttlPort.strip() == '':
            self.writeTTL = 'None'
            return
        
        try:
            if self.writeTTL == 'Sustained':
                self.portObj = serial.Serial(portName)
                self.portObj.rts = True #set the RTS value to True, moving the voltage to 0
            elif self.writeTTL == 'Pulse':
                self.portObj = serial.Serial(portName, 4000000)   
            self.ttlPortOpen = True
            print('--> New TTL port has been opened')
        except serial.serialutil.SerialException:
            print('***IMPORTANT: It looks like the port you are trying to access is already in use. It may be open in a different program, or it may have never been closed by a previous instance of Bassoon. It is recommended that you close python and restart Bassoon to release the port')
            self.writeTTL = 'None'
            self.ttlPort = ''
        except:
            print('***Could not open or set the serial port called ', portName, '. Ensure you\'ve selected the proper port and try again. (If this error persists, see the experiment.establishPort() method. You may need to change the parsing for how the port name is determined depending on your operating system).')
            self.writeTTL = 'None'
            self.ttlPort = ''
        
        #The port should now stay open for as long as the experiment persists. If a new experiment is loaded in, the port should be reset and reopened. If the experiment is saved, the port will be temporarily closed, deleted, and then reestablished and opened
        return


    def _sanitizeEdfName(self, name):
        '''
        EyeLink Host PCs require an 8.3-style EDF name (up to 8 alphanumeric characters + .EDF).
        '''
        stem = Path(str(name)).stem.upper()
        stem = ''.join(ch for ch in stem if ch.isalnum())
        if stem == '':
            stem = 'BASS'
        return stem[:8] + '.EDF'


    def _resolveEyeLinkSaveDir(self):
        '''
        Folder on this computer where downloaded EDF files are written.
        Uses Options → EDF Save Folder when it is a valid path; otherwise the current working directory.
        '''
        requested = str(self.eyeLinkEDFDir).strip()
        if requested == '':
            saveDir = Path.cwd()
        else:
            saveDir = Path(requested).expanduser()
            try:
                saveDir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print('*** Could not use EDF save folder', saveDir, '(' + str(e) + '). Using the current working directory instead.')
                saveDir = Path.cwd()
            if not saveDir.is_dir():
                print('*** EDF save folder is not a directory. Using the current working directory instead.')
                saveDir = Path.cwd()
        return saveDir


    def _eyeLinkSessionBasename(self):
        '''Shared base filename for the downloaded EDF and its calibration JSON.'''
        stamp = self._eyeLinkSessionStamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        return Path(self.eyeLinkEDF).stem + '_' + stamp


    def _eyeLinkLocalEdfPath(self):
        return self._resolveEyeLinkSaveDir() / (self._eyeLinkSessionBasename() + '.edf')


    def _eyeLinkLocalCalibrationJsonPath(self):
        return self._resolveEyeLinkSaveDir() / (self._eyeLinkSessionBasename() + '_calibration.json')


    def _eyeLinkLocalAscPath(self, edf_path=None):
        if edf_path is None:
            edf_path = self._eyeLinkLocalEdfPath()
        else:
            edf_path = Path(edf_path)
        return edf_path.with_suffix('.asc')


    def _findEdf2AscExecutable(self):
        '''Locate SR Research edf2asc, using an explicit path or common install locations.'''
        requested = str(getattr(self, 'eyeLinkEdf2AscPath', '')).strip()
        if requested:
            path = Path(requested).expanduser()
            if path.is_file():
                return path
            print('*** EDF2ASC path is not a file:', path)

        if os.name == 'nt':
            for env_name in ('ProgramFiles(x86)', 'ProgramFiles'):
                base = os.environ.get(env_name)
                if not base:
                    continue
                candidates = [
                    Path(base) / 'SR Research' / 'EyeLink' / 'bin' / 'edf2asc.exe',
                    Path(base) / 'SR Research' / 'EyeLink' / 'EDF2ASC' / 'edf2asc.exe',
                    Path(base) / 'SR Research' / 'EyeLink' / 'edf2asc.exe',
                    Path(base) / 'EyeLink' / 'edf2asc.exe',
                ]
                for candidate in candidates:
                    if candidate.is_file():
                        return candidate

        for name in ('edf2asc', 'EDF2ASC', 'edf2asc.exe', 'EDF2ASC.exe'):
            found = shutil.which(name)
            if found:
                return Path(found)
        return None


    def _convertEyeLinkEdfToAsc(self, edf_path):
        '''Convert a downloaded EDF to ASC using the EyeLink Developers Kit edf2asc tool.'''
        if not getattr(self, 'eyeLinkWriteAsc', True):
            return None

        edf_path = Path(edf_path)
        if not edf_path.is_file():
            return None

        edf2asc = self._findEdf2AscExecutable()
        if edf2asc is None:
            print('*** EDF2ASC not found. Install the EyeLink Developers Kit or set eyeLinkEdf2AscPath in configOptions.json.')
            return None

        asc_path = self._eyeLinkLocalAscPath(edf_path)
        # Default output includes samples, events, and messages. Do not use -e
        # (events only) or -m (parsed as -miss, not messages).
        command = [str(edf2asc), '-y', str(edf_path)]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=str(edf2asc.parent),
            )
        except Exception as e:
            print('*** Could not run EDF2ASC (' + str(e) + ').')
            return None

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            print('*** EDF2ASC failed (' + str(result.returncode) + ').', detail)
            return None
        if not asc_path.is_file():
            print('*** EDF2ASC finished but ASC file was not created:', asc_path)
            return None

        print('--> Wrote ASC to', asc_path)
        return asc_path


    def _updateEyeLinkCalibrationJsonAscPath(self, asc_path):
        if not self._eyeLinkCalibrationJsonPath:
            return

        json_path = Path(self._eyeLinkCalibrationJsonPath)
        if not json_path.is_file():
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            record['localAscFile'] = Path(asc_path).name
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2)
        except Exception as e:
            print('*** Could not update calibration JSON with ASC filename (' + str(e) + ').')


    def sendEyeLinkMessage(self, text):
        '''
        Send a timestamped message to the open EDF. No-op if EyeLink is not connected.
        '''
        if self._elTracker is None:
            return
        try:
            self._elTracker.sendMessage(str(text)[:120])
        except Exception:
            print('***WARNING: EyeLink message failed:', text)


    def _decodeEyeLinkCalibrationResult(self, resultCode):
        '''Map pylink getCalibrationResult() codes to a readable label.'''
        labels = {
            0: 'OK',
            1: 'POOR_CALIBRATION_OR_HIGH_VALIDATION_ERROR',
            -1: 'FAILED',
            27: 'ABORTED',
            1000: 'NO_REPLY',
        }
        return labels.get(resultCode, 'UNKNOWN')


    def _decodeEyeUsed(self, eyeCode):
        '''Map pylink getEyeUsed() codes to a readable label.'''
        labels = {
            0: 'LEFT',
            1: 'RIGHT',
            2: 'BINOCULAR',
            -1: 'NONE',
        }
        return labels.get(eyeCode, 'UNKNOWN')


    def _parseEyeLinkEyeMetrics(self, values):
        '''
        Parse one eye's numeric fields from an EyeLink validation_result message.
        EyeLink typically reports average error, max error, and a third auxiliary value.
        '''
        if not values:
            return {}

        metrics = {}
        if len(values) >= 1:
            metrics['averageErrorDegrees'] = values[0]
        if len(values) >= 2:
            metrics['maxErrorDegrees'] = values[1]
        if len(values) >= 3:
            metrics['auxValue'] = values[2]
        return metrics


    def _parseEyeLinkCalibrationMessage(self, message):
        '''
        Extract per-eye and summary error values in degrees from EyeLink cal/validation messages.
        Returns a dict with parsed values when found.
        '''
        parsed = {}
        if not message:
            return parsed

        resultMatch = re.search(
            r'(?P<type>validation_result|calibration_result)\s*:\s*(?P<values>[-0-9.\s]+)',
            message,
            re.IGNORECASE,
        )
        if resultMatch:
            resultType = resultMatch.group('type').lower()
            values = [
                float(v) for v in resultMatch.group('values').split()
                if v.strip() != ''
            ]
            parsed['resultType'] = resultType

            if len(values) >= 6:
                parsed['eyes'] = {
                    'left': self._parseEyeLinkEyeMetrics(values[0:3]),
                    'right': self._parseEyeLinkEyeMetrics(values[3:6]),
                }
            elif len(values) >= 3:
                parsed['eyes'] = {
                    'left': self._parseEyeLinkEyeMetrics(values[0:3]),
                }

            eyeMetrics = parsed.get('eyes', {})
            avgErrors = [
                eye['averageErrorDegrees']
                for eye in eyeMetrics.values()
                if 'averageErrorDegrees' in eye
            ]
            maxErrors = [
                eye['maxErrorDegrees']
                for eye in eyeMetrics.values()
                if 'maxErrorDegrees' in eye and eye['maxErrorDegrees'] > 0
            ]
            if avgErrors:
                parsed['averageErrorDegrees'] = sum(avgErrors) / len(avgErrors)
            if maxErrors:
                parsed['maxErrorDegrees'] = max(maxErrors)
            elif avgErrors:
                parsed['maxErrorDegrees'] = max(avgErrors)

            return parsed

        avgMatch = re.search(
            r'(?:average|avg\.?)\s+error[^0-9\-]*([0-9]*\.?[0-9]+)\s*(?:deg|degrees?)',
            message,
            re.IGNORECASE,
        )
        if avgMatch:
            parsed['averageErrorDegrees'] = float(avgMatch.group(1))

        maxMatch = re.search(
            r'max(?:imum)?\.?\s+error[^0-9\-]*([0-9]*\.?[0-9]+)\s*(?:deg|degrees?)',
            message,
            re.IGNORECASE,
        )
        if maxMatch:
            parsed['maxErrorDegrees'] = float(maxMatch.group(1))

        return parsed


    _VALIDATE_POINT_RE = re.compile(
        r'VALIDATE\s+'
        r'(?:(?:LR|[LR])\s+)?'
        r'(?:\d+POINT\s+)?'
        r'POINT\s+(?P<point>\d+)\s+'
        r'(?P<eye>LEFT|RIGHT)\s+'
        r'at\s+(?P<x>\d+),(?P<y>\d+)\s+'
        r'OFFSET\s+(?P<error>[-\d.]+)\s+deg\.'
        r'(?:\s+(?P<pix_x>[-\d.]+),(?P<pix_y>[-\d.]+)\s+pix\.)?',
        re.IGNORECASE,
    )
    _CAL_VALIDATION_SUMMARY_RE = re.compile(
        r'!CAL VALIDATION\s+(?P<model>HV\d+|H\d+\w*)\s+(?P<eye_code>[LR]+)\s+(?P<eye>LEFT|RIGHT)\s+'
        r'(?P<status>\w+)\s+ERROR\s+(?P<avg>[-\d.]+)\s+avg\.\s+(?P<max>[-\d.]+)\s+max',
        re.IGNORECASE,
    )


    def _parseValidationPointsFromEdfText(self, edfText):
        '''
        Parse per-point validation errors from EyeLink message text embedded in an EDF.
        Returns validation point lists grouped by eye and any validation summaries found.
        '''
        validationPoints = {'left': [], 'right': []}
        validationSummaries = {'left': [], 'right': []}
        calibrationModels = []

        for match in self._VALIDATE_POINT_RE.finditer(edfText):
            eyeKey = match.group('eye').lower()
            point = {
                'point': int(match.group('point')),
                'x': int(match.group('x')),
                'y': int(match.group('y')),
                'errorDegrees': float(match.group('error')),
            }
            if match.group('pix_x') is not None and match.group('pix_y') is not None:
                point['offsetPixels'] = {
                    'x': float(match.group('pix_x')),
                    'y': float(match.group('pix_y')),
                }
            validationPoints.setdefault(eyeKey, []).append(point)

        for eyeKey in validationPoints:
            validationPoints[eyeKey].sort(key=lambda item: item['point'])

        for match in self._CAL_VALIDATION_SUMMARY_RE.finditer(edfText):
            eyeKey = match.group('eye').lower()
            model = match.group('model').upper()
            if model not in calibrationModels:
                calibrationModels.append(model)
            validationSummaries.setdefault(eyeKey, []).append({
                'calibrationModel': model,
                'status': match.group('status').upper(),
                'averageErrorDegrees': float(match.group('avg')),
                'maxErrorDegrees': float(match.group('max')),
            })

        parsed = {}
        if any(validationPoints.values()):
            parsed['validationPoints'] = validationPoints
        if validationSummaries:
            parsed['validationSummaries'] = validationSummaries
        if calibrationModels:
            parsed['calibrationModels'] = calibrationModels
            parsed['calibrationModel'] = calibrationModels[-1]
        return parsed


    def _parseValidationPointsFromEdf(self, edfPath):
        '''Read an EDF file and extract per-point validation data from embedded messages.'''
        edfPath = Path(edfPath)
        if not edfPath.is_file():
            return {}

        try:
            edfText = edfPath.read_bytes().decode('latin-1', errors='ignore')
        except Exception as e:
            print('*** Could not read EDF for validation points (' + str(e) + ').')
            return {}

        return self._parseValidationPointsFromEdfText(edfText)


    def _printEyeLinkValidationPoints(self, validation_points):
        if not validation_points:
            return

        for eye_name, points in validation_points.items():
            for point in points:
                print(
                    '    {eye} point {n}: {error:.2f} deg at ({x}, {y})'.format(
                        eye=eye_name,
                        n=point['point'],
                        error=point['errorDegrees'],
                        x=point['x'],
                        y=point['y'],
                    )
                )


    def _updateEyeLinkCalibrationJsonFromEdf(self, edfPath):
        '''Merge per-point validation data from the downloaded EDF into the session JSON file.'''
        if not self._eyeLinkCalibrationJsonPath:
            return None

        jsonPath = Path(self._eyeLinkCalibrationJsonPath)
        if not jsonPath.is_file():
            return None

        pointData = self._parseValidationPointsFromEdf(edfPath)
        if not pointData:
            return jsonPath

        try:
            with open(jsonPath, 'r', encoding='utf-8') as f:
                record = json.load(f)
            record.update(pointData)
            record['validationPointsSource'] = 'edf'
            record['edfParsedForValidationPoints'] = True
            with open(jsonPath, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2)
            print('--> Updated EyeLink calibration JSON with per-point validation data')
            if 'validationPoints' in pointData:
                self._printEyeLinkValidationPoints(pointData['validationPoints'])
            return jsonPath
        except Exception as e:
            print('*** Could not update EyeLink calibration JSON from EDF (' + str(e) + ').')
            return None


    def _saveEyeLinkCalibrationJson(self, edfName, scn_w, scn_h):
        '''
        Save the last EyeLink calibration/validation result to a JSON file in the EDF save folder.
        '''
        if self._elTracker is None:
            return None

        try:
            resultCode = self._elTracker.getCalibrationResult()
            message = str(self._elTracker.getCalibrationMessage()).strip()
        except Exception as e:
            print('*** Could not read EyeLink calibration results (' + str(e) + ').')
            return None

        resultLabel = self._decodeEyeLinkCalibrationResult(resultCode)
        parsed = self._parseEyeLinkCalibrationMessage(message)
        try:
            eyeUsedCode = self._elTracker.getEyeUsed()
        except Exception:
            eyeUsedCode = -1
        eyeRecording = self._decodeEyeUsed(eyeUsedCode)

        jsonName = self._eyeLinkLocalCalibrationJsonPath()
        localEdfName = self._eyeLinkLocalEdfPath().name
        localAscName = self._eyeLinkLocalAscPath().name

        record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'edfName': edfName,
            'localEdfFile': localEdfName,
            'localAscFile': localAscName,
            'hostIP': self.eyeLinkIP,
            'dummyMode': self.eyeLinkDummy,
            'eyeRecording': eyeRecording,
            'eyeUsedCode': eyeUsedCode,
            'display': {
                'width': int(scn_w),
                'height': int(scn_h),
                'screen': self.screen,
                'fullscreen': self.fullscr,
            },
            'resultCode': resultCode,
            'resultLabel': resultLabel,
            'message': message,
            'success': resultCode == 0 or 'averageErrorDegrees' in parsed,
        }
        record.update(parsed)

        try:
            with open(jsonName, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2)
            self._eyeLinkCalibrationJsonPath = jsonName
            print('--> EyeLink calibration/validation saved to', jsonName)
            print('    Paired EDF will download as', localEdfName)
            if message:
                print('    ' + message)
            elif resultLabel == 'NO_REPLY':
                print('    No calibration/validation result was returned. Run validation (V) before exiting setup.')
            if 'eyes' in parsed:
                for eyeName, metrics in parsed['eyes'].items():
                    if 'averageErrorDegrees' in metrics:
                        print('    {eye} eye average error: {v:.2f} deg'.format(
                            eye=eyeName, v=metrics['averageErrorDegrees']))
            if eyeRecording != 'UNKNOWN':
                print('    Eye recording mode:', eyeRecording)
            self.sendEyeLinkMessage(
                '!V TRIAL_VAR cal_result {label}'.format(label=resultLabel.replace(' ', '_'))
            )
            self.sendEyeLinkMessage(
                '!V TRIAL_VAR eye_recording {mode}'.format(mode=eyeRecording.replace(' ', '_'))
            )
            if 'averageErrorDegrees' in parsed:
                self.sendEyeLinkMessage(
                    '!V TRIAL_VAR cal_avg_error_deg {v}'.format(v=parsed['averageErrorDegrees'])
                )
            if 'maxErrorDegrees' in parsed:
                self.sendEyeLinkMessage(
                    '!V TRIAL_VAR cal_max_error_deg {v}'.format(v=parsed['maxErrorDegrees'])
                )
            if 'eyes' in parsed:
                for eyeName, metrics in parsed['eyes'].items():
                    if 'averageErrorDegrees' in metrics:
                        self.sendEyeLinkMessage(
                            '!V TRIAL_VAR cal_{eye}_avg_error_deg {v}'.format(
                                eye=eyeName, v=metrics['averageErrorDegrees'])
                        )
                    if 'maxErrorDegrees' in metrics and metrics['maxErrorDegrees'] > 0:
                        self.sendEyeLinkMessage(
                            '!V TRIAL_VAR cal_{eye}_max_error_deg {v}'.format(
                                eye=eyeName, v=metrics['maxErrorDegrees'])
                        )
            return jsonName
        except Exception as e:
            print('*** Could not save EyeLink calibration JSON (' + str(e) + ').')
            return None


    def _hideBassoonForEyeLinkSetup(self, gui_root):
        '''Hide the Bassoon Tk window so EyeLink calibration can draw on the stimulus screen.'''
        if gui_root is None:
            return
        try:
            gui_root.withdraw()
            gui_root.update_idletasks()
            gui_root.update()
        except Exception:
            pass

    def _focusStimulusWindow(self):
        '''Bring the PsychoPy stimulus window to the front before EyeLink setup.'''
        try:
            self.win.winHandle.activate()
        except Exception:
            pass
        try:
            self.win.winHandle.set_visible(True)
        except Exception:
            pass
        if platform.system() == 'Windows':
            try:
                import ctypes
                hwnd = self.win.winHandle._hwnd
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass

    def startEyeLink(self, gui_root=None):
        '''
        Connect to EyeLink, open an EDF, optionally calibrate, and start recording.
        Failures print to the console and leave _elTracker as None so stimuli still run.
        '''
        self._elTracker = None
        self._eyeLinkSessionStamp = None
        self._eyeLinkCalibrationJsonPath = None
        if not self.useEyeLink:
            return

        try:
            import pylink
        except ImportError:
            print('*** EyeLink was enabled but pylink is not installed. Install sr-research-pylink and the EyeLink Developers Kit. Stimuli will run without the tracker.')
            return

        edfName = self._sanitizeEdfName(self.eyeLinkEDF)
        self.eyeLinkEDF = edfName
        linkAddress = None if self.eyeLinkDummy else self.eyeLinkIP

        try:
            print('--> Connecting to EyeLink at', 'dummy' if linkAddress is None else linkAddress)
            self._elTracker = pylink.EyeLink(linkAddress)
            self._eyeLinkSessionStamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._elTracker.openDataFile(edfName)
            self._elTracker.sendCommand("add_file_preamble_text 'RECORDED BY BASSOON'")

            scn_w, scn_h = self.win.size
            self._elTracker.sendCommand(
                'screen_pixel_coords = 0 0 {w} {h}'.format(w=scn_w - 1, h=scn_h - 1)
            )
            self.sendEyeLinkMessage(
                'DISPLAY_COORDS 0 0 {w} {h}'.format(w=scn_w - 1, h=scn_h - 1)
            )

            if not self.eyeLinkDummy:
                try:
                    # Targets are drawn in this PsychoPy window (stimulus PC), not on the Host monitor.
                    self.win.color = [-1, -1, -1]
                    self.win.flip()
                    genv = _BassoonEyeLinkGraphics(
                        self._elTracker,
                        self.win,
                        gui_root=gui_root,
                        cal_target_mode=self.eyeLinkCalTarget,
                        fish_image_path=_CALIBRATION_FISH_PATH,
                        fish_width_deg=1.0,
                        pix_per_deg=self.getPixPerDeg(),
                    )
                    pylink.openGraphicsEx(genv)
                    cal_label = 'fish (~1° wide)' if self.eyeLinkCalTarget == 'fish' else 'dots'
                    print('--> EyeLink setup ready.')
                    print('    Calibration targets ({t}) appear on the STIMULUS monitor (PsychoPy window), not the Host PC.'.format(
                        t=cal_label))
                    print('    Window size: {w} x {h}  |  screen #: {s}  |  fullscreen: {f}'.format(
                        w=int(scn_w), h=int(scn_h), s=self.screen, f=self.fullscr))
                    print('    On Host PC: Enter = camera setup, C = calibrate, V = validate, Enter/Esc = exit setup.')
                    if not self.fullscr:
                        print('*** TIP: Turn on Full Screen in Options so calibration targets are not hidden behind Bassoon.')
                    self._hideBassoonForEyeLinkSetup(gui_root)
                    self._focusStimulusWindow()
                    self._elTracker.doTrackerSetup()
                    self._focusStimulusWindow()
                    self.win.flip()
                    self._saveEyeLinkCalibrationJson(edfName, scn_w, scn_h)
                except Exception as calErr:
                    print('*** EyeLink connected, but calibration graphics failed (' + str(calErr) + ').')
                    print('*** Install psychopy-eyelink-coregraphics (or place EyeLinkCoreGraphicsPsychoPy.py on PYTHONPATH).')
                    print('*** Recording will continue without doTrackerSetup().')

            self._elTracker.setOfflineMode()
            pylink.pumpDelay(100)
            recErr = self._elTracker.startRecording(1, 1, 1, 1)
            if recErr:
                raise RuntimeError('startRecording returned ' + str(recErr))
            pylink.pumpDelay(100) 
            self.sendEyeLinkMessage('BASSOON_EXPERIMENT_START')
            print('--> EyeLink recording started. EDF on Host:', edfName)
        except Exception as e:
            print('*** Could not start EyeLink (' + str(e) + '). Stimuli will run without the tracker.')
            try:
                if self._elTracker is not None:
                    self._elTracker.close()
            except Exception:
                pass
            self._elTracker = None


    def stopEyeLink(self):
        '''
        Stop recording, close the EDF, and download it to the folder chosen in Options (or the working directory if that folder is blank).
        Safe to call if EyeLink never started.
        '''
        if self._elTracker is None:
            return

        try:
            import pylink
        except ImportError:
            pylink = None

        try:
            self.sendEyeLinkMessage('BASSOON_EXPERIMENT_END')
            try:
                self._elTracker.stopRecording()
            except Exception:
                pass
            try:
                self._elTracker.setOfflineMode()
            except Exception:
                pass
            try:
                self._elTracker.closeDataFile()
            except Exception:
                pass

            if not self.eyeLinkDummy:
                localName = self._eyeLinkLocalEdfPath()
                try:
                    print('--> Downloading EDF to', localName)
                    self._elTracker.receiveDataFile(self.eyeLinkEDF, str(localName))
                    self._updateEyeLinkCalibrationJsonFromEdf(localName)
                    asc_path = self._convertEyeLinkEdfToAsc(localName)
                    if asc_path is not None:
                        self._updateEyeLinkCalibrationJsonAscPath(asc_path)
                except Exception as e:
                    print('*** EyeLink recording finished, but the EDF could not be downloaded (' + str(e) + '). Copy it from the Host PC if needed.')
            try:
                self._elTracker.close()
            except Exception:
                pass
            if pylink is not None:
                try:
                    pylink.closeGraphics()
                except Exception:
                    pass
            print('--> EyeLink disconnected')
        finally:
            self._elTracker = None
            self._eyeLinkSessionStamp = None
            self._eyeLinkCalibrationJsonPath = None



    def activate(self, gui_root=None):
        '''
        Begin the experiment
        '''
        # Tkinter (Bassoon GUI) and PsychoPy both use GUI event loops; disable PsychoPy
        # GUI integration during EyeLink calibration so cal targets can redraw on the stimulus window.
        allowGui = self.allowGUI and not self.useEyeLink

        try:
            save_monitor_gamma(self.stimMonitor, self.gamma)
        except Exception as e:
            print('*** Could not save gamma to monitor profile', self.stimMonitor, '(' + str(e) + ').')

        self.win = visual.Window(
                    allowGUI = allowGui,
                    monitor = self.stimMonitor,
                    gamma = self.gamma,
                    screen = self.screen,
                    fullscr = self.fullscr,
                    color = self.backgroundColor,
                    units = self.units,
                    useFBO = self.useFBO,
                    allowStencil = self.allowStencil,
                    checkTiming = not self.useEyeLink,
                    )

        # When EyeLink is enabled, skip frame-rate measurement until protocols run.
        # PsychoPy otherwise shows "Attempting to measure frame rate..." during Window().
        if not self.useEyeLink:
            self.FR = self.win.getActualFrameRate() #log the frame rate of the stimulus window
        else:
            self.FR = 0

        #set a warper if you want to morph the stimulus
        if self.useFBO:
            warper = Warper(
                self.win,
                warp = 'warpfile',
                warpfile = self.warpFileName
                )

        #if the user would like to use a second screen to display stimulus information then initialize that screen here
        #the flips to this second window must be called in the stimulus protocol itself
        if self.useInformationMonitor:
            self.informationWin = visual.Window(
                        allowGUI = allowGui,
                        monitor = self.informationMonitor,
                        screen = self.informationScreen,
                        color = self.backgroundColor,
                        fullscr = self.informationFullScreen,
                        units = self.units,
                        checkTiming = not self.useEyeLink,
                        )


        self.activated = True
        self.loggedStimuli = [] #always resets on a new run
        self.startEyeLink(gui_root=gui_root)
        try:
            self._runProtocolLoop()
        finally:
            self.stopEyeLink()
            self.win.close()
            if self.useInformationMonitor:
                self.informationWin.close()


    def _runProtocolLoop(self):
        '''
        Play each protocol in order. Split out of activate() so EyeLink is always stopped in a finally block.
        '''
        for i, p in enumerate(self.protocolList):
            name = p[0] #note: p is not a deep copy, so the pointer in memory is to the same location as the protocol in self.protocolList and app.experiment.protocolList
            suffix = p[1].suffix
            
            if suffix == '_' or suffix.strip() == '':
                displayName = name
            else:
                displayName = name + suffix

            print('!!! Running Protocol Number ' + str(i+1) + ' of ' +  str(len(self.protocolList)) + ', with name ' + displayName)
            p = p[1] #the protocol object is the second one in the tuple

            #assign relevant experiment properties to the protocol
            p._timingReport = self.timingReport
            if hasattr(p, '_angleOffset'):
                p._angleOffset = self.angleOffset

            p.writeTTL = self.writeTTL #set the TTL write mode (inherits from the experiment)

            #set up the TTL ports based on the mode.
            if self.writeTTL == 'Pulse':
                if not hasattr(self, 'portObj'):
                    print('\n***NOTICE: stimulus ', i, 'was skipped because a TTL write method was selected, but no port has been connected to.')
                    continue
                p._portObj = self.portObj #initialize portObj for sending TTL pulses
                p._portObj.rts = True #ensure TTL is OFF to begin
                p.burstTTL(self.win) #execute a stereotyped burst to mark the start of the stimulus in pulse mode
            elif self.writeTTL == 'Sustained':
                if not hasattr(self, 'portObj'):
                    print('\n***NOTICE: stimulus ', i, 'was skipped because a TTL write method was selected, but no port has been connected to.')
                    continue
                p._portObj = self.portObj
                p._portObj.rts = True #ensure TTL is OFF to begin
                p._TTLON = False #used to track state of sustained TTL pulses                
               
                if self.ttlBookmarks: #Run the bookmark before the start of each stimulus: this is 1 frame on, 2 frames off, 3 frames on, 4 frames Off, 5 frames On, 6 frames Off at the frame frate of self.win The port should end in the off position again. Range is not inclusive
                    self.win.flip() #brief pause at frame rate in case there was just another flip from the previous stimulus (e.g., on the last frame of the previous stimulus)
                    for bookmarkStep in range(1, 7):
                        p.sendTTL(bookmark = True)
                        for m in range(bookmarkStep): #flip a number of frames that is equal to the iteration number
                            self.win.flip()
                    
                    #just ensure that the TTL pulse is actually off:
                    if p._TTLON:            
                        p.sendTTL(bookmark = True)
                    

            #run the protocol
            p._sendEyeLinkMessage = self.sendEyeLinkMessage if self._elTracker is not None else None
            p._okrLogDir = self._resolveEyeLinkSaveDir()
            if self._elTracker is not None:
                safeName = displayName.replace(' ', '_')
                self.sendEyeLinkMessage('TRIALID {n}_{name}'.format(n=i + 1, name=safeName))
                self.sendEyeLinkMessage('!V TRIAL_VAR protocol {name}'.format(name=str(name).replace(' ', '_')))
                self.sendEyeLinkMessage('!V TRIAL_VAR suffix {suf}'.format(suf=str(suffix).replace(' ', '_')))
                if not getattr(p, '_okrSyncsTrialClock', False):
                    self.sendEyeLinkMessage('SYNCTIME')
            p.run(self.win, (self.useInformationMonitor, self.informationWin)) #send informationMonitor information as a tuple: bool (whether to use), window object
            if self._elTracker is not None:
                self.sendEyeLinkMessage('TRIAL_RESULT 0')
            
            #Make sure TTL port is turned OFF if running in sustained mode (it's often left on if the user quits a stimulus early)
            if self.writeTTL == 'Sustained' and p._TTLON:
                p.sendTTL()
                                
            #print the timing report if the user asks for it
            if p._timingReport:
                p.reportTime(displayName)
               

            #write down properties from previous stimulus
            protocolProperties = vars(p)
            protocolProperties.pop('_informationWin', None) #can't save ongoing psychopy win so remove it
            protocolProperties.pop('_elTracker', None)
            protocolProperties.pop('_sendEyeLinkMessage', None)
            protocolProperties.pop('_okrLogDir', None)
            self.loggedStimuli.append(protocolProperties)
