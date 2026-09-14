# -*- coding: utf-8 -*-
"""
Custom Validation presents the EyeLink HV9 target grid while recording, so
signed/unsigned accuracy and within-fixation precision can be computed offline
from the EDF/ASC sample stream.
"""
from protocols.protocol import protocol
from psychopy import visual, event
from datetime import datetime
from pathlib import Path
import math
import random


# EyeLink HV9 point order (must match Host C/V target ordering).
_HV9_LABELS = (
    'center',
    'top',
    'bottom',
    'left',
    'right',
    'top_left',
    'top_right',
    'bottom_left',
    'bottom_right',
)


class CustomValidation(protocol):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'CustomValidation' #Shows the EyeLink HV9 grid while recording so accuracy and precision can be measured from samples.
        self.stimulusReps = 1 #number of times to present the full 9-point grid
        self.preTime = 0.0 #seconds - blank before the first target (keep 0 so the first dot appears immediately after setup)
        self.stimTime = 5.0 #seconds - each target remains on screen for this duration
        self.tailTime = 0.0 #seconds - blank after the last target of each grid pass
        self.interStimulusInterval = 0.5 #seconds - blank between successive targets
        self.backgroundColor = [0.0, 0.0, 0.0] #mid-gray (PsychoPy RGB -1 to 1); matches EyeLink setup background
        self.targetColor = [1.0, -1.0, -1.0] #red outer ring (PsychoPy RGB -1 to 1)
        self.targetInnerColor = [-1.0, -1.0, -1.0] #black inner disk
        self.targetSizeDegrees = 0.5 #degrees - outer target diameter (matches Bassoon standard cal target)
        self.targetInnerSizeDegrees = 0.15 #degrees - inner disk diameter
        self.randomizeOrder = True #if True, shuffle point order within each grid pass (like Host C/V); set False to use fixed HV9 index order
        self.useExperimentAreaDegrees = True #if True, use EyeLink cal/val ± degree extents from Options; if False, use areaDegreesH/V below
        self.areaDegreesSource = 'validation' # 'validation' or 'calibration' — which Options degree pair to use when useExperimentAreaDegrees is True
        self.areaDegreesH = 20.0 #± horizontal extent from center in visual degrees (used when useExperimentAreaDegrees is False)
        self.areaDegreesV = 11.0 #± vertical extent from center in visual degrees (used when useExperimentAreaDegrees is False)
        self.cornerScaling = 1.0 #scales corner targets toward center (1.0 = corners of the mid-edge rectangle; Host may use ~0.88)


    def internalValidation(self):
        tf = True
        errorMessage = []
        tfColors, colorErrors = self.validateColorInput()
        if not tfColors:
            tf = False
            errorMessage += colorErrors
        if self.stimTime <= 0:
            tf = False
            errorMessage.append('Stim Time must be greater than 0 seconds.')
        if self.targetSizeDegrees <= 0:
            tf = False
            errorMessage.append('Target Size Degrees must be greater than 0.')
        if self.targetInnerSizeDegrees < 0:
            tf = False
            errorMessage.append('Target Inner Size Degrees must be >= 0.')
        if self.targetInnerSizeDegrees >= self.targetSizeDegrees:
            tf = False
            errorMessage.append('Target Inner Size Degrees must be smaller than Target Size Degrees.')
        for name, value in (
            ('Area Degrees H', self.areaDegreesH),
            ('Area Degrees V', self.areaDegreesV),
        ):
            if value <= 0:
                tf = False
                errorMessage.append(name + ' must be greater than 0 degrees (± from center).')
        if self.cornerScaling <= 0 or self.cornerScaling > 1.0:
            tf = False
            errorMessage.append('Corner Scaling must be > 0 and <= 1.0.')
        source = str(self.areaDegreesSource).strip().lower()
        if source not in ('validation', 'calibration'):
            tf = False
            errorMessage.append("Area Degrees Source must be 'validation' or 'calibration'.")
        if self.stimulusReps < 1:
            tf = False
            errorMessage.append('Stimulus Reps must be at least 1.')
        return tf, errorMessage


    def estimateTime(self):
        nPoints = 9 * int(self.stimulusReps)
        betweenTargets = max(0, nPoints - 1) * self.interStimulusInterval
        self._estimatedTime = (
            self.preTime
            + nPoints * self.stimTime
            + betweenTargets
            + self.tailTime
        )
        return self._estimatedTime


    def _monitorFovDegrees(self, monitor):
        '''Horizontal and vertical FOV in degrees from a PsychoPy monitor profile.'''
        try:
            eyeDistance = monitor.getDistance()
            cmWide = monitor.getWidth()
            sizePix = monitor.getSizePix() or getattr(monitor, 'currentCalib', {}).get('sizePix')
            if not eyeDistance or not cmWide or eyeDistance <= 0 or cmWide <= 0:
                return None, None
            if sizePix and sizePix[0]:
                cmHigh = cmWide * (float(sizePix[1]) / float(sizePix[0]))
            else:
                cmHigh = cmWide * 9.0 / 16.0
            hFov = 2 * math.degrees(math.atan((cmWide / 2.0) / eyeDistance))
            vFov = 2 * math.degrees(math.atan((cmHigh / 2.0) / eyeDistance))
            return hFov, vFov
        except Exception:
            return None, None


    def _degreesToProportion(self, degrees_h, degrees_v, monitor):
        '''Convert ± degree extents from center to EyeLink-style area proportions.'''
        hFov, vFov = self._monitorFovDegrees(monitor)
        if hFov is None or vFov is None or hFov <= 0 or vFov <= 0:
            return None
        try:
            # Total span = 2 × entered ± extent
            prop_h = (2.0 * abs(float(degrees_h))) / hFov
            prop_v = (2.0 * abs(float(degrees_v))) / vFov
        except (TypeError, ValueError):
            return None
        return [
            min(1.0, max(0.2, prop_h)),
            min(1.0, max(0.2, prop_v)),
        ]


    def _resolvedAreaDegrees(self):
        '''Return (±H°, ±V°) extents from screen center for the HV9 rectangle.'''
        useExperiment = getattr(self, 'useExperimentAreaDegrees', None)
        if useExperiment is None:
            useExperiment = getattr(self, 'useExperimentAreaProportion', True)
        if useExperiment:
            source = str(getattr(self, 'areaDegreesSource', getattr(self, 'areaProportionSource', 'validation'))).strip().lower()
            if source == 'calibration':
                degs = getattr(self, '_eyeLinkCalibrationAreaDegrees', None)
            else:
                degs = getattr(self, '_eyeLinkValidationAreaDegrees', None)
            if degs is not None and len(degs) >= 2:
                try:
                    if float(degs[0]) > 0 and float(degs[1]) > 0:
                        return [float(degs[0]), float(degs[1])]
                except (TypeError, ValueError):
                    pass
        # Local protocol degrees (with legacy proportion fallback converted via defaults)
        try:
            h = float(getattr(self, 'areaDegreesH', None))
            v = float(getattr(self, 'areaDegreesV', None))
            if h > 0 and v > 0:
                return [h, v]
        except (TypeError, ValueError):
            pass
        return [20.0, 11.0]


    def _resolvedAreaProportion(self, monitor=None):
        '''Return (H, V) proportions for the HV9 rectangle.'''
        degs = self._resolvedAreaDegrees()
        if monitor is not None:
            props = self._degreesToProportion(degs[0], degs[1], monitor)
            if props is not None:
                return props
        # Fall back to injected experiment proportions if FOV is unavailable.
        useExperiment = getattr(self, 'useExperimentAreaDegrees', None)
        if useExperiment is None:
            useExperiment = getattr(self, 'useExperimentAreaProportion', True)
        if useExperiment:
            source = str(getattr(self, 'areaDegreesSource', getattr(self, 'areaProportionSource', 'validation'))).strip().lower()
            if source == 'calibration':
                props = getattr(self, '_eyeLinkCalibrationAreaProportion', None)
            else:
                props = getattr(self, '_eyeLinkValidationAreaProportion', None)
            if props is not None and len(props) >= 2:
                try:
                    return [
                        min(1.0, max(0.2, float(props[0]))),
                        min(1.0, max(0.2, float(props[1]))),
                    ]
                except (TypeError, ValueError):
                    pass
        # Legacy local proportion attributes
        try:
            return [
                min(1.0, max(0.2, float(getattr(self, 'areaProportionH', 0.88)))),
                min(1.0, max(0.2, float(getattr(self, 'areaProportionV', 0.83)))),
            ]
        except (TypeError, ValueError):
            return [0.88, 0.83]


    @staticmethod
    def hv9TargetsEyeLinkPix(width, height, prop_h, prop_v, corner_scaling=1.0):
        '''
        EyeLink-style HV9 target locations in Host/display pixels
        (origin top-left, y down), matching VALIDATE POINT ordering.
        '''
        cx = width / 2.0
        cy = height / 2.0
        half_w = (float(prop_h) * width) / 2.0
        half_h = (float(prop_v) * height) / 2.0
        left = cx - half_w
        right = cx + half_w
        top = cy - half_h
        bottom = cy + half_h
        scale = float(corner_scaling)
        c_left = cx - half_w * scale
        c_right = cx + half_w * scale
        c_top = cy - half_h * scale
        c_bottom = cy + half_h * scale
        return [
            (cx, cy),
            (cx, top),
            (cx, bottom),
            (left, cy),
            (right, cy),
            (c_left, c_top),
            (c_right, c_top),
            (c_left, c_bottom),
            (c_right, c_bottom),
        ]


    @staticmethod
    def eyeLinkPixToPsychoPy(x_el, y_el, win_width, win_height):
        '''Convert EyeLink display pixels to PsychoPy pix units (origin center, y up).'''
        return (
            float(x_el) - win_width / 2.0,
            win_height / 2.0 - float(y_el),
        )


    def _sendEyeLinkMsg(self, text):
        sendMessage = getattr(self, '_sendEyeLinkMessage', None)
        if sendMessage is not None:
            sendMessage(text)


    def _writePointLog(self, events, prop_h, prop_v, deg_h=None, deg_v=None):
        if not events:
            return None
        logDir = getattr(self, '_okrLogDir', None)
        if logDir is None:
            logDir = Path.cwd()
        else:
            logDir = Path(logDir)
        logDir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logPath = logDir / ('CustomValidation_Log_{stamp}.txt'.format(stamp=stamp))
        headerLines = [
            '# Custom Validation Condition Log',
            '# StimulusName: Bassoon CustomValidation',
            '# TimeBase: seconds from EyeLink SYNCTIME (sent when stimulus timing clock starts)',
            '# HV9Order: center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right',
            '# AreaDegreesH: {h} (± from center)'.format(h='{:.4f}'.format(deg_h) if deg_h is not None else 'NA'),
            '# AreaDegreesV: {v} (± from center)'.format(v='{:.4f}'.format(deg_v) if deg_v is not None else 'NA'),
            '# AreaTotalSpanH: {h}'.format(h='{:.4f}'.format(2.0 * deg_h) if deg_h is not None else 'NA'),
            '# AreaTotalSpanV: {v}'.format(v='{:.4f}'.format(2.0 * deg_v) if deg_v is not None else 'NA'),
            '# AreaProportionH: {h:.4f}'.format(h=prop_h),
            '# AreaProportionV: {v:.4f}'.format(v=prop_v),
            '# CornerScaling: {s:.4f}'.format(s=float(self.cornerScaling)),
            '# StimTimeSec: {t:.4f}'.format(t=float(self.stimTime)),
            '# Note: Use POINT_START/POINT_END messages (or startTime/endTime) to window ASC samples per target.',
            '# Note: Drop ~0.5-1.0 s after POINT_START when computing precision so saccades settle.',
            'eventIndex\tpointIndex\tpointLabel\trep\tstartTime\tendTime\ttargetEyeLinkX\ttargetEyeLinkY\ttargetPsychoPyX\ttargetPsychoPyY',
        ]
        rowLines = []
        for event in events:
            rowLines.append('\t'.join([
                str(event['eventIndex']),
                str(event['pointIndex']),
                str(event['pointLabel']),
                str(event['rep']),
                '{:.6f}'.format(event['startTime']),
                '{:.6f}'.format(event['endTime']),
                '{:.3f}'.format(event['targetEyeLinkX']),
                '{:.3f}'.format(event['targetEyeLinkY']),
                '{:.3f}'.format(event['targetPsychoPyX']),
                '{:.3f}'.format(event['targetPsychoPyY']),
            ]))
        logPath.write_text('\n'.join(headerLines + rowLines) + '\n', encoding='utf-8')
        return logPath


    def run(self, win, informationWin):
        self._completed = 0
        self._informationWin = informationWin

        self.getFR(win)
        self._interStimulusIntervalNumFrames = round(self._FR * self.interStimulusInterval)
        self._actualInterStimulusInterval = self._interStimulusIntervalNumFrames * (1.0 / self._FR)

        pixPerDeg = self.getPixPerDeg(win.monitor)
        outerSize = float(self.targetSizeDegrees) * pixPerDeg
        innerSize = float(self.targetInnerSizeDegrees) * pixPerDeg

        deg_h, deg_v = self._resolvedAreaDegrees()
        prop_h, prop_v = self._resolvedAreaProportion(win.monitor)
        width = float(win.size[0])
        height = float(win.size[1])
        targets_el = self.hv9TargetsEyeLinkPix(
            width, height, prop_h, prop_v, self.cornerScaling,
        )

        # Create targets before timing starts so the first POINT_START frame can
        # draw immediately (no stimulus-construction hitch on the first flip).
        outer = visual.Circle(
            win=win,
            radius=outerSize / 2.0,
            fillColor=self.targetColor,
            lineColor=None,
            units='pix',
        )
        inner = visual.Circle(
            win=win,
            radius=max(innerSize / 2.0, 0.5),
            fillColor=self.targetInnerColor,
            lineColor=None,
            units='pix',
        )

        if self.userInitiated:
            self.showInformationText(
                win,
                'Stimulus Information: Custom Validation\n'
                'HV9 targets for {t:.1f}s each\nPress any key to begin'.format(
                    t=float(self.stimTime),
                ),
            )
            event.waitKeys()

        trialClock = self._startTrialClock()
        events = []
        eventCounter = [0]

        win.color = self.backgroundColor
        for _ in range(self._preTimeNumFrames):
            win.flip()
            if self.checkQuitOrPause():
                return

        self._sendEyeLinkMsg(
            'CUSTOM_VAL GRID propH={h:.4f} propV={v:.4f} corner={c:.4f}'.format(
                h=prop_h, v=prop_v, c=float(self.cornerScaling),
            )
        )

        for rep in range(int(self.stimulusReps)):
            order = list(range(9))
            if self.randomizeOrder:
                random.shuffle(order)

            for orderIdx, pointIndex in enumerate(order):
                if self._informationWin[0]:
                    self.showInformationText(
                        win,
                        'Custom Validation\n'
                        'Rep {r}/{nRep}  Point {p}/9 ({label})'.format(
                            r=rep + 1,
                            nRep=int(self.stimulusReps),
                            p=orderIdx + 1,
                            label=_HV9_LABELS[pointIndex],
                        ),
                    )

                if not (rep == 0 and orderIdx == 0):
                    win.color = self.backgroundColor
                    for _ in range(self._interStimulusIntervalNumFrames):
                        win.flip()
                        if self.checkQuitOrPause():
                            return

                x_el, y_el = targets_el[pointIndex]
                x_pp, y_pp = self.eyeLinkPixToPsychoPy(x_el, y_el, width, height)
                outer.pos = (x_pp, y_pp)
                inner.pos = (x_pp, y_pp)

                startTime = trialClock.getTime()
                self._stimulusStartLog.append(startTime)
                self.sendTTL()
                self._numberOfEpochsStarted += 1
                self._sendEyeLinkMsg(
                    'CUSTOM_VAL POINT_START {idx} {label} at {x:.1f},{y:.1f}'.format(
                        idx=pointIndex,
                        label=_HV9_LABELS[pointIndex],
                        x=x_el,
                        y=y_el,
                    )
                )

                for _ in range(self._stimTimeNumFrames):
                    outer.draw()
                    if self.targetInnerSizeDegrees > 0:
                        inner.draw()
                    win.flip()
                    if self.checkQuitOrPause():
                        return

                endTime = trialClock.getTime()
                self._stimulusEndLog.append(endTime)
                self.sendTTL()
                self._numberOfEpochsCompleted += 1
                self._sendEyeLinkMsg(
                    'CUSTOM_VAL POINT_END {idx} {label} at {x:.1f},{y:.1f}'.format(
                        idx=pointIndex,
                        label=_HV9_LABELS[pointIndex],
                        x=x_el,
                        y=y_el,
                    )
                )

                eventCounter[0] += 1
                events.append({
                    'eventIndex': eventCounter[0],
                    'pointIndex': pointIndex,
                    'pointLabel': _HV9_LABELS[pointIndex],
                    'rep': rep + 1,
                    'startTime': startTime,
                    'endTime': endTime,
                    'targetEyeLinkX': x_el,
                    'targetEyeLinkY': y_el,
                    'targetPsychoPyX': x_pp,
                    'targetPsychoPyY': y_pp,
                })

        win.color = self.backgroundColor
        for _ in range(self._tailTimeNumFrames):
            win.flip()
            if self.checkQuitOrPause():
                return

        self._sendEyeLinkMsg('CUSTOM_VAL GRID_END')
        logPath = self._writePointLog(events, prop_h, prop_v, deg_h=deg_h, deg_v=deg_v)
        if logPath is not None:
            print('--> Wrote Custom Validation log:', logPath)
            print(
                '    HV9 area proportions H={h:.3f} V={v:.3f}, corner scaling={c:.3f}'.format(
                    h=prop_h, v=prop_v, c=float(self.cornerScaling),
                )
            )

        self._completed = 1
