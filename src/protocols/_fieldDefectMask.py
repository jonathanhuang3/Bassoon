# -*- coding: utf-8 -*-
"""
Shared gaze-contingent field-defect mask for Contrast Dots.

Not a stimulus by itself. Protocol files subclass FieldDefectMask.
Screen coordinates are degrees from the current gaze: +x right, +y up.
A right-eye field has nasal on the left and the blind spot on the right.
A left-eye field is mirrored.
"""
import math

import numpy as np
from psychopy import visual

from protocols.ContrastDots import ContrastDots
from protocols.ContrastDotsWithTunnelMask import ContrastDotsWithTunnelMask


def temporal_sign(eye):
    '''+1 when the temporal field is to the right of fixation (right eye).'''
    return 1.0 if str(eye).strip().lower() == 'right' else -1.0


def catmull_rom_samples(points, samples_per_segment=24):
    '''Evenly sampled Catmull-Rom curve through points, shape (N, 2).'''
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 1:
        return points.copy()
    padded = np.vstack([points[0], points, points[-1]])
    chunks = []
    count = samples_per_segment
    t = np.linspace(0.0, 1.0, count, endpoint=False)
    t2 = t * t
    t3 = t2 * t
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        curve = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3[:, None]
        )
        chunks.append(curve)
    chunks.append(points[-1][None, :])
    return np.vstack(chunks)


def arc_length(samples):
    '''Cumulative arc length of a polyline, starting at 0.'''
    step = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def tapered_widths(arc_s, knot_s, temporal_width, fixation_width, nasal_width):
    '''Full band width along arc length: temporal → fixation knot → nasal.'''
    arc_s = np.asarray(arc_s, dtype=np.float64)
    end_s = float(arc_s[-1]) if len(arc_s) else 0.0
    knot_s = float(np.clip(knot_s, 0.0, end_s))
    widths = np.empty(len(arc_s), dtype=np.float64)
    if end_s <= 1e-9:
        widths.fill(float(temporal_width))
        return widths
    before = arc_s <= knot_s
    if knot_s > 1e-9:
        t_before = arc_s[before] / knot_s
        widths[before] = temporal_width + t_before * (fixation_width - temporal_width)
    else:
        before = np.zeros(len(arc_s), dtype=bool)
    after = ~before
    span = max(end_s - knot_s, 1e-9)
    t_after = (arc_s[after] - knot_s) / span
    widths[after] = fixation_width + t_after * (nasal_width - fixation_width)
    return widths


def band_occlusion(x, y, samples, widths):
    '''
    Degrees inside a variable-width band around a polyline.

    widths are full widths. Positive inside the band, negative outside.
    '''
    samples = np.asarray(samples, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    best = np.full(np.broadcast(x, y).shape, -1.0e6, dtype=np.float32)
    if len(samples) == 1:
        dist = np.hypot(x - samples[0, 0], y - samples[0, 1])
        return (widths[0] / 2.0 - dist).astype(np.float32)
    for i in range(len(samples) - 1):
        ax, ay = samples[i]
        bx, by = samples[i + 1]
        abx = bx - ax
        aby = by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            continue
        t = ((x - ax) * abx + (y - ay) * aby) / ab2
        t = np.clip(t, 0.0, 1.0)
        dist = np.hypot(x - (ax + t * abx), y - (ay + t * aby))
        half = 0.5 * (widths[i] + t * (widths[i + 1] - widths[i]))
        best = np.maximum(best, (half - dist).astype(np.float32))
    return best


def nasal_ceiling(canonical_x, boundary_x, boundary_y):
    '''
    Smooth upper boundary of a right-eye nasal step.

    boundary points use right-eye degrees (nasal x is negative). Nasal to the
    last point, the ceiling stays at that point's y. Temporal to the first
    point, it stays at the first y.
    '''
    points = np.stack([
        np.asarray(boundary_x, dtype=np.float64),
        np.asarray(boundary_y, dtype=np.float64),
    ], axis=1)
    order = np.argsort(points[:, 0])
    points = points[order]
    samples = catmull_rom_samples(points, samples_per_segment=32)
    sample_order = np.argsort(samples[:, 0])
    xs = samples[sample_order, 0]
    ys = samples[sample_order, 1]
    # A spline can fold back in x; keep the first sample at each x.
    unique_x, unique_index = np.unique(np.round(xs, 5), return_index=True)
    return np.interp(
        np.asarray(canonical_x, dtype=np.float64),
        unique_x,
        ys[unique_index],
        left=float(ys[unique_index][0]),
        right=float(ys[unique_index][-1]),
    ).astype(np.float32)


class FieldDefectMask(ContrastDotsWithTunnelMask):
    '''Gaze-following scotoma. Subclasses define the occluded region.'''

    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self._stripLegacyTunnelParams()
        self.blockedHemifield = 'upper'  # 'upper' or 'lower' visual field, relative to gaze
        self.transitionWidthDegrees = 1.0  # raised-cosine width on the soft edges, in degrees
        self.maskColor = [0.0, 0.0, 0.0]  # mid-gray, matching the background
        self.persistentDots = False
        self.contrasts = [1.0]
        self.directions = [0.0, 90.0, 180.0, 270.0]  # degrees - right, up, left, down
        self.eye = 'right'  # 'right' or 'left' visual field. Nasal and temporal sides follow this eye.

    def _stripLegacyTunnelParams(self):
        for name in ('tunnelVisibleDiameterDegrees', 'tunnelEdgeSigmaDegrees'):
            if hasattr(self, name):
                delattr(self, name)

    def _defectName(self):
        return 'field defect'

    def _okrExtraFields(self, blockIndex, direction=None):
        fields = super()._okrExtraFields(blockIndex, direction)
        fields['hemifield'] = str(self.blockedHemifield).strip().lower()
        fields['tunnelDiameterDeg'] = 'NA'
        return fields

    def _stimulusTitle(self):
        eye = str(self.eye).strip().lower()
        return 'Contrast Dots with {eye} {defect}'.format(
            eye='Right' if eye == 'right' else 'Left',
            defect=self._defectName(),
        )

    def _gazeMissingMessage(self):
        return (
            '*** {title}: No valid gaze sample yet; holding mask at last position.'.format(
                title=self._stimulusTitle(),
            )
        )

    def _trackerInactiveMessage(self):
        return (
            '*** {title}: EyeLink is not active. The mask will stay at screen center.'.format(
                title=self._stimulusTitle(),
            )
        )

    def _normalizeEyeAndHemifield(self):
        self.eye = str(self.eye).strip().lower()
        self.blockedHemifield = str(self.blockedHemifield).strip().lower()

    def internalValidation(self):
        self._stripLegacyTunnelParams()
        tf, errorMessage = ContrastDots.internalValidation(self)
        self._normalizeEyeAndHemifield()
        if self.eye not in ('left', 'right'):
            tf = False
            errorMessage.append("Eye must be 'left' or 'right'.")
        if self.blockedHemifield not in ('upper', 'lower'):
            tf = False
            errorMessage.append("Blocked Hemifield must be 'upper' or 'lower'.")
        if float(self.transitionWidthDegrees) < 0:
            tf = False
            errorMessage.append('Transition Width must be 0 or greater degrees.')
        return tf, errorMessage

    def _maskDegrees(self, tex_size, overlay_size_pix, ppd_xy):
        '''Pixel grid of the mask texture, in degrees from gaze. +x right, +y up.'''
        ppd_h, ppd_v = ppd_xy
        center = (tex_size - 1) / 2.0
        yy, xx = np.mgrid[0:tex_size, 0:tex_size]
        scale = overlay_size_pix / float(tex_size)
        x_deg = (xx - center) * scale / ppd_h
        y_deg = (center - yy) * scale / ppd_v
        return x_deg, y_deg

    def _oppositeHemifield(self, y_deg):
        '''True on the seeing side of the horizontal meridian.'''
        if self.blockedHemifield == 'upper':
            return y_deg < 0.0
        return y_deg > 0.0

    def _buildDefectMask(self, x_deg, y_deg):
        raise NotImplementedError

    def _buildTunnelOpacityMap(self, tex_size, overlay_size_pix, ppd_xy):
        x_deg, y_deg = self._maskDegrees(tex_size, overlay_size_pix, ppd_xy)
        return self._buildDefectMask(x_deg, y_deg)

    def _maskLogLine(self):
        return (
            'blocking {side} hemifield of the {eye} eye, '
            'raised-cosine transition = {w:g}°'
        ).format(
            side=self.blockedHemifield,
            eye=self.eye,
            w=float(self.transitionWidthDegrees),
        )

    def prepareBeforeWaitingRoom(self, win):
        '''Build the scotoma, dots, and fixation cross while the waiting room is up.'''
        print('--> Preparing the scotoma mask before the trial starts.')
        super().prepareBeforeWaitingRoom(win)

    def _maskIsReady(self, win, ppd_xy):
        ppd_h, ppd_v = ppd_xy
        return (
            getattr(self, '_gazeMask', None) is not None
            and getattr(self, '_preparedWin', None) is win
            and getattr(self, '_preparedWinSize', None) == (int(win.size[0]), int(win.size[1]))
            and getattr(self, '_preparedPpd', None) == (float(ppd_h), float(ppd_v))
        )

    def _initPerRunStimulus(self, win, ppd_xy):
        self._stripLegacyTunnelParams()
        self._normalizeEyeAndHemifield()
        ppd_h, ppd_v = ppd_xy
        if self._maskIsReady(win, ppd_xy):
            self._initGazeReadCache(win)
            return
        transition_pix = max(
            0.0,
            float(self.transitionWidthDegrees) * max(ppd_h, ppd_v),
        )
        full_diagonal = math.hypot(float(win.size[0]), float(win.size[1]))
        gaze_margin_pix = 200.0
        overlay_half_size = int(math.ceil(
            full_diagonal + gaze_margin_pix + transition_pix + 1.0
        ))
        overlay_size_pix = max(overlay_half_size * 2, 2)
        tex_size = 1024
        self._tunnelOverlaySizePix = overlay_size_pix
        self._tunnelTexSize = tex_size
        self._tunnelPpdXy = (ppd_h, ppd_v)
        opacity = self._buildTunnelOpacityMap(tex_size, overlay_size_pix, ppd_xy)
        self._gazeMask = visual.ImageStim(
            win,
            image=np.ones((tex_size, tex_size), dtype=np.float32),
            mask=opacity,
            color=self.maskColor,
            size=(overlay_size_pix, overlay_size_pix),
            units='pix',
            pos=(0.0, 0.0),
            interpolate=True,
        )
        self._lastGaze = (0.0, 0.0)
        self._gazeMaskWarningShown = False
        self._initGazeReadCache(win)
        print(
            '--> {title}: {detail} '
            '(mask overlay {s} px, ppd_h={h:.2f} ppd_v={v:.2f}).'.format(
                title=self._stimulusTitle(),
                detail=self._maskLogLine(),
                s=overlay_size_pix,
                h=ppd_h,
                v=ppd_v,
            )
        )
        if self._tracker is None:
            print(self._trackerInactiveMessage())
            self._gazeMaskWarningShown = True
        self._preparedWin = win
        self._preparedWinSize = (int(win.size[0]), int(win.size[1]))
        self._preparedPpd = (float(ppd_h), float(ppd_v))

    def _teardownPerRunStimulus(self):
        super()._teardownPerRunStimulus()
        self._preparedWin = None
        self._preparedWinSize = None
        self._preparedPpd = None


class NasalStepMask(FieldDefectMask):
    '''
    Nasal step: a finite patch on the horizontal meridian in the nasal field.

    For a right eye and an upper defect, the step runs from 10° to 30° nasal
    and stays above the horizontal meridian. The upper edge passes through
    (-10°, 3°), (-15°, 5°), (-21°, 7°), and (-27°, 9°). The meridian is a
    hard edge. The central 10° stays clear. A left eye mirrors x. A lower
    hemifield mirrors y. The patch keeps this retinal size as gaze moves.
    '''

    def __init__(self):
        super().__init__()
        self.blockedHemifield = 'upper'  # 'upper' or 'lower'. The step sits on the horizontal meridian.
        self.nasalBoundaryX = [-10.0, -15.0, -21.0, -27.0]  # right-eye upper border x, degrees. Nasal is negative. Left eye mirrors x.
        self.nasalBoundaryY = [3.0, 5.0, 7.0, 9.0]  # right-eye upper border y, degrees. Lower hemifield mirrors y.
        self.nasalOuterDegrees = 30.0  # far edge of the step, degrees nasal of fixation. The defect ends here.
        self.centralClearDegrees = 10.0  # radius around fixation that stays clear.
        self.transitionWidthDegrees = 1.0  # soft edge on the ceiling and both nasal walls, in degrees. The meridian stays hard.

    def _defectName(self):
        return 'Nasal Step'

    def internalValidation(self):
        tf, errorMessage = super().internalValidation()
        xs = list(self.nasalBoundaryX)
        ys = list(self.nasalBoundaryY)
        if len(xs) < 2 or len(xs) != len(ys):
            tf = False
            errorMessage.append('Nasal boundary X and Y must be lists of the same length, with at least 2 points.')
        if float(self.centralClearDegrees) < 0:
            tf = False
            errorMessage.append('Central Clear must be 0 or greater degrees.')
        if float(self.nasalOuterDegrees) <= 0:
            tf = False
            errorMessage.append('Nasal Outer must be greater than 0 degrees.')
        elif len(xs) >= 1 and float(self.nasalOuterDegrees) <= -float(np.max(xs)):
            tf = False
            errorMessage.append('Nasal Outer must be farther from fixation than the inner nasal wall.')
        return tf, errorMessage

    def _maskLogLine(self):
        return (
            '{eye} eye, {side} nasal step from the inner wall to {outer:g}° nasal, '
            'central {clear:g}° clear, transition {w:g}°'
        ).format(
            eye=self.eye,
            side=self.blockedHemifield,
            outer=float(self.nasalOuterDegrees),
            clear=float(self.centralClearDegrees),
            w=float(self.transitionWidthDegrees),
        )

    def _buildDefectMask(self, x_deg, y_deg):
        canonical_x = np.asarray(x_deg, dtype=np.float64) * temporal_sign(self.eye)
        into_hemifield = y_deg if self.blockedHemifield == 'upper' else -y_deg
        ceiling = nasal_ceiling(canonical_x, self.nasalBoundaryX, self.nasalBoundaryY)
        wall_x = float(np.max(self.nasalBoundaryX))
        # Far edge in right-eye degrees. The step is a finite patch, so looking
        # temporally cannot recruit an endless nasal field onto the screen.
        outer_x = -abs(float(self.nasalOuterDegrees))
        # Positive inside: between the inner and outer nasal walls, and below the ceiling.
        occlusion = np.minimum(wall_x - canonical_x, canonical_x - outer_x)
        occlusion = np.minimum(occlusion, ceiling - into_hemifield)
        radius = np.hypot(x_deg, y_deg)
        occlusion = np.minimum(occlusion, radius - float(self.centralClearDegrees))
        mask = self._raisedCosineOpacity(occlusion, float(self.transitionWidthDegrees))
        # Hard edge on the horizontal meridian.
        mask = np.where(self._oppositeHemifield(y_deg), -1.0, mask)
        return mask.astype(np.float32)


class ArcuateMask(FieldDefectMask):
    '''
    Arcuate defect: a tapered band along a spline in one hemifield.

    For a right eye and an upper defect the centerline runs from (15°, 0°)
    through (6°, 9°), (0°, 10°), and (−8°, 8.5°) to (−20°, 0°). Width grows
    from 2.5° at the temporal end to 7° above fixation to 12° at the nasal end.
    The horizontal meridian is a hard edge. Other borders are soft. The central
    5° stays clear. A left eye mirrors x. A lower hemifield mirrors y.
    '''

    def __init__(self):
        super().__init__()
        self.blockedHemifield = 'upper'  # 'upper' or 'lower'. The arc bows through this hemifield.
        self.arcuateCenterX = [15.0, 6.0, 0.0, -8.0, -20.0]  # right-eye centerline x, degrees. Left eye mirrors this.
        self.arcuateCenterY = [0.0, 9.0, 10.0, 8.5, 0.0]  # right-eye upper centerline y, degrees. Lower hemifield mirrors this.
        self.temporalWidthDegrees = 2.5  # full band width at the temporal end of the centerline.
        self.fixationWidthDegrees = 7.0  # full band width at the centerline point closest to the vertical meridian.
        self.nasalWidthDegrees = 12.0  # full band width at the nasal end of the centerline.
        self.centralClearDegrees = 5.0  # radius around fixation that stays clear.
        self.transitionWidthDegrees = 1.0  # soft edge everywhere except the horizontal meridian, in degrees.

    def _defectName(self):
        return 'Arcuate Defect'

    def internalValidation(self):
        tf, errorMessage = super().internalValidation()
        xs = list(self.arcuateCenterX)
        ys = list(self.arcuateCenterY)
        if len(xs) < 2 or len(xs) != len(ys):
            tf = False
            errorMessage.append('Arcuate centerline X and Y must be lists of the same length, with at least 2 points.')
        for name in ('temporalWidthDegrees', 'fixationWidthDegrees', 'nasalWidthDegrees'):
            if float(getattr(self, name)) <= 0:
                tf = False
                errorMessage.append(name + ' must be greater than 0 degrees.')
        if float(self.centralClearDegrees) < 0:
            tf = False
            errorMessage.append('Central Clear must be 0 or greater degrees.')
        return tf, errorMessage

    def _maskLogLine(self):
        return (
            '{eye} eye, {side} arcuate spline, widths {temporal:g}° / {fix:g}° / {nasal:g}°, '
            'central {clear:g}° clear, transition {w:g}°'
        ).format(
            eye=self.eye,
            side=self.blockedHemifield,
            temporal=float(self.temporalWidthDegrees),
            fix=float(self.fixationWidthDegrees),
            nasal=float(self.nasalWidthDegrees),
            clear=float(self.centralClearDegrees),
            w=float(self.transitionWidthDegrees),
        )

    def _arcuateCenterline(self):
        '''Centerline in degrees for this eye and hemifield.'''
        xs = np.asarray(self.arcuateCenterX, dtype=np.float64)
        ys = np.asarray(self.arcuateCenterY, dtype=np.float64)
        if temporal_sign(self.eye) < 0:
            xs = -xs
        if self.blockedHemifield == 'lower':
            ys = -ys
        return np.stack([xs, ys], axis=1)

    def _buildDefectMask(self, x_deg, y_deg):
        centerline = self._arcuateCenterline()
        canonical_x = np.asarray(self.arcuateCenterX, dtype=np.float64)
        knot_index = int(np.argmin(np.abs(canonical_x)))
        samples = catmull_rom_samples(centerline)
        distance = arc_length(samples)
        # Arc length of the knot on the unsampled polyline, measured the same way
        # on the sampled curve by the sample nearest that control point.
        knot_point = centerline[knot_index]
        knot_s = float(distance[np.argmin(np.linalg.norm(samples - knot_point, axis=1))])
        widths = tapered_widths(
            distance,
            knot_s,
            float(self.temporalWidthDegrees),
            float(self.fixationWidthDegrees),
            float(self.nasalWidthDegrees),
        )
        occlusion = band_occlusion(x_deg, y_deg, samples, widths)
        radius = np.hypot(x_deg, y_deg)
        occlusion = np.minimum(occlusion, radius - float(self.centralClearDegrees))
        mask = self._raisedCosineOpacity(occlusion, float(self.transitionWidthDegrees))
        # Hard edge on the horizontal meridian. The rest of the border stays soft.
        mask = np.where(self._oppositeHemifield(y_deg), -1.0, mask)
        return mask.astype(np.float32)
