# -*- coding: utf-8 -*-
"""
Contrast Dots with Tunnel Mask is Contrast Dots with a gaze-contingent tunnel
mask: a circular aperture follows EyeLink gaze while the periphery is occluded
with a soft raised-cosine edge using the gray background color.
"""
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from psychopy import core, visual

from protocols.ContrastDots import ContrastDots


class ContrastDotsWithTunnelMask(ContrastDots):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithTunnelMask'
        self.tunnelVisibleDiameterDegrees = 10.0  # clear aperture diameter in degrees
        self.transitionWidthDegrees = 1.0  # raised-cosine edge width in degrees (0 = hard edge)
        self.maskColor = [0.0, 0.0, 0.0]  # defaults to the gray background color
        self.persistentDots = False  # True = dots never respawn (infinite lifetime)
        self.contrasts = [1.0, 0.1, 0.05, -0.05, -0.1, -1.0]
        self.logGazeLatency = True  # True = log per-frame gaze-contingent latency to a text file
        if hasattr(self, 'tunnelEdgeSigmaDegrees'):
            delattr(self, 'tunnelEdgeSigmaDegrees')

    def _usePersistentDots(self):
        return bool(self.persistentDots)

    def _stimulusTitle(self):
        return 'Contrast Dots with Tunnel Mask'

    def _gazeMissingMessage(self):
        return (
            '*** Contrast Dots with Tunnel Mask: No valid gaze sample yet; '
            'holding tunnel at last position.'
        )

    def _trackerInactiveMessage(self):
        return (
            '*** Contrast Dots with Tunnel Mask: EyeLink is not active. '
            'The tunnel will stay at screen center.'
        )

    def internalValidation(self):
        if hasattr(self, 'tunnelEdgeSigmaDegrees'):
            delattr(self, 'tunnelEdgeSigmaDegrees')
        tf, errorMessage = super().internalValidation()
        if self.tunnelVisibleDiameterDegrees < 0:
            tf = False
            errorMessage.append('Tunnel Visible Diameter must be 0 or greater degrees.')
        if getattr(self, 'transitionWidthDegrees', 0) < 0:
            tf = False
            errorMessage.append('Transition Width must be 0 or greater degrees.')
        return tf, errorMessage

    @staticmethod
    def _raisedCosineOpacity(t, width):
        '''
        Map signed distance to a PsychoPy ImageStim mask in [-1, 1].

        PsychoPy numpy masks use -1 = fully transparent and +1 = fully opaque.
        Using 0..1 incorrectly leaves the "clear" side at mid-alpha and veils dots.

        t and width must share units (visual degrees for the scotoma maps).
        t < 0 is the clear side; t > 0 is the occluded side. width <= 0 gives a
        hard edge at t=0.
        '''
        if width <= 0:
            # -1 clear, +1 occluded
            return np.where(t >= 0, 1.0, -1.0).astype(np.float32)
        half = width / 2.0
        # Start fully clear (-1); ramp to fully occluded (+1) across the fringe.
        mask = np.full(t.shape, -1.0, dtype=np.float32)
        mask[t >= half] = 1.0
        in_zone = (t > -half) & (t < half)
        u = (t[in_zone] + half) / width  # 0..1 across the transition
        # raised-cosine in 0..1, then map to -1..+1
        alpha = 0.5 * (1.0 - np.cos(np.pi * u))
        mask[in_zone] = (2.0 * alpha - 1.0).astype(np.float32)
        return mask

    def _eyeGazeValid(self, eye):
        '''
        True if this eye sample is usable for mask placement.

        During blinks EyeLink often emits garbage gaze (or pupil=0) for a few
        samples before MISSING_DATA. Updating the mask from those samples yanks
        it off-screen and briefly reveals the full dot field.
        '''
        try:
            gaze = eye.getGaze()
            pupil = eye.getPupilSize()
        except Exception:
            return False
        missing = self._missingData
        if gaze[0] == missing or gaze[1] == missing:
            return False
        if pupil == missing or pupil is None:
            return False
        try:
            px = float(pupil)
            gx = float(gaze[0])
            gy = float(gaze[1])
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(px) and math.isfinite(gx) and math.isfinite(gy)):
            return False
        if px <= 0.0:
            return False
        # Reject gaze far outside the stimulus display (common blink artifact).
        margin = 200.0
        width = 2.0 * self._halfWidthPix
        height = 2.0 * self._halfHeightPix
        if gx < -margin or gx > width + margin or gy < -margin or gy > height + margin:
            return False
        return True

    def _readGazePix(self, win):
        '''Return the latest gaze position in PsychoPy pixel coordinates, or None.'''
        tracker = self._tracker
        if tracker is None:
            return None
        try:
            sample = tracker.getNewestSample()
            if sample is None:
                return None

            gazes = []
            if sample.isLeftSample():
                left = sample.getLeftEye()
                if self._eyeGazeValid(left):
                    gazes.append(left.getGaze())
            if sample.isRightSample():
                right = sample.getRightEye()
                if self._eyeGazeValid(right):
                    gazes.append(right.getGaze())
            if not gazes:
                return None

            if len(gazes) == 2:
                gx = 0.5 * (float(gazes[0][0]) + float(gazes[1][0]))
                gy = 0.5 * (float(gazes[0][1]) + float(gazes[1][1]))
            else:
                gx = float(gazes[0][0])
                gy = float(gazes[0][1])

            if self._pendingLatency is not None:
                self._pendingLatency[1] = sample.getTime()
            x = gx - self._halfWidthPix
            y = self._halfHeightPix - gy
            return (x, y)
        except Exception:
            return None

    def _buildTunnelOpacityMap(self, tex_size, overlay_size_pix, ppd_xy):
        '''Radial raised-cosine edge in visual degrees (isotropic aperture).'''
        ppd_h, ppd_v = ppd_xy
        radius_deg = self.tunnelVisibleDiameterDegrees / 2.0
        width_deg = float(self.transitionWidthDegrees)
        center = (tex_size - 1) / 2.0
        yy, xx = np.mgrid[0:tex_size, 0:tex_size]
        scale = overlay_size_pix / float(tex_size)
        x_deg = (xx - center) * scale / ppd_h
        y_deg = (yy - center) * scale / ppd_v
        r_deg = np.sqrt(x_deg ** 2 + y_deg ** 2)
        # Positive outside the clear aperture -> opaque.
        return self._raisedCosineOpacity(r_deg - radius_deg, width_deg)

    def _initPerRunStimulus(self, win, ppd_xy):
        if hasattr(self, 'tunnelEdgeSigmaDegrees'):
            delattr(self, 'tunnelEdgeSigmaDegrees')
        ppd_h, ppd_v = ppd_xy
        transition_pix = max(
            0.0,
            float(self.transitionWidthDegrees) * max(ppd_h, ppd_v),
        )
        # Mask is centered on gaze. Worst case: gaze at one corner, cover the
        # opposite corner → full screen diagonal. Extra margin matches the
        # off-screen tolerance in _eyeGazeValid so extreme samples still occlude.
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
            '--> Contrast Dots with Tunnel Mask: tunnel diameter = {d:g}°, '
            'raised-cosine transition = {w:g}° '
            '(mask overlay {s} px, ppd_h={h:.2f} ppd_v={v:.2f}).'.format(
                d=float(self.tunnelVisibleDiameterDegrees),
                w=float(self.transitionWidthDegrees),
                s=overlay_size_pix,
                h=ppd_h,
                v=ppd_v,
            )
        )
        if self._tracker is None:
            print(self._trackerInactiveMessage())
            self._gazeMaskWarningShown = True


    def _refreshTunnelMask(self):
        '''Rebuild the gaze-mask opacity from the current tunnelVisibleDiameterDegrees.'''
        if getattr(self, '_gazeMask', None) is None:
            return
        opacity = self._buildTunnelOpacityMap(
            int(self._tunnelTexSize),
            float(self._tunnelOverlaySizePix),
            self._tunnelPpdXy,
        )
        self._gazeMask.mask = opacity

    def _initGazeReadCache(self, win):
        '''
        Resolve everything the per-frame gaze read needs once, so the work that
        happens between reading gaze and flipping stays as small as possible.
        '''
        self._tracker = getattr(self, '_elTracker', None)
        self._halfWidthPix = win.size[0] / 2.0
        self._halfHeightPix = win.size[1] / 2.0
        try:
            import pylink
            self._missingData = pylink.MISSING_DATA
        except ImportError:
            self._missingData = -32768

        self._gazeLatencyRecords = []
        self._gazeLatencySummary = {}
        self._pendingLatency = None
        self._lastFlipTime = None
        self._gazeLatencyActive = bool(self.logGazeLatency) and self._tracker is not None
        if bool(self.logGazeLatency) and self._tracker is None:
            print('*** Gaze latency logging was requested, but EyeLink is not active. Skipping.')

    def _recordFlipTime(self):
        '''Runs immediately after the buffer swap, so it timestamps the presented frame.'''
        self._lastFlipTime = core.getTime()

    def _closePendingLatencyRecord(self):
        '''Complete the previous frame's record now that its flip time is known.'''
        pending = self._pendingLatency
        self._pendingLatency = None
        if pending is None or self._lastFlipTime is None:
            return
        pending[4] = self._lastFlipTime
        self._gazeLatencyRecords.append(pending)

    def _renderDotsFrame(self, win, dots):
        dots.draw()
        if self._gazeLatencyActive:
            self._closePendingLatencyRecord()
            win.callOnFlip(self._recordFlipTime)
            # [readTime, sampleTimeMs, gazeX, gazeY, flipTime]
            self._pendingLatency = [core.getTime(), None, None, None, None]
        gaze = self._readGazePix(win)
        if gaze is not None:
            self._lastGaze = gaze
            if self._pendingLatency is not None:
                self._pendingLatency[2] = gaze[0]
                self._pendingLatency[3] = gaze[1]
        elif not self._gazeMaskWarningShown and self._tracker is not None:
            print(self._gazeMissingMessage())
            self._gazeMaskWarningShown = True
        # Always draw the mask at the last good gaze (hold through blinks / missing data).
        self._gazeMask.pos = self._lastGaze
        self._gazeMask.draw()

    def _writeGazeLatencyLog(self):
        records = [r for r in self._gazeLatencyRecords if r[4] is not None and r[1] is not None]
        if not records:
            return None

        logDir = getattr(self, '_okrLogDir', None)
        logDir = Path.cwd() if logDir is None else Path(logDir)
        logDir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logPath = logDir / ('{name}_GazeLatency_{stamp}.txt'.format(
            name=self.protocolName, stamp=stamp,
        ))

        readTimes = np.array([r[0] for r in records], dtype=float)
        sampleTimesMs = np.array([r[1] for r in records], dtype=float)
        flipTimes = np.array([r[4] for r in records], dtype=float)

        readToFlipMs = (flipTimes - readTimes) * 1000.0
        # EyeLink sample times and the stimulus PC clock have different origins.
        # The smallest observed (readTime - sampleTime) is the best estimate of a
        # zero-age sample, so referencing to it gives sample age without a
        # separate clock-sync handshake.
        readMinusSampleMs = readTimes * 1000.0 - sampleTimesMs
        sampleAgeMs = readMinusSampleMs - readMinusSampleMs.min()
        totalMs = sampleAgeMs + readToFlipMs

        droppedFrames = 0
        if self._FR:
            framePeriodMs = 1000.0 / self._FR
            flipIntervalsMs = np.diff(flipTimes) * 1000.0
            # Gaps longer than a few frames are epoch boundaries (fixation cross,
            # inter-stimulus interval), not dropped frames.
            withinEpoch = flipIntervalsMs < 10.0 * framePeriodMs
            droppedFrames = int(np.count_nonzero(
                withinEpoch & (flipIntervalsMs > 1.5 * framePeriodMs)
            ))

        def stats(values):
            return {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'p95': float(np.percentile(values, 95)),
                'max': float(np.max(values)),
            }

        self._gazeLatencySummary = {
            'frames': int(len(records)),
            'frameRateHz': float(self._FR) if self._FR else None,
            'droppedFrames': droppedFrames,
            'sampleAgeMs': stats(sampleAgeMs),
            'readToFlipMs': stats(readToFlipMs),
            'totalMs': stats(totalMs),
        }

        headerLines = [
            '# Gaze-Contingent Latency Log',
            '# StimulusName: Bassoon {name}'.format(name=self.protocolName),
            '# FrameRateHz: {fr:.3f}'.format(fr=self._FR) if self._FR else '# FrameRateHz: NA',
            '# Frames: {n}'.format(n=len(records)),
            '# DroppedFrames (flip interval > 1.5 frames): {n}'.format(n=droppedFrames),
            '# sampleAgeMs: age of the newest EyeLink sample when it was read,',
            '#   referenced to the freshest sample observed in this run (link + tracker delay).',
            '# readToFlipMs: gaze read -> buffer swap that presented the updated mask.',
            '# totalMs: sampleAgeMs + readToFlipMs. Display pixel response is NOT included;',
            '#   measure that with a photodiode if an absolute number is needed.',
            '# sampleTimeMs is EyeLink tracker time, so rows can be aligned to the EDF/ASC.',
            'frame\tsampleTimeMs\tsampleAgeMs\treadToFlipMs\ttotalMs\tgazeX\tgazeY',
        ]
        rowLines = []
        for i, record in enumerate(records):
            rowLines.append('\t'.join([
                str(i),
                '{:.0f}'.format(sampleTimesMs[i]),
                '{:.3f}'.format(sampleAgeMs[i]),
                '{:.3f}'.format(readToFlipMs[i]),
                '{:.3f}'.format(totalMs[i]),
                'NA' if record[2] is None else '{:.2f}'.format(record[2]),
                'NA' if record[3] is None else '{:.2f}'.format(record[3]),
            ]))
        logPath.write_text('\n'.join(headerLines + rowLines) + '\n', encoding='utf-8')

        summary = self._gazeLatencySummary
        print('--> Wrote gaze latency log:', logPath)
        print(
            '    Frames: {n}   Dropped: {d}   Frame rate: {fr:.1f} Hz'.format(
                n=summary['frames'], d=summary['droppedFrames'],
                fr=summary['frameRateHz'] if summary['frameRateHz'] else float('nan'),
            )
        )
        for label, key in (
            ('Sample age', 'sampleAgeMs'),
            ('Read to flip', 'readToFlipMs'),
            ('Total', 'totalMs'),
        ):
            s = summary[key]
            print(
                '    {label:<13} median {med:6.2f} ms   mean {mean:6.2f} ms   '
                'p95 {p95:6.2f} ms   max {mx:6.2f} ms'.format(
                    label=label, med=s['median'], mean=s['mean'],
                    p95=s['p95'], mx=s['max'],
                )
            )
        return logPath

    def _teardownPerRunStimulus(self):
        if getattr(self, '_gazeLatencyActive', False):
            self._closePendingLatencyRecord()
            try:
                self._writeGazeLatencyLog()
            except Exception as e:
                print('*** Could not write the gaze latency log (' + str(e) + ').')
        # Per-frame records are dropped so they do not bloat the saved experiment file.
        self._gazeLatencyRecords = []
        self._pendingLatency = None
        self._gazeMask = None
        self._tracker = None
