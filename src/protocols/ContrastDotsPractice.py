# -*- coding: utf-8 -*-
"""
Contrast Dots Practice is a simplified Contrast Dots run for training subjects
on the attention-probe task (spacebar) before the real multi-contrast stimulus.

White dots at 100% contrast move upward by default, with the same size, speed,
and probe settings as Contrast Dots.

After a completed run, shows a short Candy-Crush-style jar celebration: one
candy drop per caught probe, a yay SFX at the start, confetti, and
optional replay.
"""
import math
import time
from pathlib import Path

import numpy as np
from psychopy import event, visual

from protocols.ContrastDots import ContrastDots

_PRACTICE_SOUND_DIR = Path(__file__).resolve().parent / 'sounds'
_PRACTICE_YAY_SOUND = _PRACTICE_SOUND_DIR / 'practice_yay.mp3'


class ContrastDotsPractice(ContrastDots):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsPractice'
        # Practice: full white only, up motion and probe task (same defaults as Contrast Dots otherwise).
        self.contrasts = [1.0]
        self.directions = [90.0]  # up only by default
        self.attentionProbe = True
        # Give subjects time to press after the brief red flash (default field lifetime is short).
        self.attentionProbeResponseWindowSec = 1.0
        # End-of-run celebration (jar fill + sound).
        self.practiceScoreCelebration = True
        self._practiceReplayRequested = False


    def _stimulusTitle(self):
        return 'Contrast Dots Practice'


    def run(self, win, informationWin):
        '''Allow R on the celebration screen to replay practice in the same window.'''
        while True:
            self._practiceReplayRequested = False
            super().run(win, informationWin)
            if getattr(self, '_stoppedEarly', 0):
                break
            if not self._practiceReplayRequested:
                break
            print('--> Practice again!')


    def _attentionProbeScore(self):
        '''Return (hits, totalProbes, falseAlarms, fillFraction 0..1).'''
        events = getattr(self, '_attentionEvents', None) or []
        ends = [e for e in events if e.get('eventType') == 'ProbeEnd']
        if ends:
            hits = sum(1 for e in ends if int(e.get('hit', 0) or 0) > 0)
            total = len(ends)
        else:
            spawns = [e for e in events if e.get('eventType') == 'ProbeSpawn']
            responses = [e for e in events if e.get('eventType') == 'ProbeResponse']
            total = len(spawns)
            hitNums = {e.get('probeNumber') for e in responses if e.get('probeNumber') != 'NA'}
            hits = len(hitNums)
        falseAlarms = sum(1 for e in events if e.get('eventType') == 'ProbeFalseAlarm')
        if total <= 0:
            return 0, 0, falseAlarms, 0.0
        fill = hits / float(total)
        return hits, total, falseAlarms, float(min(1.0, max(0.0, fill)))


    def _starCount(self, hits, total):
        if total <= 0:
            return 0
        if hits >= total:
            return 3
        if hits * 3 >= total * 2:  # at least ~2/3
            return 2
        if hits * 3 >= total:  # at least ~1/3
            return 1
        return 0


    def _makeTone(self, freqs, noteDur=0.10, gapDur=0.02, volume=0.55):
        try:
            from psychopy import sound
        except Exception:
            return None
        sampleRate = 44100
        pieces = []
        for i, freq in enumerate(freqs):
            n = int(sampleRate * noteDur)
            t = np.linspace(0.0, noteDur, n, endpoint=False)
            env = np.ones(n)
            attack = max(1, int(0.008 * sampleRate))
            release = max(1, int(0.035 * sampleRate))
            env[:attack] *= np.linspace(0.0, 1.0, attack)
            env[-release:] *= np.linspace(1.0, 0.0, release)
            wave = volume * np.sin(2 * np.pi * freq * t)
            wave += 0.16 * volume * np.sin(2 * np.pi * 3 * freq * t)
            wave *= env
            pieces.append(wave)
            pieces.append(np.zeros(int(gapDur * sampleRate)))
        mono = np.concatenate(pieces).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        try:
            return sound.Sound(value=stereo, sampleRate=sampleRate, stereo=True, name='practiceTone')
        except Exception as err:
            print('*** Practice tone failed:', err)
            return None


    def _keepCelebrationSound(self, snd):
        if snd is None:
            return None
        refs = getattr(self, '_celebrationSoundRefs', None)
        if refs is None:
            refs = []
            self._celebrationSoundRefs = refs
        refs.append(snd)
        return snd


    def _loadCelebrationFileSound(self, path, volume=0.55, name='practiceFileSound'):
        '''Load an mp3/wav celebration sound; return Sound or None.'''
        try:
            from psychopy import sound
        except Exception:
            return None
        path = Path(path)
        if not path.is_file():
            print('*** Practice celebration sound missing:', path)
            return None
        try:
            snd = sound.Sound(value=str(path), name=name)
            try:
                snd.setVolume(float(volume))
            except Exception:
                pass
            return snd
        except Exception as err:
            print('*** Practice celebration file sound failed ({0}): {1}'.format(path.name, err))
            return None


    def _playPreparedSound(self, snd):
        if snd is None:
            return
        try:
            snd.stop()
            snd.play()
        except Exception as err:
            print('*** Practice celebration tone play failed:', err)


    def _afterProtocolComplete(self, win):
        self._practiceReplayRequested = False
        if not getattr(self, 'practiceScoreCelebration', True):
            return
        if not getattr(self, 'attentionProbe', False):
            return
        hits, total, falseAlarms, fill = self._attentionProbeScore()
        print(
            '--> Practice score: caught {h}/{t} probes '
            '(false alarms={fa}, jar fill={f:.0%})'.format(
                h=hits, t=total, fa=falseAlarms, f=fill,
            )
        )
        try:
            action = self._showPracticeJarCelebration(win, hits, total, falseAlarms, fill)
        except Exception as err:
            print('*** Practice celebration screen failed:', err)
            action = 'continue'
        if action == 'replay':
            self._practiceReplayRequested = True


    def _showPracticeJarCelebration(self, win, hits, total, falseAlarms, fillFraction):
        '''
        Discrete candy drops with yay SFX at the start, confetti, continue/replay.
        Returns 'continue', 'replay', or 'quit'.
        '''
        win.color = [-0.88, -0.62, -0.42]
        ppd_h, ppd_v = self.getPixPerDegXY(win.monitor)
        fr = float(getattr(self, '_FR', 60) or 60)
        jarW, jarH = 5.5 * ppd_h, 8.0 * ppd_v
        jarX, jarY = 0.0, -0.8 * ppd_v
        rimW, rimH = 6.6 * ppd_h, 1.0 * ppd_v
        glassColor = [0.92, 0.86, 0.78]
        fillColor = [1.0, 0.12, 0.02]
        candyColors = [
            [1.0, -0.15, -0.15],
            [1.0, 0.55, -0.55],
            [-0.15, 0.65, 1.0],
            [0.95, 0.9, -0.65],
            [0.65, -0.35, 0.95],
        ]
        perfect = total > 0 and hits >= total and falseAlarms == 0

        # Soft jar body (filled glass tint) + outline + rim.
        jarGlass = visual.Rect(
            win, width=jarW, height=jarH, pos=(jarX, jarY),
            units='pix', lineColor=None, fillColor=[0.15, 0.08, 0.02], opacity=0.55,
        )
        jarOutline = visual.Rect(
            win, width=jarW, height=jarH, pos=(jarX, jarY),
            units='pix', lineColor=glassColor, fillColor=None, lineWidth=8,
        )
        jarRim = visual.Rect(
            win, width=rimW, height=rimH,
            pos=(jarX, jarY + jarH / 2.0 + rimH / 2.0 - 0.1 * ppd_v),
            units='pix', lineColor=glassColor, fillColor=[-0.2, -0.25, -0.3],
            lineWidth=6,
        )
        jarFill = visual.Rect(
            win, width=jarW - 0.45 * ppd_h, height=0.01,
            pos=(jarX, jarY - jarH / 2.0),
            units='pix', lineColor=None, fillColor=fillColor, opacity=0.8,
        )

        tipText = visual.TextStim(
            win, text='Space = continue      R = practice again',
            pos=(0.0, -7.5 * ppd_v), units='pix', height=0.55 * ppd_v,
            color=[0.92, 0.8, 0.62],
        )

        dropFrames = max(8, int(round(0.35 * fr)))
        settleSec = 0.05
        self._celebrationSoundRefs = []
        # Freesound "yay" at celebration start (fallback: short chord).
        yayTone = self._keepCelebrationSound(
            self._loadCelebrationFileSound(
                _PRACTICE_YAY_SOUND, volume=0.55, name='practiceYay',
            )
        )
        if yayTone is None:
            yayTone = self._keepCelebrationSound(
                self._makeTone(
                    [659.25, 880.0] if perfect else [523.25, 659.25],
                    noteDur=0.14, gapDur=0.04, volume=0.55,
                )
            )

        # One slot per probe: filled candy if hit, empty ring if miss.
        nSlots = max(total, 1)
        slots = []
        rng = np.random.default_rng(int((hits + 1) * 997 + total * 13))
        for i in range(nSlots):
            cx = jarX + float(rng.uniform(-jarW * 0.28, jarW * 0.28))
            cy = (jarY - jarH / 2.0 + 0.9 * ppd_v) + (jarH - 1.8 * ppd_v) * (i + 0.5) / float(nSlots)
            slots.append({
                'home': (cx, cy),
                'candy': visual.Circle(
                    win, radius=0.42 * min(ppd_h, ppd_v), pos=(cx, cy + jarH),
                    units='pix', fillColor=candyColors[i % len(candyColors)],
                    lineColor=[-0.15, -0.15, -0.15], lineWidth=3,
                ),
                'empty': visual.Circle(
                    win, radius=0.42 * min(ppd_h, ppd_v), pos=(cx, cy),
                    units='pix', fillColor=None,
                    lineColor=[0.45, 0.35, 0.25], lineWidth=3,
                ),
                'filled': False,
            })

        # Confetti burst (shown after candy drops whenever there was at least one hit).
        confetti = []
        for i in range(36):
            confetti.append({
                'stim': visual.Circle(
                    win, radius=0.14 * min(ppd_h, ppd_v),
                    pos=(0.0, 0.0), units='pix',
                    fillColor=candyColors[i % len(candyColors)], lineColor=None,
                ),
                'vx': float(rng.uniform(-10, 10)) * ppd_h,
                'vy': float(rng.uniform(5, 14)) * ppd_v,
                'x': jarX + float(rng.uniform(-0.8, 0.8)) * ppd_h,
                'y': jarY + jarH / 2.0,
            })

        event.clearEvents()

        def drawScene(shakeX=0.0, showConfetti=False, confettiT=0.0, filledCount=0):
            ox = shakeX
            jarGlass.pos = (jarX + ox, jarY)
            jarOutline.pos = (jarX + ox, jarY)
            jarRim.pos = (jarX + ox, jarY + jarH / 2.0 + rimH / 2.0 - 0.1 * ppd_v)
            level = filledCount / float(max(total, 1))
            fillH = max(0.01, (jarH - 0.35 * ppd_v) * level)
            jarFill.height = fillH
            jarFill.pos = (jarX + ox, jarY - jarH / 2.0 + fillH / 2.0 + 0.1 * ppd_v)

            jarGlass.draw()
            jarFill.draw()
            for i, slot in enumerate(slots):
                hx, hy = slot['home']
                slot['empty'].pos = (hx + ox, hy)
                slot['empty'].draw()
                if slot['filled']:
                    slot['candy'].pos = (hx + ox, hy)
                    slot['candy'].draw()
            jarOutline.draw()
            jarRim.draw()
            if showConfetti:
                for c in confetti:
                    c['stim'].pos = (
                        c['x'] + c['vx'] * confettiT + ox,
                        c['y'] + c['vy'] * confettiT - 6.0 * ppd_v * confettiT * confettiT,
                    )
                    c['stim'].draw()
            tipText.draw()

        # Start empty, then yay immediately as the celebration begins.
        drawScene(filledCount=0)
        win.flip()
        if hits > 0:
            self._playPreparedSound(yayTone)
        time.sleep(0.25)

        for hitIndex in range(hits):
            slot = slots[hitIndex]
            hx, hy = slot['home']
            startY = hy + 3.5 * ppd_v
            for f in range(dropFrames):
                u = (f + 1) / float(dropFrames)
                eased = 1.0 - (1.0 - u) ** 2
                # Slight overshoot bounce at the end.
                bounce = 0.0
                if u > 0.85:
                    bounce = 0.25 * ppd_v * math.sin((u - 0.85) / 0.15 * math.pi)
                cy = startY + (hy - startY) * eased - bounce
                slot['candy'].pos = (hx, cy)
                # Temporary draw of falling candy.
                drawScene(filledCount=hitIndex)
                slot['candy'].draw()
                jarOutline.draw()
                jarRim.draw()
                tipText.draw()
                win.flip()
                keys = event.getKeys()
                if keys:
                    if 'q' in keys:
                        self._stoppedEarly = 1
                        return 'quit'
            slot['filled'] = True
            # Brief settle frame.
            drawScene(filledCount=hitIndex + 1)
            win.flip()
            time.sleep(settleSec)

        # Mark empty slots for misses (already drawn as rings).
        for i in range(hits, nSlots):
            slots[i]['filled'] = False

        # Confetti whenever at least one probe was caught.
        showConfetti = hits > 0

        if showConfetti:
            flourishFrames = max(12, int(round(0.9 * fr)))
            for f in range(flourishFrames):
                t = (f + 1) / float(flourishFrames)
                shake = (
                    (0.18 * ppd_h) * math.sin(t * 10.0 * math.pi) * (1.0 - t)
                    if perfect else 0.0
                )
                drawScene(
                    shakeX=shake, showConfetti=True, confettiT=t * 0.9, filledCount=hits,
                )
                win.flip()
                keys = event.getKeys()
                if keys:
                    if 'q' in keys:
                        self._stoppedEarly = 1
                        return 'quit'
                    if 'r' in keys:
                        return 'replay'
                    if 'space' in keys or 'return' in keys or 'enter' in keys:
                        return 'continue'

        # Wait for Space (continue) or R (replay).
        event.clearEvents()
        while True:
            drawScene(filledCount=hits, showConfetti=showConfetti, confettiT=0.85)
            win.flip()
            keys = [k.lower() for k in event.getKeys()]
            if not keys:
                continue
            if 'q' in keys:
                self._stoppedEarly = 1
                return 'quit'
            if 'r' in keys:
                return 'replay'
            if 'space' in keys or 'return' in keys or 'enter' in keys:
                return 'continue'
