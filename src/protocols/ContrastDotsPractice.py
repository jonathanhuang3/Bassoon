# -*- coding: utf-8 -*-
"""
Contrast Dots Practice is a simplified Contrast Dots run for training subjects
on the attention-probe task (spacebar) before the real multi-contrast stimulus.

White dots at 100% contrast move upward by default, with the same size, speed,
and probe settings as Contrast Dots.

After a completed run, shows a first-level jar celebration (single cup):
starts from nothing, cup animates in, Level Complete!, confetti, and
optional replay. The staircase later stacks a second cup on top.
"""
import math
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
        # End-of-run celebration (castle + sound).
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


    def _makeTone(self, freqs, noteDur=0.10, gapDur=0.02, volume=0.55):
        try:
            from psychopy import sound
        except Exception:
            return None
        sampleRate = 44100
        pieces = []
        for i, freq in enumerate(freqs):
            n = max(2, int(sampleRate * noteDur))
            t = np.linspace(0.0, noteDur, n, endpoint=False)
            env = np.ones(n)
            attack = max(1, int(0.008 * sampleRate))
            release = max(1, int(0.035 * sampleRate))
            if attack + release > n:
                attack = max(1, n // 3)
                release = max(1, n - attack)
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


    def _makeJarBundle(self, win, jarX, jarY, jarW, jarH, rimW, rimH, ppd_h, ppd_v,
                       glassColor, fillColor, fillLevel=1.0):
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
        return {
            'x': jarX, 'y': jarY, 'w': jarW, 'h': jarH,
            'glass': jarGlass, 'outline': jarOutline, 'rim': jarRim, 'fill': jarFill,
            'fillLevel': float(fillLevel),
        }


    def _placeJarBundle(self, jar, ox=0.0, oy=0.0, ppd_v=1.0, fillLevel=None):
        x = jar['x'] + ox
        y = jar['y'] + oy
        jar['glass'].pos = (x, y)
        jar['outline'].pos = (x, y)
        jar['rim'].pos = (x, y + jar['h'] / 2.0 + jar['rim'].height / 2.0 - 0.1 * ppd_v)
        level = jar['fillLevel'] if fillLevel is None else float(fillLevel)
        level = max(0.0, min(1.0, level))
        fillH = max(0.01, (jar['h'] - 0.35 * ppd_v) * level)
        jar['fill'].height = fillH
        jar['fill'].pos = (x, y - jar['h'] / 2.0 + fillH / 2.0 + 0.1 * ppd_v)


    def _drawJarBundle(self, jar, ox=0.0, oy=0.0, ppd_v=1.0, fillLevel=None):
        self._placeJarBundle(jar, ox=ox, oy=oy, ppd_v=ppd_v, fillLevel=fillLevel)
        jar['glass'].draw()
        jar['fill'].draw()
        jar['outline'].draw()
        jar['rim'].draw()


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
        First-level celebration: blank start, one cup drops in, Level Complete!,
        fill, outward confetti. Returns 'continue', 'replay', or 'quit'.
        '''
        self.getFR(win)
        fr = float(getattr(self, '_FR', 60) or 60)
        ppd_h, ppd_v = self.getPixPerDegXY(win.monitor)
        # Same cup size as staircase base so level 2 can stack on the same visual language.
        jarW, jarH = 4.6 * ppd_h, 5.8 * ppd_v
        rimW, rimH = 5.5 * ppd_h, 0.85 * ppd_v
        jarX, jarY = 0.0, -1.2 * ppd_v
        glassColor = [0.92, 0.86, 0.78]
        fillColor = [1.0, 0.12, 0.02]
        candyColors = [
            [1.0, -0.15, -0.15],
            [1.0, 0.55, -0.55],
            [-0.15, 0.65, 1.0],
            [0.95, 0.9, -0.65],
            [0.65, -0.35, 0.95],
        ]
        fillTarget = float(min(1.0, max(0.0, fillFraction)))
        if fillTarget <= 0.0 and hits <= 0:
            fillTarget = 0.35  # modest fill so finishing practice still feels rewarding
        win.color = [-0.88, -0.62, -0.42]

        jar = self._makeJarBundle(
            win, jarX, jarY, jarW, jarH, rimW, rimH, ppd_h, ppd_v,
            glassColor, fillColor, fillLevel=0.0,
        )

        titleHeightFinal = 0.85 * ppd_v
        titleText = visual.TextStim(
            win, text='Level Complete!',
            pos=(0.0, jarY - jarH / 2.0 - 1.35 * ppd_v),
            units='pix', height=titleHeightFinal,
            color=[0.98, 0.92, 0.72], bold=True, opacity=0.0,
        )
        tipText = visual.TextStim(
            win, text='Space = continue      R = practice again',
            pos=(0.0, jarY - jarH / 2.0 - 2.35 * ppd_v),
            units='pix', height=0.45 * ppd_v,
            color=[0.92, 0.8, 0.62], opacity=0.0,
        )

        rng = np.random.default_rng(int((hits + 1) * 997 + total * 13) or 42)
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

        self._celebrationSoundRefs = []
        yayTone = self._keepCelebrationSound(
            self._loadCelebrationFileSound(
                _PRACTICE_YAY_SOUND, volume=0.55, name='practiceYay',
            )
        )
        if yayTone is None:
            yayTone = self._keepCelebrationSound(
                self._makeTone(
                    [523.25, 659.25, 783.99],
                    noteDur=0.12, gapDur=0.03, volume=0.50,
                )
            )

        def checkKeys():
            keys = [k.lower() for k in event.getKeys()]
            if not keys:
                return None
            if 'q' in keys:
                self._stoppedEarly = 1
                return 'quit'
            if 'r' in keys:
                return 'replay'
            if 'space' in keys or 'return' in keys or 'enter' in keys:
                return 'continue'
            return None

        def drawScene(
            showJar=False, jarOy=0.0, jarFill=0.0,
            showConfetti=False, confettiT=0.0, shakeX=0.0,
            titleScale=0.0, showTip=False,
        ):
            if showJar:
                self._drawJarBundle(
                    jar, ox=shakeX, oy=jarOy, ppd_v=ppd_v, fillLevel=jarFill,
                )
            if showConfetti:
                for c in confetti:
                    c['stim'].pos = (
                        c['x'] + c['vx'] * confettiT + shakeX,
                        c['y'] + c['vy'] * confettiT - 6.0 * ppd_v * confettiT * confettiT,
                    )
                    c['stim'].draw()
            if titleScale > 0.01:
                s = max(0.01, min(1.15, float(titleScale)))
                titleText.height = titleHeightFinal * s
                titleText.opacity = min(1.0, s / 0.85)
                titleText.draw()
            if showTip:
                tipText.opacity = 1.0
                tipText.draw()

        event.clearEvents()
        self._playPreparedSound(yayTone)

        # 0) Start from nothing.
        blankFrames = max(4, int(round(0.20 * fr)))
        for _ in range(blankFrames):
            drawScene()
            win.flip()
            action = checkKeys()
            if action is not None:
                return action

        # 1) First-level cup drops in empty.
        dropFrames = max(10, int(round(0.50 * fr)))
        dropStartOy = -4.0 * ppd_v
        for f in range(dropFrames):
            u = (f + 1) / float(dropFrames)
            eased = 1.0 - (1.0 - u) ** 2
            bounce = 0.0
            if u > 0.82:
                bounce = 0.18 * ppd_v * math.sin((u - 0.82) / 0.18 * math.pi)
            jarOy = dropStartOy * (1.0 - eased) - bounce
            drawScene(showJar=True, jarOy=jarOy, jarFill=0.0)
            win.flip()
            action = checkKeys()
            if action is not None:
                return action

        # Brief settle shake.
        settleFrames = max(6, int(round(0.25 * fr)))
        for f in range(settleFrames):
            t = (f + 1) / float(settleFrames)
            shake = (0.12 * ppd_h) * math.sin(t * 8.0 * math.pi) * (1.0 - t)
            drawScene(showJar=True, jarFill=0.0, shakeX=shake)
            win.flip()
            action = checkKeys()
            if action is not None:
                return action

        # 2) "Level Complete!" pops in below the cup.
        titleFrames = max(10, int(round(0.45 * fr)))
        for f in range(titleFrames):
            u = (f + 1) / float(titleFrames)
            eased = 1.0 - (1.0 - u) ** 3
            if u < 0.85:
                scale = 1.12 * eased / 0.85
            else:
                settle = (u - 0.85) / 0.15
                scale = 1.12 + (1.0 - 1.12) * settle
            drawScene(showJar=True, jarFill=0.0, titleScale=scale)
            win.flip()
            action = checkKeys()
            if action is not None:
                return action

        # 3) Cup fills (probe catch rate).
        fillFrames = max(12, int(round(0.55 * fr)))
        for f in range(fillFrames):
            t = (f + 1) / float(fillFrames)
            drawScene(showJar=True, jarFill=t * fillTarget, titleScale=1.0)
            win.flip()
            action = checkKeys()
            if action is not None:
                return action

        # 4) Confetti explodes outward.
        flourishFrames = max(12, int(round(0.9 * fr)))
        holdConfettiT = 0.85
        for f in range(flourishFrames):
            t = (f + 1) / float(flourishFrames)
            shake = (0.18 * ppd_h) * math.sin(t * 10.0 * math.pi) * (1.0 - t)
            drawScene(
                showJar=True, jarFill=fillTarget,
                showConfetti=True, confettiT=t * 0.9, shakeX=shake,
                titleScale=1.0, showTip=True,
            )
            win.flip()
            action = checkKeys()
            if action is not None:
                return action

        # Hold / wait for continue or replay.
        event.clearEvents()
        while True:
            drawScene(
                showJar=True, jarFill=fillTarget,
                showConfetti=True, confettiT=holdConfettiT,
                titleScale=1.0, showTip=True,
            )
            win.flip()
            action = checkKeys()
            if action is not None:
                return action
