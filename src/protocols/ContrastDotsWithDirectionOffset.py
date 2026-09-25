# -*- coding: utf-8 -*-
"""
Contrast Dots with Direction Offset is Contrast Dots specialized for testing
OKR across cardinal directions and small diagonal / torsional offsets.

Example: directions = [90], directionOffsets = [0, 2, 5] yields epochs at
90°, 92°, 88°, 95°, and 85° (0° offset once; each nonzero offset as ±).
"""
from protocols.ContrastDots import ContrastDots


class ContrastDotsWithDirectionOffset(ContrastDots):
    _okrSyncsTrialClock = True

    def __init__(self):
        super().__init__()
        self.protocolName = 'ContrastDotsWithDirectionOffset'
        # Default to full positive contrast; additional contrast levels can still be listed.
        self.contrasts = [1.0]
        # Up only by default (90°); add 270 etc. to test other bases.
        self.directions = [90.0]
        # Magnitude list in degrees. 0 keeps the exact base direction; each nonzero
        # value expands to both +offset and -offset from every base direction.
        self.directionOffsets = [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0]


    def _stimulusTitle(self):
        return 'Contrast Dots with Direction Offset'


    def internalValidation(self):
        tf = True
        errorMessage = []
        offsets = getattr(self, 'directionOffsets', None)
        if offsets is None or len(offsets) == 0:
            tf = False
            errorMessage.append('Direction Offsets must contain at least one value.')
        else:
            for offset in offsets:
                try:
                    float(offset)
                except (TypeError, ValueError):
                    tf = False
                    errorMessage.append('Direction Offset values must be numbers (degrees).')
                    break
        parentTf, parentErrors = super().internalValidation()
        tf = tf and parentTf
        errorMessage += parentErrors
        if tf and len(self._expandedDirectionConditions()) == 0:
            tf = False
            errorMessage.append('Direction Offsets produced no motion directions.')
        return tf, errorMessage


    def _signedDirectionOffsets(self):
        '''
        Expand configured offset magnitudes into signed offsets.
        0 stays 0; each nonzero magnitude becomes +mag and -mag (deduplicated).
        '''
        offsets = getattr(self, 'directionOffsets', [0.0]) or [0.0]
        signed = []
        seen = set()
        for raw in offsets:
            value = float(raw)
            if abs(value) < 1e-12:
                candidates = [0.0]
            else:
                mag = abs(value)
                candidates = [mag, -mag]
            for candidate in candidates:
                key = round(candidate, 6)
                if key in seen:
                    continue
                seen.add(key)
                signed.append(candidate)
        return signed


    def _expandedDirectionConditions(self):
        '''
        Return one condition dict per unique actual motion angle:
        baseDirection, directionOffset (signed), direction (actual angle).
        '''
        conditions = []
        seenActual = set()
        for base in self._baseDirectionPool():
            for signedOffset in self._signedDirectionOffsets():
                actual = self.deg0to360(base + signedOffset)
                key = round(actual, 6)
                if key in seenActual:
                    continue
                seenActual.add(key)
                conditions.append({
                    'baseDirection': base,
                    'directionOffset': signedOffset,
                    'direction': actual,
                })
        return conditions


    def _baseDirectionPool(self):
        '''Cardinal / base directions before applying offsets.'''
        pool = getattr(self, 'directions', None)
        if not pool:
            return [self.deg0to360(self.direction)]
        return [self.deg0to360(d) for d in pool]


    def _directionPool(self):
        '''Actual motion angles after applying directionOffsets (used for timing / run limits).'''
        return [condition['direction'] for condition in self._expandedDirectionConditions()]


    def _buildFactorialEpochPairs(self):
        '''Every contrast × (base direction × signed offset), repeated per stimulusRep.'''
        conditions = self._expandedDirectionConditions()
        pairs = []
        for _ in range(self.stimulusReps):
            for contrast in self.contrasts:
                for condition in conditions:
                    pairs.append({
                        'contrast': contrast,
                        'direction': condition['direction'],
                        'baseDirection': condition['baseDirection'],
                        'directionOffset': condition['directionOffset'],
                    })
        return pairs


    def _offsetDirectionLabel(self, baseDirection, signedOffset):
        baseLabel = self._directionLabel(baseDirection)
        if abs(float(signedOffset)) < 1e-12:
            return baseLabel
        return '{base}{offset:+g}'.format(base=baseLabel, offset=float(signedOffset))


    def _epochDirectionExtra(self, epoch):
        base = epoch.get('baseDirection', epoch.get('direction'))
        offset = epoch.get('directionOffset', 0.0)
        return ' | base {b:g}\u00b0 offset {o:+g}\u00b0'.format(b=base, o=offset)


    def _appendOkrContrastBlock(self, events, counter, blockIndex, contrast, direction, startTime, endTime,
                                baseDirection=None, directionOffset=None):
        eventIndex = self._nextOkrEventIndex(counter)
        if baseDirection is None:
            baseDirection = direction
        if directionOffset is None:
            directionOffset = 0.0
        directionLabel = self._offsetDirectionLabel(baseDirection, directionOffset)
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
        self._recordOkrSessionEvent(
            'ContrastBlock', startTime, endTime,
            direction=directionLabel,
            contrastLevel=contrast,
            blockOrEpochIndex=blockIndex,
            dotColor=dotColor,
            usePersistentDots=int(self._usePersistentDots()),
            isAnchor100=isAnchor100,
        )
        self._sendOkrEyeLinkMessage(
            'OKR ContrastBlock B{bi} contrast {c:g} dir {d} '
            'base {base:g} offset {off:+g} actual {act:g} {t0:.3f}-{t1:.3f}'.format(
                bi=blockIndex,
                c=contrast,
                d=directionLabel,
                base=baseDirection,
                off=directionOffset,
                act=direction,
                t0=startTime,
                t1=endTime,
            ),
        )


    def _writeOkrLogFile(self, events):
        path = super()._writeOkrLogFile(events)
        if path is None:
            return None
        offsetText = ', '.join('{g:g}'.format(g=float(o)) for o in getattr(self, 'directionOffsets', []))
        baseText = ', '.join('{g:g}'.format(g=d) for d in self._baseDirectionPool())
        actualText = ', '.join('{g:g}'.format(g=d) for d in self._directionPool())
        extra = [
            '# BaseDirectionsDeg: {dirs}'.format(dirs=baseText),
            '# DirectionOffsetsDeg: {offs}'.format(offs=offsetText),
            '# ActualDirectionsDeg: {dirs}'.format(dirs=actualText),
            '# Note: Nonzero offsets are applied as both +offset and -offset from each base direction.',
        ]
        existing = path.read_text(encoding='utf-8').splitlines()
        insertAt = 0
        for i, line in enumerate(existing):
            if not line.startswith('#'):
                insertAt = i
                break
            insertAt = i + 1
        merged = existing[:insertAt] + extra + existing[insertAt:]
        path.write_text('\n'.join(merged) + '\n', encoding='utf-8')
        return path
