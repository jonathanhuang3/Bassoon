# -*- coding: utf-8 -*-
'''
Helpers for registering Bassoon stimulus monitors.
'''
import os

from psychopy import monitors

# Built-in monitor definitions for Bassoon rigs.
# sizePix is filled in automatically from the configured screen resolution.
BASSOON_MONITORS = [
    {
        'name': 'Dell E2720HS',
        'widthCm': 59.6,
        'heightCm': 33.8,
        'distanceCm': 70.0,
    },
]


def get_screen_size_pixels(screen_index=0):
    '''
    Return [widthPx, heightPx] for the requested display index.
    Falls back to the primary display if enumeration fails.
    '''
    if os.name == 'nt':
        try:
            import ctypes
            from ctypes import wintypes

            monitor_sizes = []

            def _callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                rect = lprcMonitor.contents
                monitor_sizes.append([
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                ])
                return True

            monitor_enum_proc = ctypes.WINFUNCTYPE(
                ctypes.c_int,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.POINTER(wintypes.RECT),
                ctypes.c_double,
            )
            ctypes.windll.user32.EnumDisplayMonitors(
                0, 0, monitor_enum_proc(_callback), 0,
            )
            if 0 <= screen_index < len(monitor_sizes):
                return monitor_sizes[screen_index]
        except Exception:
            pass

    try:
        from tkinter import Tk
        root = Tk()
        root.withdraw()
        size = [root.winfo_screenwidth(), root.winfo_screenheight()]
        root.destroy()
        return size
    except Exception:
        return [1920, 1080]


def register_monitor(name, width_cm, distance_cm, size_pix):
    '''Create or update a PsychoPy monitor calibration file.'''
    monitor = monitors.Monitor(name)
    monitor.setWidth(width_cm)
    monitor.setDistance(distance_cm)
    monitor.setSizePix(size_pix)
    monitor.save()
    return monitor


def save_monitor_gamma(monitor_name, gamma):
    '''Persist gamma on a PsychoPy monitor profile for visual.Window linearization.'''
    gamma = float(gamma)
    monitor = monitors.Monitor(monitor_name)
    monitor.setGamma(gamma)
    monitor.save()
    return gamma


def ensure_bassoon_monitors(screen_index=0, definitions=None):
    '''
    Register predefined Bassoon monitors if they are missing or incomplete.
    Uses the native resolution of the configured stimulus screen for sizePix.
    '''
    if definitions is None:
        definitions = BASSOON_MONITORS

    size_pix = get_screen_size_pixels(screen_index)
    registered = []

    for definition in definitions:
        name = definition['name']
        existing = monitors.Monitor(name)
        existing_size_pix = list(existing.getSizePix() or [])
        needs_save = (
            existing.getDistance() != definition['distanceCm']
            or existing.getWidth() != definition['widthCm']
            or existing_size_pix != list(size_pix)
        )

        if needs_save:
            register_monitor(
                name=name,
                width_cm=definition['widthCm'],
                distance_cm=definition['distanceCm'],
                size_pix=size_pix,
            )
            print(
                '--> Registered monitor {name}: {w}x{h} cm, '
                '{d} cm viewing distance, {px_w}x{px_h} px'.format(
                    name=name,
                    w=definition['widthCm'],
                    h=definition['heightCm'],
                    d=definition['distanceCm'],
                    px_w=size_pix[0],
                    px_h=size_pix[1],
                )
            )
        registered.append(name)

    return registered
