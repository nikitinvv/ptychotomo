# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for 3D ptychography."""

import xraylib as xl
import numpy as np
import pyfftw
import shutil
import warnings
warnings.filterwarnings("ignore")

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class Material(object):
    """Material property definitions.

    Attributes
    ----------
    compound : string
        Molecular formula of the material.
    density : scalar
        Density of the compound [g/cm^3].
    energy : scalar
        Illumination energy [keV].
    """

    def __init__(self, compound, density, energy):
        self.compound = compound
        self.density = density
        self.energy = energy

    @property
    def beta(self):
        """Absorption coefficient."""
        return xl.Refractive_Index_Im(self.compound, self.energy, self.density)

    @property
    def delta(self):
        """Decrement of refractive index."""
        return 1 - xl.Refractive_Index_Re(self.compound, self.energy, self.density)


class Object(object):
    """Discrete object represented in a 3D regular grid.

    Attributes
    ----------
    beta : ndarray
        Absorption index.
    delta : ndarray
        Refractive index decrement.
    voxelsize : scalar [cm]
        Size of the voxels in the grid.
    """

    def __init__(self, beta, delta, voxelsize):
        self.beta = beta
        self.delta = delta
        self.voxelsize = voxelsize

    @property
    def shape(self):
        return self.beta.shape

    @property
    def complexform(self):
        return self.delta + 1j * self.beta


class Detector(object):
    """A planar area detector.

    Attributes
    ----------
    numx : int
        Number of horizontal pixels.
    numy : int
        Number of vertical pixels.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Probe(object):
    """Illumination probe represented on a 2D regular grid.

    A finite-extent circular shaped probe is represented as 
    a complex wave. The intensity of the probe is maximum at 
    the center and damps to zero at the borders of the frame. 

    Attributes
    size : int
    ----------
        Size of the square 2D frame for the probe.
    rin : float
        Value between 0 and 1 determining where the 
        dampening of the intensity will start.
    rout : float
        Value between 0 and 1 determining where the 
        intensity will reach to zero.
    maxint : float
        Maximum intensity of the probe at the center.
    """

    def __init__(self, weights, maxint=1e5):
        self.weights = weights
        self.size = weights.shape[0]
        self.maxint = maxint
        self.shape = weights.shape

    @property
    def amplitude(self):
        """Amplitude of the probe wave"""
        return np.sqrt(self.maxint) * self.weights

    @property
    def phase(self):
        """Phase of the probe wave."""
        return 0.2 * self.weights

    @property
    def intensity(self):
        """Intensity of the probe wave."""
        return np.power(self.amplitude, 2)

    @property
    def complex(self):
        return self.amplitude * np.exp(1j * self.phase)


class Scanner(object):
    def __init__(self, shape, sx, sy, margin=[0, 0], offset=[0, 0]):
        self.shape = shape
        self.sx = sx
        self.sy = sy
        self.margin = margin
        self.offset = offset

    @property
    def x(self):
        return np.arange(self.offset[0], self.shape[0]-self.margin[0]+1, self.sx)

    @property
    def y(self):
        return np.arange(self.offset[1], self.shape[1]-self.margin[1]+1, self.sy)


def gaussian(size, rin=0.8, rout=1):
    r, c = np.mgrid[:size, :size] + 0.5
    rs = np.sqrt((r - size/2)**2 + (c - size/2)**2)
    rmax = np.sqrt(2) * 0.5 * rout * rs.max() + 1.0
    rmin = np.sqrt(2) * 0.5 * rin * rs.max()
    img = np.zeros((size, size), dtype='float32')
    img[rs < rmin] = 1.0
    img[rs > rmax] = 0.0
    zone = np.logical_and(rs > rmin, rs < rmax)
    img[zone] = np.divide(rmax - rs[zone], rmax - rmin)
    return img


def scanner3(theta, shape, sx, sy, margin=[0, 0], offset=[0, 0], spiral=0):
    a = spiral
    scan = []
    lenmaxx = 0
    lenmaxy = 0

    for m in range(len(theta)):
        s = Scanner(shape, sx, sy, margin, offset=[
            offset[0], np.mod(offset[1]+a, sy)])
        scan.append(s)
        a += spiral
        lenmaxx = max(lenmaxx, len(s.x))
        lenmaxy = max(lenmaxy, len(s.y))

    scanax = -1+np.zeros([len(theta), lenmaxx], dtype='int32')
    scanay = -1+np.zeros([len(theta), lenmaxy], dtype='int32')

    for m in range(len(theta)):
        scanax[m, :len(scan[m].x)] = scan[m].x
        scanay[m, :len(scan[m].y)] = scan[m].y

    return scanax, scanay
