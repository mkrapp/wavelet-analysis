import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
import sys

# from mpl_toolkits.axes_grid1 import make_axes_locatable

from waveletFunctions import wave_signif, wavelet

__author__ = 'Evgeniya Predybaylo'


# WAVETEST Example Python script for WAVELET, using NINO3 SST dataset
#
# See "http://paos.colorado.edu/research/wavelets/"
# The Matlab code written January 1998 by C. Torrence
# modified to Python by Evgeniya Predybaylo, December 2014
#
# Modified Oct 1999, changed Global Wavelet Spectrum (GWS) to be sideways,
#   changed all "log" to "log2", changed logarithmic axis on GWS to
#   a normal axis.
# ---------------------------------------------------------------------------

def data2df():
    # READ THE DATA
    sst = np.loadtxt('sst_nino3.dat')  # input SST time series
    dt = 0.25
    time = np.arange(len(sst)) * dt + 1871.0  # construct time array
    series = pd.Series(sst,index=time,name='NINO3 SST (\u2103)')
    series.index.name = "Time (year)"
    return series

def plotWavelet(fnm,params):

    # READ THE DATA
    #sst = np.loadtxt('sst_nino3.dat')  # input SST time series
    #df_y = data2df()
    #df_y.to_csv("sst_nino3.csv")

    df = pd.read_csv(fnm,index_col=0).squeeze("columns").dropna()
    #df.index = (df.index - df.index.min())  / np.timedelta64(1,'Y')
    if params["transform"] == "log":
        df = df.apply(np.log) # log-transform
    y = df.values.flatten()
    time = df.index.values
    # check for dates in index, e.g., "2021-05-19" 
    dates = None
    try:
        dt = np.diff(time)[0]
    except:
        time = np.arange(len(time))
        dt = np.diff(time)[0]
        dates = pd.to_datetime(df.index)
    name = df.name
    x_unit = df.index.name
    xlim = ([df.index[0],df.index[-1]])

    lag1 = df.autocorr(1)

    y = y - np.mean(y)
    variance = np.std(y, ddof=1) ** 2
    #print(f"variance = {variance:.2g}")

    ## to be specificed by user
    #max_powers = 7
    #scale1 = 2
    #scale2 = 8
    #units = "\u2103"
    #title = "NINO3 Sea Surface Temperature (seasonal)"
    #levels = [0, 0.5, 1, 2, 4, 999]

    max_powers = params["max_power"]
    scale1     = params["scale1"]
    scale2     = params["scale2"]
    units      = params["units"]
    dt_units   = params["dt_units"]
    title      = params["title"]
    levels     = params["levels"]

    # ----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E---------------

    # normalize by standard deviation (not necessary, but makes it easier
    # to compare with plot on Interactive Wavelet page, at
    # "http://paos.colorado.edu/research/wavelets/plot/"
    if 0:
        variance = 1.0
        y = y / np.std(y, ddof=1)
    n = len(y)
    #dt = 0.25
    #time = np.arange(len(y)) * dt + 1871.0  # construct time array
    #xlim = ([1870, 2000])  # plotting range
    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.25  # this will do 4 sub-octaves per octave
    s0 = 2 * dt  # this says start at a scale of 6 months
    j1 = max_powers / dj  # this says do 7 powers-of-two with dj sub-octaves each
    #lag1 = 0.72  # lag-1 autocorrelation for red noise background
    #print(f"lag1 = {lag1:.2f}")
    mother = 'MORLET'

    # Wavelet transform:
    wave, period, scale, coi = wavelet(y, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

    # Significance levels:
    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
        lag1=lag1, mother=mother)
    # expand signif --> (J+1)x(N) array
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    sig95 = power / sig95  # where ratio > 1, power is significant

    # Global wavelet spectrum & significance levels:
    dof = n - scale  # the -scale corrects for padding at edges
    global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1,
        lag1=lag1, dof=dof, mother=mother)

    # Scale-average between El Nino periods of 2--8 years
    avg = np.logical_and(scale >= scale1, scale < scale2)
    Cdelta = 0.776  # this is for the MORLET wavelet
    # expand scale --> (J+1)x(N) array
    scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    scale_avg = power / scale_avg  # [Eqn(24)]
    scale_avg = dj * dt / Cdelta * sum(scale_avg[avg, :])  # [Eqn(24)]
    scaleavg_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=2,
        lag1=lag1, dof=([scale1, scale2-0.1]), mother=mother)

    # ------------------------------------------------------ Plotting

    # --- Plot time series
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                        wspace=0, hspace=0)
    plt.subplot(gs[0, 0:3])
    if dates is None:
        x = time
    else:
        x = dates
    plt.plot(x, y, 'k')
    #plt.xlim(xlim[:])
    plt.xlabel(f'{x_unit}')
    plt.ylabel(name)
    plt.title(f'a) {title}')

    plt.text(1.25, 0.5, 'Wavelet Analysis\nC. Torrence & G.P. Compo\n'
        'http://paos.colorado.edu/\nresearch/wavelets/',
        horizontalalignment='center', verticalalignment='center',transform=plt.gca().transAxes)

    # --- Contour plot wavelet power spectrum
    # plt3 = plt.subplot(3, 1, 2)
    plt3 = plt.subplot(gs[1, 0:3])
    #levels = [0, 0.5, 1, 2, 4, 999]
    # *** or use 'contour'
    CS = plt.contourf(x, period, power, len(levels))
    im = plt.contourf(CS, levels=levels,
        colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
    plt.xlabel(f'{x_unit}')
    plt.ylabel(f'Period ({dt_units})')
    plt.title(f'b) Wavelet Power Spectrum (contours at {str(levels[1:-1])[1:-1]} ({units})\u00b2)')
    #plt.xlim(xlim[:])
    # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(x, period, sig95, [-99, 1], colors='k')
    # cone-of-influence, anything "below" is dubious
    plt.fill_between(x, coi * 0 + period[-1], coi, facecolor="none",
        edgecolor="#00000040", hatch='x')
    plt.plot(x, coi, 'k')
    # format y-scale
    plt3.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()
    # set up the size and location of the colorbar
    # position=fig.add_axes([0.5,0.36,0.2,0.01])
    # plt.colorbar(im, cax=position, orientation='horizontal')
    #   , fraction=0.05, pad=0.5)

    # plt.subplots_adjust(right=0.7, top=0.9)

    # --- Plot global wavelet spectrum
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws, period, label="data")
    plt.plot(global_signif, period, '--',label="red\nnoise")
    plt.legend(fontsize=8)
    plt.xlabel(f'Power ({units})\u00b2')
    plt.title('c) Global Wavelet Spectrum')
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt4.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()

    # --- Plot 2--8 yr scale-average time series
    plt.subplot(gs[2, 0:3])
    plt.plot(x, scale_avg, 'k')
    #plt.xlim(xlim[:])
    plt.xlabel(f'{x_unit}')
    plt.ylabel(f'Avg variance ({units})\u00b2')
    plt.title(f'd) {scale1}-{scale2} {dt_units} Scale-average Time Series')
    #plt.plot(xlim, scaleavg_signif + [0, 0], '--')

    #plt.show()
    return fig

def main():
    file_path = sys.argv[1]
    params= {
            "transform": "log",
            #"transform": "lin",
            "max_power": 7,
            "scale1": 2,
            "scale2": 8,
            "units": "\u2103",
            "title": "NINO3 Sea Surface Temperature (seasonal)",
            "levels": [0, 0.5, 1, 2, 4, 999]
    }
    fig = plotWavelet(file_path,params)
    plt.show()

if __name__ == "__main__":
    main()
