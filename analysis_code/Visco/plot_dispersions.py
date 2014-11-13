# -----------------------------------------------------------------------------
#
#   plot_dispersions.py
#   Last modified: Thu  6 Nov 15:02:33 2014
#
# -----------------------------------------------------------------------------
"""
 Given a list of files of leading growth rates, plot the dispersion 
 relations
"""
# MODULES

import glob
import argparse
import operator

from scipy import *
from matplotlib import pyplot as plt
from matplotlib import rcParams, rc

# Settings --------------------------------------------------------------------

parser = argparse.ArgumentParser()

#parser.add_argument('-e', "--extent", nargs=4, type=float, default=None)

parser.add_argument('-y', "--yaxis", 
                    help= 'add to plot the speed instead of the growth rate',
                    action="store_true")

parser.add_argument('-b', "--beta", type=float,
                   help = 'Set the viscosity ratio, default is all beta',
                   default=None)

parser.add_argument("-o", "--output", 
                    help= 'output filename for final graph')

args = parser.parse_args()

#extent = args.extent

# -----------------------------------------------------------------------------

# MAIN

# Go through the files in the folder and pick out the ones with data in
rootName = 'lead-ev-'
fileList = glob.glob(rootName + '*.dat')

fileListActual = []

for filename in fileList:
    splitString = filename.split('-')

    if len(splitString) < 4: continue
    sN = splitString[2]
    sM = splitString[3]
    sb = splitString[6]
    sWi = splitString[7]

    if sN[0] != 'N': continue
    if sM[0] != 'M': continue
    if sb[0] != 'b': continue
    if sWi[:2] != 'Wi': continue
    if sWi[5:-4] == '_low' : continue
    print filename

    fN = int(sN[1:])
    fM = int(sM[1:])
    fb = float(sb[1:])
    fWi = float(sWi[2:-4])

    # add the option to only read in one set of beta 
    if args.beta is not None:
        beta_ = float(args.beta)
        if fb != beta_:
            continue 

    kwargs = {'fN':fN, 'fM':fM, 'fb':fb, 'fWi':fWi}
    print 'N={fN}, M={fM}, b={fb}'.format(**kwargs)

    fileListActual.append([filename, fN, fM, fb, fWi])

# Sort file names by beta

fileListActual.sort(key=operator.itemgetter(4))
fileListActual.sort(key=operator.itemgetter(3))
fileListActual = array(fileListActual) 

# Set up the plot
#fig_width_pt = 452.0  # Get this from LaTeX using \showthe\columnwidth
#inches_per_pt = 1.0/72.27                # Convert pt to inch

golden_ratio = (sqrt(5)+1.0)/2.0         # Aesthetic ratio
fig_width = 10.5 #fig_width_pt*inches_per_pt   # width in inches
fig_height = fig_width/golden_ratio      # height in inches
fig_size =  [fig_width,fig_height]

rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.linewidth': 2})
rcParams.update({'lines.linewidth': 3})
rcParams.update({'font.size': 14})
fig = plt.figure(figsize=fig_size)


plt.xlabel('$k_{z}$')

if args.yaxis:
    plt.ylabel(r'wave speed |$\nu_{z}$|')

else:
    plt.ylabel('growth rate')

# This is a stupid little trick to get the axis label to not ovelap when
# latexed.
ax = fig.gca()
ax.xaxis.labelpad = 10

markerList = [ '+' , '.' , '1' , '2' , '3' , '4' ]

# Set up plots depending on the yaxis

plotsList = []
labelsList = []
counter = 0 
for [filename, N, M, beta, Wi] in fileListActual:

    if counter is len(markerList):
        counter = 0 

    # Read in the data
    # Re    kx  KE0 Nu  Wi
    data = genfromtxt(filename)
    print type(shape(data))
    kz = data[:,0]
    growthRate = data[:,1]
    speed = data[:,2]

    # Plot this data to the graph

    if args.yaxis:
        plot = ax.plot(kz, abs(speed), label=r'{N}, {M}, $\beta$ = {beta}, Wi = {Wi}'.format(N=N, M=M,
                                                                 beta=beta, Wi=Wi),
                       linewidth=1.0, marker=markerList[counter])
    else:
        plot = ax.plot(kz, growthRate, label=r'{N}, {M}, $\beta$ = {beta}, Wi = {Wi}'.format(N=N, M=M,
                                                                 beta=beta, Wi=Wi),
                       linewidth=1.0, marker=markerList[counter])

    # Add a label for this line 
    plotsList.append(plot)
    labelsList.append(r'{N}, {M}, $\beta$ = {beta}, Wi = {Wi}'.format(N=N, M=M, beta=beta, Wi=Wi))

    counter += 1

# Add a legend
ax.legend(labelsList, loc='best')

if args.output:
    plt.savefig(args.output)
else:
    plt.show(block=True)

