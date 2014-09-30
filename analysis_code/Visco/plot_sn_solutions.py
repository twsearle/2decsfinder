# -----------------------------------------------------------------------------
#
#   plot_dispersions.py
#   Last modified: Mon 29 Sep 16:53:35 2014
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
                    help= 'the dependent variable, one of Re, kx, KE0, Nu',
                    default='Re')

parser.add_argument('-b', "--beta", type=float,
                   help = 'Set the viscosity ratio, default is all beta',
                   default=None)

parser.add_argument("-o", "--output", 
                    help= 'output filename for final graph')

args = parser.parse_args()

yax_test = ['Re', 'kx', 'KE0', 'Nu']

if not args.yaxis in yax_test:
    print 'Error: invalid dependent variable choice'
    exit(1)

#extent = args.extent

# -----------------------------------------------------------------------------

# MAIN

# Go through the files in the folder and pick out the ones with data in
rootName = 'sn_points'
fileList = glob.glob(rootName + '*.dat')

fileListActual = []

for filename in fileList:
    splitString = filename.split('_')

    if len(splitString) < 4: continue
    sN = splitString[2]
    sM = splitString[3]
    sb = splitString[4]

    if sN[0] != 'N': continue
    if sM[0] != 'M': continue
    if sb[0] != 'b': continue

    fN = int(sN[1:])
    fM = int(sM[1:])
    fb = float(sb[1:-4])

    # add the option to only read in one set of beta 
    if args.beta is not None:
        beta_ = float(args.beta)
        if fb != beta_:
            continue 

    kwargs = {'fN':fN, 'fM':fM, 'fb':fb}
    print 'N={fN}, M={fM}, b={fb}'.format(**kwargs)

    fileListActual.append([filename, fN, fM, fb])

# Sort file names by beta

fileListActual.sort(key=operator.itemgetter(3))
fileListActual = array(fileListActual) 

# Set up the plot
#fig_width_pt = 452.0  # Get this from LaTeX using \showthe\columnwidth
#inches_per_pt = 1.0/72.27                # Convert pt to inch

golden_ratio = (sqrt(5)+1.0)/2.0         # Aesthetic ratio
fig_width = 9.5 #fig_width_pt*inches_per_pt   # width in inches
fig_height = fig_width/golden_ratio      # height in inches
fig_size =  [fig_width,fig_height]

rcParams.update({'figure.autolayout': True})
rcParams.update({'axes.linewidth': 2})
rcParams.update({'lines.linewidth': 3})
rcParams.update({'font.size': 14})
fig = plt.figure(figsize=fig_size)


plt.xlim(-0.1, 2.1)




plt.xlabel('$Wi$')
plt.ylabel(args.yaxis)

# This is a stupid little trick to get the axis label to not ovelap when
# latexed.
ax = fig.gca()
ax.xaxis.labelpad = 10

markerList = [ '+' , '.' , '1' , '2' , '3' , '4' ]

# Set up plots depending on the yaxis

plotsList = []
labelsList = []
counter = 0 
for [filename, N, M, beta] in fileListActual:

    if counter is len(markerList):
        counter = 0 

    # Read in the data
    # Re    kx  KE0 Nu  Wi
    data = genfromtxt(filename)
    print type(shape(data))
    print shape(data)
    if shape(data)[0] is 5:
        print 'file containing only 1 point'
        Res = data[0]
        kxs = data[1]
        KEs = data[2]
        Nus = data[3]
        Wis = data[4]
    else:
        Res = data[:,0]
        kxs = data[:,1]
        KEs = data[:,2]
        Nus = data[:,3]
        Wis = data[:,4]

    # Plot this data to the graph

    if args.yaxis == 'Re':
        plot = ax.plot(Wis, Res, label='{N}, {M}, {beta}'.format(N=N, M=M,
                                                                 beta=beta),
                       linewidth=0.0, ms=20.0, marker=markerList[counter])
    elif args.yaxis == 'KE0':
        plot = ax.plot(Wis, KEs, label='{N}, {M}, {beta}'.format(N=N, M=M,
                                                                 beta=beta),
                       linewidth=0.0, ms=20.0, marker=markerList[counter])
    elif args.yaxis == 'Nu':
        plot = ax.plot(Wis, Nus, label='{N}, {M}, {beta}'.format(N=N, M=M,
                                                                 beta=beta),
                       linewidth=0.0, ms=20.0, marker=markerList[counter])
    elif args.yaxis == 'kx':
        plot = ax.plot(Wis, kxs, label='{N}, {M}, {beta}'.format(N=N, M=M,
                                                                 beta=beta),
                       linewidth=0.0, ms=20.0, marker=markerList[counter])

    # Add a label for this line 
    plotsList.append(plot)
    labelsList.append('{N}, {M}, {beta}'.format(N=N, M=M, beta=beta))

    counter += 1

# Add a legend
ax.legend(labelsList, loc='best')

if args.output:
    plt.savefig(args.output)
else:
    plt.show(block=True)

