#!/usr/bin/env python

from numpy import random as nprand
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from intervals import *
import pylab
import sys
import os


def draw_y(p):
    c = nprand.uniform()
    if c < p:
        return 1
    return 0


def draw_x():
    return nprand.uniform()


def check_x_interval(x):
    if 0 <= x <= 0.25 or 0.5 <= x <= 0.75:
        return 0.8
    return 0.1


def draw_m_points(m):
    points = []
    for i in xrange(m):
        x = draw_x() # draw uniform x
        p = check_x_interval(x) # check the interval of x
        y = draw_y(p) # draw a label to x with respect to the interval
        points.append((x,y))
    return points


def get_points_and_intervals(points, k=2):
    '''
    seperates points (which contains the data and the respective label to each data point)
    to a data set (X) and label set (Y)
    '''
    m = len(points)
    X = []
    Y = []
    
    for i in xrange(m):
        X.append(points[i][0])
        Y.append(points[i][1])
        
    intervals, error = find_best_interval(X, Y, k)
    return (X, Y, intervals, error)


def plot_points(m, path = ''):
    '''
    Subsection a:
    plots a label(data) graph, with intervals marked on graph
    '''
    # find the best hypothesis given a random training set
    X, Y, intervals, error = get_points_and_intervals(sorted(draw_m_points(m), key = lambda p: p[0]))
    
    plt.axis([0, 1, -0.1, 1.1])
    fig = plt.figure(figsize=(9,7))
    plt.scatter(X, Y) # data points and their labels

    vert_lines = [0.25, 0.5, 0.75]
    for v_line in vert_lines:
        plt.axvline(x=v_line, color='y') # mark the required x values

    for h_line in intervals:
        plt.axhline(0.5, h_line[0], h_line[1], color='r') # mark the hypothesis (intervals) on graph
    fig.savefig(path + 'out_a.png') # output plot for subsection a
    plt.close()


######## Auxiliary functions for subsections c and d ########

def calc_error_good(x_right, x_left, x, x_is_left_edge):
        if x_is_left_edge:
            P = 0.8
        else:
            P = 0.2

        return P * (min(x, x_right) - x_left)

def calc_error_bad(x_right, x_left, x, x_is_left_edge):
        if x_is_left_edge:
            P = 0.1
        else:
            P = 0.9

        return P * (min(x, x_right) - x_left)

def is_good(x):
    return x in [0.25, 0.75]

def calc_current_error(x, x_right, x_left, is_left):
    if is_good(x_right):
        return calc_error_good(x_right, x_left, x, is_left)
    return calc_error_bad(x_right, x_left, x, is_left)


def get_P(is_odd, i):
    if is_odd:
        return 0.2 if i % 2 == 1 else 0.8
    else:
        return 0.9 if i % 2 == 1 else 0.1

def calc_true_error(intervals):
    '''
    'good' intervals are [0, 0.25] and [0.5, 0.75]
    '''

    # making sure the intervals will be sorted
    sorted_intervals = sorted(intervals, key = lambda interval : interval[0])
    edges = []
    for interval in sorted_intervals:
        edges.append(interval[0])
        edges.append(interval[1])
    
    error = 0.0
    x_left = 0.0
    i = 0

    # checking the error for each quartile (x_right = 0.25, 0.5, 0.75, 1)
    for j in xrange(1, 5):
        x_right = 0.25*j
        while i < len(edges) and edges[i] <= x_right:
            # if i is even the edge opens an interval, else it closes one
            error += calc_current_error(edges[i], x_right, x_left, i % 2 == 0)
            i += 1
            x_left = edges[i - 1]

        # P is calculated according to weather x is in a good interval or not
        P = get_P(j % 2 == 1, i)
        error += (P * (x_right - x_left))

        x_left = x_right
        
    return error


def plot_graph(X, Y1, Y2, col1, col2, filename):
    fig = plt.figure(figsize=(9,7))
    plt.plot(X, Y1, color=col1, label='empirical')
    plt.plot(X, Y2, color=col2, label='true')
    plt.legend(loc='upper right')
    fig.savefig(filename)
    plt.close()

################################################


def plot_errors(T, path = ''):
    '''
    Subsection c:
    plots the true and empirical error as a function of m (after averaging T times)
    '''

    ms = [5 * m for m in xrange(2, 21)]
    empirical = []
    true = []
    for m in ms:
        empirical_error, true_error = 0, 0
        mf = float(m)
        for i in xrange(T):
            X, Y, intervals, error = get_points_and_intervals(sorted(draw_m_points(m), key = lambda p: p[0]))
            empirical_error += (error/mf)
            true_error += calc_true_error(intervals)
            
        empirical.append(empirical_error/T)
        true.append(true_error/T)

    plot_graph(ms, empirical, true, 'y', 'b', path + 'out_c.png')


def find_best_hypot(m, kmax, to_plot = True, path = ''):
    '''
    Subsection d:
    plots the true and empirical error as a function of k (for k = 1,...,kmax)
    '''

    points = draw_m_points(m) # same training set for each k
    points.sort(key = lambda p: p[0])
    empirical = []
    true = []
    hypothesis = [] # set of hypothesis, will be in use only in subsection e
    ks = [k for k in xrange(1, kmax+1)]
    mf = float(m)
    for k in ks:
        # find for each k the best suited intervals for the training set
        X, Y, hyp, err = get_points_and_intervals(points, k)
        empirical.append(err/mf)
        true.append(calc_true_error(hyp))
        hypothesis.append(hyp)

    if to_plot: #plotting in subsection d only
        plot_graph(ks, empirical, true, 'y', 'b', path + 'out_d.png')
    else:
        return hypothesis


######## Auxiliary functions for subsection e ########
    
def check_in_interval(x, interval):
    return interval[0] <= x <= interval[1]


def calc_holdout_error(intervals, test_set):
    error = 0
    
    for x, y in test_set:
        is_error = False
        for interval in intervals:
            # is_error will be true iff there exists an interval that contains x
            is_error = is_error or check_in_interval(x, interval)

        if (y == 0 and is_error) or (y == 1 and not is_error):
            error += 1

    return error

######################################################


def holdout(m, path = ''):
    kmax = 20
    hs = find_best_hypot(m, kmax, False)
    errors = []
    points = draw_m_points(m)
    ks = []
    errs = []
    for k in xrange(kmax):
        h = hs[k]
        err = calc_holdout_error(h, points)
        errors.append((err, k + 1))
        ks.append(k + 1)
        errs.append(float(err)/float(m))

    fig = plt.figure(figsize=(9,7))
    plt.plot(ks, errs)
    fig.savefig(path + 'out_e.png')
    plt.close()

    return min(errors, key = lambda t : t[0])[1]

####### main flow #######
def execute_subsection(s, output_path):
    if s == 'a':
        plot_points(100, output_path)

    elif s == 'c':
        plot_errors(100, output_path)

    elif s == 'd':
        find_best_hypot(50, 20, path = output_path)

    elif s == 'e':
        holdout(50, output_path)

def usage():
    print('[ERROR]\nUsage:')
    print('\tpython ML_ex2.py subsection [path]')
    print('\tsubsection - one of (\'a\', \'c\', \'d\', \'e\')')
    print('\tpath - where to output the plots (OPTIONAL; default is module\'s directory)')
        
def main(args):
    subsections = ['a', 'c', 'd', 'e']
    l = len(args)
    if l < 1:
        usage()
        sys.exit(1)

    else:
        s = args[0]
        if s not in subsections:
            usage()
            sys.exit(1)
            
        if l == 1:
            path = os.path.dirname(os.path.realpath(__file__))
        else:
            path = args[1]

        execute_subsection(s, path + os.path.sep)    

if __name__ == '__main__':
    main(sys.argv[1:])





    
