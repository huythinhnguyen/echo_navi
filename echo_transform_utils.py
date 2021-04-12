#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List of utility functions for processing and transforming echo waveform

@author: thinh
"""
"""  LIST OF COMPLETED FUNCTION:
    ----------------------------
def get_raw_echo(klass, dist, angle, dataset):
    ... take in klass, dist, angle and put it in a tuple "spec"
    ... randomly select an echo that meet the spec
    ... output --> echo = {"left", "right"}
    return echo

def subtract_bg(echo, angle, dataset):
    ... for a given input angle, select the approriate background echo then subtract the input echo for the background
    ... the background is randomly select from different sample in the dataset    
    return echo_nobg


reference: https://en.wikibooks.org/wiki/Engineering_Acoustics/Outdoor_Sound_Propagation
def timeshift_echo(echo, from_dist, to_dist, keepfront=True):
    ... shift the echo assuming speed of sound is 340 m/s
    ... sampling frequency should be 300,000
    return echo_shifted

----THIS FUNCTION IS FUNCTIONING BUT THE CORRECTNESS IS IFFY--------
----NEED TO CHECK THE PHYSICS OF GEOMETRIC SPREADING LOSS!!!--------
def attenuate_echo(echo, from_dist, to_dist,alpha = 1.31, spreadloss = True):
    ... 2 loss phenomena: atmospheric absorption and geometric spreading
    ... attenuation coefficient of air is set to 1.31 db/m
    ... geometric loss is assumed to be sqrt(r_1/r_2) (from r1 to r2)
    return echo_att

def echo_dist_transform(echo,from_dist,to_dist):
    ... call timeshift then call attenuate
    return echo_dist_trans

"""
import numpy as np
from ears_model import ear_filter

root = 'dataset/'
k_dict = {0: 'background', 1: 'pole', 2: 'planter'}


def retrieve_echo(klass, dist, angle, random_mode=True, index=None):
    path = root + k_dict[klass] + '/' + str(dist) + '_' + str(angle) + '/'
    left_set = np.load(path + 'left.npy')
    right_set = np.load(path + 'right.npy')
    if not random_mode:
        np.random.seed(1)
    number_of_results = left_set.shape[0]
    sel_idx = np.random.randint(number_of_results)
    if index is not None:
        sel_idx == index
    l_echo = left_set[sel_idx, :]
    r_echo = right_set[sel_idx, :]
    echo = {'left': l_echo,
            'right': r_echo}
    return echo


def get_raw_echo(klass, dist, angle, random_mode=True, index=None):
    echo = retrieve_echo(klass, dist, angle, random_mode=random_mode, index=index)
    return echo


def subtract_bg(echo, angle, random_mode=True, index=None):
    bg = retrieve_echo(0, 0.0, angle, random_mode=random_mode, index=index)
    l_nobg = echo["left"] - bg['left']
    r_nobg = echo["right"] - bg['right']
    echo_nobg = {
        "left"  : l_nobg ,
        "right" : r_nobg
    }
    return echo_nobg


def timeshift_echo(echo, from_dist, to_dist, keepfront=True):
    """
    assumption:
    speed of sound is c = 343m/s
    sampling freq = 3e5
    """
    c = 340 #m/s
    fs = 300000
    d_point = int((((2*(to_dist - from_dist)/c)) * fs) )
    #now, shift the array right by d_point
    
    temp_l = echo["left"][:d_point]
    temp_r = echo["right"][:d_point]
    l_shifted = np.roll(echo["left"],d_point)
    r_shifted = np.roll(echo["right"],d_point)
    if (keepfront == True) & (d_point>=0):
        l_shifted[:d_point] = temp_l
        r_shifted[:d_point] = temp_r       
    
    echo_shifted = {
        "left": l_shifted,
        "right": r_shifted
    }
    
    return echo_shifted


def attenuate_echo(echo, from_dist, to_dist, alpha=1.31, spreadloss=True):
    """
    assumption:
    we use this equation to calculate the attenuation
    A = Ao * exp(-alpha*r)
    A --> attenuated echo
    Ao --> original echo
    alpha --> attenuation factor at 20C, RH=50%, 1.32dB/m
    r = distance the sound travel = 2*(to_dist - from_dist)
    """
    r = to_dist - from_dist
    att = np.exp(-alpha*r)
    if spreadloss:
        att = att * (( (from_dist/to_dist) **(0.5) ))
    l_att = echo["left"]*att
    r_att = echo["right"]*att
    
    echo_att = {
        "left": l_att,
        "right": r_att
    }
    return echo_att


def echo_dist_transform(echo, from_dist, to_dist, keepfront=True, spreadloss=True):
    spec = (from_dist, to_dist)
    echo_dist_trans = timeshift_echo(echo, *spec, keepfront=keepfront)
    echo_dist_trans = attenuate_echo(echo_dist_trans, *spec, spreadloss=spreadloss)
    return echo_dist_trans


def get_raw_echo_nobg(klass,dist,angle, random_mode=True, index=None):
    spec = (klass, dist, angle)
    echo = get_raw_echo(*spec, random_mode=random_mode, index=index)
    echo = subtract_bg(echo,angle)
    return echo


def get_echo_dist_trans(klass, dist, angle, to_dist, keepfront=True, spreadloss=True):
    spec = (klass, dist, angle)
    dist_spec = (dist, to_dist)
    echo = get_raw_echo_nobg(*spec)
    echo = echo_dist_transform(echo, *dist_spec, keepfront=keepfront, spreadloss=spreadloss)
    return echo


def get_echo_at_dist(klass, dist, angle, keepfront=True, spreadloss=True):
    d4x = 4*dist
    d4x_floor = np.floor(d4x)
    d4x_ceil = np.ceil(d4x)
    floor_gap = (d4x - d4x_floor)/4
    ceil_gap = (d4x_ceil - d4x)/4
    from_dist = d4x_ceil/4 if (ceil_gap < 0.05) else d4x_floor/4
    echo = get_echo_dist_trans(klass, from_dist, angle, dist,keepfront=keepfront, spreadloss=spreadloss)
    return echo


def echo_floor_ceil_angle_echo(klass, dist, angle, keepfront=True, spreadloss=True):
    sign = -1 if angle < 0 else 1
    floor_angle = sign*np.floor(np.abs(angle))
    ceil_angle = sign*np.ceil(np.abs(angle))
    floor_gap = np.abs(angle - floor_angle)
    ceil_gap = np.abs(ceil_angle - angle)
    echo_floor = get_echo_at_dist(klass, dist, floor_angle, keepfront=keepfront, spreadloss=spreadloss)
    echo_ceil = get_echo_at_dist(klass, dist, ceil_angle, keepfront=keepfront, spreadloss=spreadloss)
    return echo_floor, echo_ceil, floor_gap, ceil_gap


def get_echo_trans(klass, dist, angle, keepfront=True, spreadloss=True):
    echo_floor, echo_ceil, floor_gap, ceil_gap = echo_floor_ceil_angle_echo(klass, dist, angle, keepfront=keepfront, spreadloss=spreadloss)
    l_echo = ceil_gap * echo_floor['left'] + floor_gap * echo_ceil['left']
    r_echo = ceil_gap * echo_floor['right'] + floor_gap * echo_ceil['right']
    echo = {
        "left": l_echo,
        "right": r_echo
    }
    return echo


def get_total_echo(inView, inView_dist, inView_angle, keepfront=True, spreadloss=True):
    if inView.shape[0] == 0:
        bg = retrieve_echo(0,0.0,0.0)
        l_echo = bg['left']
        r_echo = bg['right']
    else:
        echo_set_left = np.zeros((inView.shape[0], 7000))
        echo_set_right = np.zeros((inView.shape[0], 7000))
        bg = retrieve_echo(0,0.0,0.0)
        for i in range(echo_set_left.shape[0]):
            echo_i = get_echo_trans(int(inView[i,2]), inView_dist[i,0], inView_angle[i,0], keepfront=keepfront, spreadloss=spreadloss)
            echo_set_left[i, :] = echo_i['left'] + bg['left']
            echo_set_right[i, :] = echo_i['right'] + bg['right']
        l_echo = np.sum(echo_set_left, axis=0)
        r_echo = np.sum(echo_set_right, axis=0)
    echo = {
        "left": l_echo,
        "right": r_echo
    }
    return echo


def echo2envelope(echo, single_channel=True, banksize=10, center_freq=None):
    l_echo = np.copy(echo['left'])
    r_echo = np.copy(echo['right'])
    l_envelope = ear_filter(l_echo, single_channel=single_channel, banksize=banksize, center_freq=center_freq)
    r_envelope = ear_filter(r_echo, single_channel=single_channel, banksize=banksize, center_freq=center_freq)
    envelope = {
        'left': l_envelope,
        'right': r_envelope
    }
    return envelope


def compress_envelope(envelope, subsample=140):
    l_envelope = envelope['left']
    r_envelope = envelope['right']
    l_z_envelope = np.mean(l_envelope.reshape(-1,subsample),axis=1)
    r_z_envelope = np.mean(r_envelope.reshape(-1,subsample),axis=1)
    z_envelope = {
        'left': l_z_envelope,
        'right': r_z_envelope
    }

    return z_envelope


def echo2envelope_z(echo, single_channel=True, banksize=10, center_freq=None, subsample=140):
    envelope = echo2envelope(echo, single_channel=single_channel, banksize=banksize, center_freq=center_freq)
    z_envelope = compress_envelope(envelope, subsample=subsample)
    return z_envelope

