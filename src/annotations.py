#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import matplotlib
import matplotlib.pyplot as plt


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


class Annotate():

    # Class Instance to fix landmark annotations

    # init method
    def __init__( self, 
                  dataframe,
                  volume,
                  index,
                  slice_id,
                  labels ):

            # Instance Variables
            self.dataframe = dataframe          # dataframe of landmark points
            self.volume = volume                # volume of dicom images (generated using landmark module)
            self.index = index                  # index number in dataframe to edit
            self.slice_id = slice_id            # slice id in dataframe to edit
            self.labels = labels                # point labels to edit (list)
            self.linebuilder = None
            self.image = None

    def plot(self):

        # function to display plot

        # get correct image
        phase = int(self.dataframe.loc[self.index, 'Time Frame'])
        self.image = self.volume[self.slice_id, phase, :, :, 0]

        # draw plot
        plt.figure(figsize=(16,16))
        plt.imshow(self.image, cmap='gray')

        # initiate empty line
        line, = plt.plot([], [])  # empty line
        self.linebuilder = LineBuilder(line)

        # format plot
        plt.xticks([])
        plt.yticks([])
        plt.title('Draw line for correct valve plane by clicking both valve insertion ')
        plt.show()


    def update(self):

        # updates dataframe with corrected points

        if self.linebuilder.ys:
            p1 = [int(self.linebuilder.ys[0]), int(self.linebuilder.xs[0])]
            p2 = [int(self.linebuilder.ys[1]), int(self.linebuilder.xs[1])]

        # display updated points
        plt.figure()
        plt.imshow(self.image, cmap='gray')
        plt.scatter(x=p1[1], y=p1[0])
        plt.scatter(x=p2[1], y=p2[0])
        plt.title('Updated Points')
        plt.show()

        # update dataframe
        self.dataframe[self.labels[0]][self.index] = p1
        self.dataframe[self.labels[1]][self.index] = p2

        return self.dataframe