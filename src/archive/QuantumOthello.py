import kivy
from kivy.app import App
from kivy.animation import Animation
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from math import fabs
from copy import deepcopy
from functools import partial
from os import sep

import random

kivy.require('1.11.1')


class QuantumOthello(App):
    def build(self):
        return OthelloBoard()


class OthelloBoard(GridLayout):

    def __init__(self, **kwargs):
        super(OthelloBoard, self).__init__(**kwargs)
        self.layout = GridLayout(cols=3)
        self.root = Widget()
        self.b1 = Button()
        self.b2 = Button()
        self.root.add_widget(b1)
        self.root.add_widget(b2)

if __name__ == '__main__':
    QuantumOthello().run()