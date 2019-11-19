import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.graphics import *
from kivy.uix.splitter import Splitter
from kivy.uix.widget import Widget

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute


kivy.require('1.11.1')


class OthelloBoard(GridLayout):

    def __init__(self, **kwargs):
        super(OthelloBoard, self).__init__(**kwargs)

        self.cols = 3
        
        self.add_widget(Label(text='qubit 1'), index=0)
        self.add_widget(Label(text='qubit 2'), index=1)
        self.add_widget(Label(text='qubit 3'), index=2)

        self.add_widget(Label(text='qubit 4'), index=0)
        self.add_widget(Label(text='qubit 5'), index=1)
        self.add_widget(Label(text='qubit 6'), index=2)

        self.add_widget(Label(text='qubit 7'), index=0)
        self.add_widget(Label(text='qubit 8'), index=1)
        self.add_widget(Label(text='qubit 9'), index=2)


class CircuitLiner(Screen):
    def __init__(self, **kwargs):
        self.canvas = Canvas
        with self.canvas:
            Line(points=[10, 10, 20, 20, 30, 30], width=10)


# class PlayScreen:
#     def __init__(self, **kwargs):
#         self.splitter = Splitter(sizable_from='right')
#         self.splitter.add_widget(OthelloBoard())
#         self.splitter.add_widget(CircuitLiner())


class QuantumOthello(App):
    def build(self):
        sm = ScreenManager()
        # sm.add_widget(OthelloBoard)
        sm.add_widget(CircuitLiner)
        return OthelloBoard()


if __name__ == '__main__':
    QuantumOthello().run()
