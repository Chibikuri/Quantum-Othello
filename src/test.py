import kivy

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.lang import Builder
kivy.require("1.9.0")
Window.clearcolor = (0.5, 0.5, 0.5, 1)


class CalcGridLayout(GridLayout):

    def reduced_image(self):
    #ignore

    def build1(self):
        Window.bind(on_dropfile=self._on_file_drop)
        return

    def _on_file_drop(self, window, file_path):
        return file_path


class dropdownApp(App):

    def build(self):
        return Builder.load_string(kv)


dropApp = dropdownApp()
dropApp.run()
kv = '''
# Custom button
<CustButton@Button>:
    font_size: 32
    background_normal: 'Colour_yellow.png'
    background_down: 'Colour_blue.png'

<Cust2@Button>:
    font_size: 32
    background_normal: 'Colour_red.png'
    background_down: 'Colour_blue.png'

<Cust3@Button>:
    font_size: 32
    background_normal: 'Colour_white.png'
    background_down: 'Colour_blue.png'

<Cust4@Button>:
    font_size: 32
    background_normal: 'Colour_blue.png'
    background_down: 'Colour_white.png'

<CalcGridLayout>:
    id: calculator
    display: entry
    rows: 5
    padding: 10
    spacing: 10

    BoxLayout:
        spacing: 100
        size_hint: .5, None
        Cust2:
            text: "Whats the intensity you want?"

    BoxLayout:

        size_hint: .5, None
        TextInput:
            id: entry
            font_size: 70
            multiline: False
            hint_text: "Type here"

    BoxLayout:
        spacing: 100
        size_hint: .5, None
        Cust4:
            text: "Drag and Drop picture below:"
            on_release: root.build1()

    #THIS IS WHERE I'M STUCK ON
    BoxLayout:
        Image:
            source: root._on_file_drop(file_path)

    BoxLayout:
        size_hint: 1, .3
        spacing: 10

        CustButton:
            text: "Click here for \n reduced size"

        CustButton:
            text: "Click here for pos \n  and intensity of \n      each pixel"
            on_release: root.reduced_image()

        CustButton:
            text: "Click here \n for graph"

        CustButton:
            text: "Click here \n     for all"

        CustButton:
            text: "Extra"
'''