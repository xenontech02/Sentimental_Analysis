from tkinter import Tk
from tkinter.filedialog import askopenfilename

def load_text_file():
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(filetypes=[("Text files", "*.txt")])  # Show an "Open" dialog box and return the path to the selected file
    with open(filename, 'r') as file:
        text = file.read()
    return text