import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import AutoEncoder as AE
import PCAALG as PC
import KNNALG as KN

bgcolor="#d71868"
bgcolor1="#9d2933"
fgcolor="#ffffff"

window = tk.Tk()
window.title("Digital Data Forgetting")

window.geometry('1280x720')
window.configure(background=bgcolor)
#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message1 = tk.Label(window, text="Digital Data Forgetting" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=2,font=('times', 25, 'italic bold underline')) 
message1.place(x=100, y=10)

lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl.place(x=100, y=170)

txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=400, y=175)


def clear():
	print("Clear1")
	txt.delete(0, 'end') 


def browse():
	path=filedialog.askopenfilename()
	print(path)
	txt.delete(0, 'end') 
	txt.insert('end',path)
	if path !="":
		print(path)
	else:
		tm.showinfo("Input error", "Select Dataset")

	
def PCAprocess():
	path=txt.get()
	if path !="":
		PC.process(path)
		tm.showinfo("Input ", "PCA Successfully Finished")	
	else:
		tm.showinfo("Input error", "Select Dataset")	
	

def KNNprocess():
	path=txt.get()
	if path !="":
		KN.process(path)
		tm.showinfo("Input ", "KNN Successfully Finished")	
	else:
		tm.showinfo("Input error", "Select Dataset")	

def AEprocess():
	path=txt.get()
	if path !="":
		AE.process(path)
		tm.showinfo("Input ", "AUTOENCODER Successfully Finished")	
	else:
		tm.showinfo("Input error", "Select Dataset")	


browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
browse.place(x=650, y=160)

proc1 = tk.Button(window, text="PCA", command=PCAprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
proc1.place(x=360, y=600)

proc2 = tk.Button(window, text="KNN", command=KNNprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
proc2.place(x=580, y=600)

proc3 = tk.Button(window, text="AUTOENCODER", command=AEprocess  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
proc3.place(x=820, y=600)


clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=920, y=160)
	
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1060, y=600)

window.mainloop()