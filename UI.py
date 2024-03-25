import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter.ttk import Combobox
import numpy as np

root = tk.Tk()
root.geometry("1150x700")

global global_image
global_image = None

global selected2
selected2="RGB"

global center_x, center_y
center_x = center_y = None

global sigma_x, sigma_y
sigma_x = sigma_y = None

def open_image():
    global global_image
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path)
    global_image = cv2.resize(image, (300, 300))
    image = cv2.cvtColor(global_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    image = ImageTk.PhotoImage(pil_image)  # Store the image globally
    label.configure(image=image)
    label.image = image

def on_select(event):
    selected = selected_format.get()
    show_mat(selected)

def select(event):
    global selected2
    global global_image
    selected2=selected_input_format.get()
    print(selected2)
    if selected2 == "Grey":
        image = cv2.cvtColor(global_image, cv2.COLOR_RGB2GRAY)
        pil_image = Image.fromarray(image)
        image = ImageTk.PhotoImage(pil_image)  # Store the image globally
        label.configure(image=image)
        label.image = image
    elif selected2 == "HSV":
        image = cv2.cvtColor(global_image, cv2.COLOR_RGB2HSV)
        pil_image = Image.fromarray(image)
        image = ImageTk.PhotoImage(pil_image)  # Store the image globally
        label.configure(image=image)
        label.image = image
def print_center():
    global center_x, center_y
    center_x = entry_x.get()
    center_y = entry_y.get()
    print("Center X:", center_x)
    print("Center Y:", center_y)

def print_sigma():
    global sigma_x, sigma_y
    sigma_x = entry_x2.get()
    sigma_y = entry_y2.get()
    print("Sigma X:", sigma_x)
    print("Sigma Y:", sigma_y)

def perform_convolution():
    # You can perform convolution here
    global global_image
    print("Performing convolution...")
    image = cv2.cvtColor(global_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    image = ImageTk.PhotoImage(pil_image)  # Store the image globally
    label_out.configure(image=image)
    label_out.image = image

frame = tk.Frame(root, width=300, height=100)
frame.place(x=410, y=0)

button_open_image = tk.Button(frame, text="Open Image", command=open_image)
button_open_image.grid(row=0, column=1, padx=10, pady=10)

image_formats = ['Blur', 'Sharp', 'Edge']
selected_format = tk.StringVar()

combo_box = Combobox(frame, textvariable=selected_format, values=image_formats)
combo_box.grid(row=1, column=1, padx=10, pady=10)

combo_box.bind("<<ComboboxSelected>>", on_select)

label2 = tk.Label(frame, text="Matrix")
label2.grid(row=3, column=1, padx=10, pady=10)

frame2 = tk.Frame(root, width=300, height=300)
frame2.place(x=20, y=100)

label = tk.Label(frame2)
label.grid(row=0, column=0, padx=10, pady=10)


#image output
frame7 = tk.Frame(root, width=300, height=300)
frame7.place(x=750, y=100)

label_out = tk.Label(frame7)
label_out.grid(row=0, column=0, padx=10, pady=10)

text = tk.Text(frame2, width=11, height=1)
text.insert(tk.END, "Input Photo")
text.grid(row=1, column=0, padx=10, pady=10)

frame3 = tk.Frame(root, width=300, height=200)
frame3.place(x=400, y=150)

matrix1 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
matrix2 = np.array([[7, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

matrix3 = np.array([[8, 2, 3, 5, 1, 2, 3],
                    [4, 5, 6, 7, 5, 2, 8],
                    [8, 8, 9, 9, 6, 2, 8],
                    [8, 8, 9, 9, 6, 2, 8],
                    [8, 8, 9, 9, 6, 2, 8],
                    [8, 8, 9, 9, 6, 2, 8],
                    [8, 8, 9, 9, 6, 2, 8]])

matrices = [matrix1, matrix2, matrix3]

def show_mat(s):
    global matrix
    if s == "Blur":
        v = 0
        for widget in frame3.winfo_children():
            widget.destroy()
    elif s == "Sharp":
        v = 1
        for widget in frame3.winfo_children():
            widget.destroy()
    elif s == "Edge":
        v = 2
        for widget in frame3.winfo_children():
            widget.destroy()
    matrix = matrices[v]
    m = matrix.shape[0]
    n = matrix.shape[1]
    for i in range(m):
        for j in range(n):
            value = matrix[i, j]
            if 3 <= j <= n - 2:
                label_text = "..."
            else:
                label_text = str(value)
            label1 = tk.Label(frame3, text=label_text, bg='lightgreen')
            label1.place(relx=(j + 0.5) / 5, rely=(i + 0.5) / 5, anchor='center')

frame4 = tk.Frame(root, width=300, height=300)
frame4.place(x=450, y=350)

label_center = tk.Label(frame4, text="Center of the kernel")
label_center.grid(row=4, column=1, padx=10, pady=10)

entry_x = tk.Entry(frame4, width=5)  # Smaller width
entry_x.grid(row=5, column=0, padx=(0.5, 0.5), pady=5)  # Adjusted padx

entry_y = tk.Entry(frame4, width=5)  # Smaller width
entry_y.grid(row=5, column=2, padx=(0.5, 0.5), pady=5)  # Adjusted padx

button_center = tk.Button(frame4, text="Center", command=print_center)
button_center.grid(row=6, column=1, padx=10, pady=10)

frame5 = tk.Frame(root, width=300, height=300)
frame5.place(x=450, y=470)

label_center = tk.Label(frame5, text="SigmaX and SigmaY")
label_center.grid(row=4, column=1, padx=10, pady=10)

entry_x2 = tk.Entry(frame5, width=5)  # Smaller width
entry_x2.grid(row=5, column=0, padx=(0.5, 0.5), pady=5)  # Adjusted padx

entry_y2 = tk.Entry(frame5, width=5)  # Smaller width
entry_y2.grid(row=5, column=2, padx=(0.5, 0.5), pady=5)  # Adjusted padx

button_sigma = tk.Button(frame5, text="Sigma Values", command=print_sigma)
button_sigma.grid(row=6, column=1, padx=10, pady=10)

frame6 = tk.Frame(root, width=300, height=300)
frame6.place(x=450, y=600)

button_convolution = tk.Button(frame6, text="Convolution", command=perform_convolution)
button_convolution.pack(pady=10)

frame8 = tk.Frame(root, width=300, height=100)
frame8.place(x=40, y=470)

label_input = tk.Label(frame8, text="Image type")
label_input.grid(row=0, column=0, padx=10, pady=10)

input_formats = ['Grey', 'BRG', 'HSV']  # Modify this list as per your requirement
selected_input_format = tk.StringVar()

combo_box_input = Combobox(frame8, textvariable=selected_input_format, values=input_formats)
combo_box_input.grid(row=0, column=1, padx=10, pady=10)
combo_box_input.bind("<<ComboboxSelected>>", select)

root.mainloop()
