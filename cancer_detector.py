import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pandas as pd
import os

class CancerDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cancer Detector")
        window_width = 500
        window_height = 500
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
        self.root.configure(bg='gray16')
        self.frame = tk.Frame(root, bg='gray16')
        self.frame.place(relx=0.5, rely=0.5, anchor='center')
        self.result_label = tk.Label(self.frame, bg='gray16', fg='white')
        self.result_label.pack(pady=20)
        self.image_label = tk.Label(self.frame, bg='gray16')
        self.image_label.pack()
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image, 
                                    bg="white", fg="black", font=("Helvetica", 16, "bold"), 
                                    relief="groove", bd=2, highlightbackground="black", 
                                    highlightthickness=1)
        self.upload_button.pack(pady=20)
        self.vote_label = tk.Label(self.frame, bg='gray16', fg='white')
        self.vote_label.pack()
        self.model = load_model(model_name)
        self.annotations = pd.read_csv('annotations.csv')

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            image = Image.open(file_path)
            image = image.resize((300, 300))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array / 255.0
            predictions = self.model.predict(image_array)
            predicted_class = np.round(predictions)
            result = "No Cancer" if predicted_class == 0 else "Cancer"
            self.result_label.config(text=f"Model Prediction: {result}", font=("Helvetica", 20, "bold"))
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            image_name = os.path.basename(file_path)
            annotation_row = self.annotations[self.annotations['Image Name'] == image_name]
            if not annotation_row.empty:
                vote = annotation_row['Majority Vote Label'].values[0]
                annotators = annotation_row['Number of Annotators who Selected SSA (Out of 7)'].values[0]
                if vote == 'HP':
                    vote = 'No Cancer'
                    annotators = 7 - annotators
                elif vote == 'SSA':
                    vote = 'Cancer'
                self.vote_label.config(text=f"{annotators} out of 7 histopathologists voted: {vote}")

if __name__ == "__main__":
    model_name = 'ibrahim.keras'
    root = tk.Tk()
    app = CancerDetectorApp(root)
    root.mainloop()