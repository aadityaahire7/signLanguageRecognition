import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('C:\\Users\\Aaditya Ahire\\Desktop\\signLanguage\\action.h5')  # Replace with the actual path


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")

        # Set the initial size of the window
        self.root.geometry("500x500")

        # Create and configure GUI elements
        self.title_label = tk.Label(root, text="Sign Language Recognition", font=("Arial", 20, "bold"), pady=10)
        self.title_label.pack(pady=10)  # Add some vertical padding

        self.label = tk.Label(root)
        self.label.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse and Select the Image", command=self.load_image)
        self.browse_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict the Selected Image", command=self.predict_image)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 16))  # Increase font size
        self.result_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(initialdir="/", title="Select File",
                                               filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.image = cv2.resize(self.original_image, (28, 28))
            self.image = self.image / 255.0  # Normalize pixel values
            self.image = self.image.reshape((1, 28, 28, 1))
            

            # Display the selected image
            image = Image.fromarray(self.original_image)
            photo = ImageTk.PhotoImage(image)
            self.label.configure(image=photo)
            self.label.image = photo
        else:
            self.result_label.config(text='No image selected.')

    def predict_image(self):
        if hasattr(self, 'image'):
            predictions = model.predict(self.image)
            predicted_class = np.argmax(predictions)

            class_to_letter = {
                0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
                6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
                18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X'
            }

            predicted_letter = class_to_letter[predicted_class]
            self.result_label.config(text=f'Predicted Letter: {predicted_letter}')
        else:
            self.result_label.config(text='Please select an image first.')


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
