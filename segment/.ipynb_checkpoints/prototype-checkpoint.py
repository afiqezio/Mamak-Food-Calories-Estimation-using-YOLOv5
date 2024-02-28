import tkinter as tk
from tkinter import ttk, filedialog, Toplevel, Label, messagebox
from PIL import Image, ImageTk
import predict
from ttkthemes import ThemedStyle
from functools import partial
import os

class YOLOv5App:
    def __init__(self, root):
        self.root = root
        self.root.title("Mamak Breakfast Food Calories Estimation")

        # Apply a themed style
        self.style = ThemedStyle(self.root)
        self.style.set_theme("plastik")  # You can choose a different theme

        # Initialize variables
        self.image_path = None
        self.segmented_image = None

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Create a frame for the main content
        main_frame = ttk.Frame(self.root)
        main_frame.pack(pady=20)

        # Label to notify the user
        notification_label = ttk.Label(main_frame, text="Make sure you import a photo with a reference card in the picture.", font=('Helvetica', 12, 'italic'))
        notification_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Create a frame to contain the buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=10)

        # Style for the buttons
        style = ttk.Style()
        style.configure("TButton", padding=(10, 5, 10, 5), font='Helvetica 10')

        # Open Image Button
        open_button = ttk.Button(button_frame, text="Open Image", command=self.open_image)
        open_button.grid(row=0, column=0, padx=(0, 10))

        # Segment Image Button
        segment_button = ttk.Button(button_frame, text="Segment Image", command=self.segment_image)
        segment_button.grid(row=0, column=1, padx=(0, 10))

        # Create a frame to contain the image canvases and labels
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=2, column=0, columnspan=3, pady=20)

        # Canvas to display the original image
        self.canvas_original = tk.Canvas(image_frame, width=500, height=500)
        self.canvas_original.grid(row=0, column=0, padx=(0, 20))

        # Canvas to display the segmented image
        self.canvas_segmented = tk.Canvas(image_frame, width=500, height=500)
        self.canvas_segmented.grid(row=0, column=1)

        # Label to display the current file name
        self.filename_label = ttk.Label(main_frame, text="File Name: No file selected", font=('Helvetica', 12))
        self.filename_label.grid(row=3, column=0, columnspan=3, pady=10)

        # LabelFrame for combined weights and predictions
        combined_frame = ttk.LabelFrame(main_frame, text="Weights and Predictions", labelanchor="n", padding=(10, 10))
        combined_frame.grid(row=4, column=0, columnspan=3, pady=10)

        # Create a frame to contain the weights and predictions labels
        labels_frame = ttk.Frame(combined_frame)
        labels_frame.grid(row=0, column=0, pady=10)

        # Label to display weights
        self.weights_label = ttk.Label(labels_frame, text="Weights predictions will be displayed here.", font=('Helvetica', 12))
        self.weights_label.grid(row=0, column=0, padx=(0, 10), pady=10)

        # Label to display predictions
        self.prediction_label = ttk.Label(labels_frame, text="Calorie predictions will be displayed here.", font=('Helvetica', 12))
        self.prediction_label.grid(row=0, column=1, padx=(10, 0), pady=10)


        # LabelFrame for total predictions
        total_frame = ttk.LabelFrame(main_frame, text="Total Prediction", labelanchor="n", padding=(10, 10))
        total_frame.grid(row=5, column=0, columnspan=3, pady=10)

        # Label to display total prediction
        self.total_label = ttk.Label(total_frame, text="Total Prediction", font=('Helvetica', 12))
        self.total_label.pack(pady=10)

        
        
        
        

    def open_image(self):
        self.refresh_process()
        
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            self.image_path = file_path
            self.display_original_image()
            filename = os.path.basename(file_path)
            self.filename_label.config(text=f"File Name: {filename}")


    def display_original_image(self):
        # Open and display the original image on the canvas
        image = Image.open(self.image_path)
        image.thumbnail((500, 500))  # Resize image to fit canvas
        photo = ImageTk.PhotoImage(image)

        # Configure and create the canvas image
        self.canvas_original.config(width=photo.width(), height=photo.height())
        self.canvas_original.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas_original.image = photo

    def segment_image(self):
        if self.image_path:
            # Open a loading window
            loading_window = Toplevel(self.root)
            loading_window.title("Loading...")
            Label(loading_window, text="Segmenting Image, Please Wait...").pack(padx=20, pady=20)

            # Update the GUI to handle events
            self.root.update_idletasks()
            
            
            # Call the run function from predict.py with necessary parameters
            predict.run(
                weights="C:\\Users\\afiqe\\Downloads\\yolov5\\runs\\train-seg\\yolov53\\weights\\best.pt",
                source=self.image_path,
                project="C:\\Users\\afiqe\\Downloads\\yolov5\\runs\\predict-seg",
                name="exp",
                exist_ok=False,
                # Add other parameters as needed
            )
            # Update the segmented image variable
            self.segmented_image = predict.get_segmented_image()

            # Close the loading window
            loading_window.destroy()

            # Display the segmented image
            self.display_segmented_image()

            # Display predictions
            predictions = predict.get_prediction()
            minpred = predict.get_minprediction()
            maxpred = predict.get_maxprediction()
            
            weightrange = predict.get_weight()
            
            self.display_predictions(predictions,minpred, maxpred, weightrange)
        else:
            # Display a warning if no image is imported
            messagebox.showwarning("Warning", "Please import an image before pressing the 'Segment Image' button.")

    def display_segmented_image(self):
        if isinstance(self.segmented_image, bool):
            # If segmented_image is a boolean, set it to None
            self.segmented_image = None
        elif self.segmented_image is not None:
            # Open and resize the original image
            original_image = Image.open(self.image_path)
            original_image.thumbnail((500, 500))  # Resize image to fit canvas

            # Resize the segmented image to match the original image size
            segmented_image = Image.fromarray(self.segmented_image)
            segmented_image = segmented_image.resize(original_image.size)

            # Convert the segmented image to Tkinter PhotoImage
            segmented_photo = ImageTk.PhotoImage(segmented_image)

            # Update the existing canvas with the segmented image
            self.canvas_segmented.config(width=segmented_photo.width(), height=segmented_photo.height())
            self.canvas_segmented.create_image(0, 0, anchor=tk.NW, image=segmented_photo)
            self.canvas_segmented.image = segmented_photo

    def display_predictions(self, predictions, minpred, maxpred, weightrange):
        # Update the label with predictions
        total_value = 0  # Variable to store the total value

        if predictions is not None:
            predictions_text = "\n".join([f"{key}: {value} Kcal" for key, value in predictions.items()])
            self.prediction_label.config(text=predictions_text)
            
            weights_text = "\n".join([f"{key}: {value} g" for key, value in weightrange.items()])
            self.weights_label.config(text=f"{weights_text}")
            
            # Calculate total calorie range
            total_min = sum(minpred.values())
            total_max = sum(maxpred.values())
            total_value = f"{round(total_min,2)} - {round(total_max,2)} Kcal"
            
        else:
            # Show an error popup if no reference card is detected
            messagebox.showerror("Error", "No Reference Card!")

            # Clear the prediction label
            self.prediction_label.config(text="Error")
            self.weights_label.config(text="Error")

        # total_value = f"{round(total_min,2)} - {round(total_max,2)} Kcal"

        # Display the total value in the interface
        self.total_label.config(text=total_value)
        
    def refresh_process(self):
        # Reset the state of the application
        self.image_path = None
        self.segmented_image = None
        self.canvas_original.delete("all")  # Clear the original image canvas
        self.canvas_segmented.delete("all")  # Clear the segmented image canvas
        self.prediction_label.config(text="Calorie predictions will be displayed here.")
        self.weights_label.config(text="Weight predictions will be displayed here.")
        self.total_label.config(text="Total Prediction")

        # If there was a previous prediction, clear it
        predict.clear_previous_prediction()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x1000")
    app = YOLOv5App(root)
    root.mainloop()