import tkinter as tk
import random

import torch

from gui.EEGModelThread import EEGModelThread
from gui.EEGThread import EEGThread


class Booth:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.root = None
        self.canvas = None
        self.cursor = None
        self.capture = None
        self.label = None
        self.label_output = None

        self.cursor_size = 5
        self.init_cursor_x = 280
        self.init_cursor_y = 280
        self.cursor_x = self.init_cursor_x
        self.cursor_y = self.init_cursor_y
        self.boxes = []

        self.init_capture()
        self.init_model(model, device)
        self.init_tk()
        self.update()

    def init_tk(self):
        print("[!] Creating tkinter GUI")
        self.root = tk.Tk()
        self.root.geometry("640x480")
        self.root.title("EEG Motor Imagery")
        self.root.bind("<Key>", self.update_cursor)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()
        self.label_output = tk.StringVar()
        self.draw_cursor()
        self.draw_boxes(3)
        self.root.mainloop()

    def draw_boxes(self, num_boxes):
        for _ in range(num_boxes):
            x1 = random.randint(20, 620)
            y1 = random.randint(20, 460)
            x2 = x1 + 60
            y2 = y1 + 60
            box = self.canvas.create_rectangle(x1, y1, x2, y2, fill="yellow", outline="orange")
            self.boxes.append(box)

    def draw_cursor(self):
        if self.cursor:
            self.canvas.delete(self.cursor)

        x1 = self.cursor_x - self.cursor_size
        y1 = self.cursor_y - self.cursor_size
        x2 = self.cursor_x + self.cursor_size
        y2 = self.cursor_y + self.cursor_size

        self.cursor = self.canvas.create_oval(x1, y1, x2, y2, fill="red")

        return self.cursor

    def update_cursor(self, event):

        command = self.model.current_output

        if event.keysym == "Up":
            self.cursor_y -= 10
        elif event.keysym == "Left":
            self.cursor_x -= 10
        elif event.keysym == "Right":
            self.cursor_x += 10
        elif event.keysym == "Down":
            self.cursor_y += 10

        self.label_output = event.keysym
        print(command)
        # print(self.label_output)
        self.draw_cursor()
        self.check_collision()

    def check_collision(self):
        cursor_bbox = self.canvas.bbox(self.cursor)

        for box_id in self.boxes:
            box_bbox = self.canvas.bbox(box_id)

            if self.is_overlap(cursor_bbox, box_bbox):
                print("Collision detected!")
                self.reset_cursor()
                self.clear_boxes()
                self.draw_boxes(3)  # TODO: Function to receive object count
                break

    @staticmethod
    def is_overlap(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        if x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1:
            return False
        return True

    def reset_cursor(self):
        self.cursor_x = self.init_cursor_x
        self.cursor_y = self.init_cursor_y

        if self.cursor:
            self.canvas.delete(self.cursor)
            self.draw_cursor()

    def clear_boxes(self):
        for box_id in self.boxes:
            self.canvas.delete(box_id)
        self.boxes = []

    def init_model(self, model, device):
        print("[!] Model initialization")
        self.model = EEGModelThread(self.capture, model, device)
        self.model.start()

    def init_capture(self):
        print("[!] Async capture initialization")
        self.capture = EEGThread()
        self.capture.start()

    def update(self):
        pass

    def on_close(self):
        print("Ending processes")
        self.capture.stop()
        self.model.stop()
        del self.capture
        self.root.destroy()
        print("Processes closed")


if __name__ == '__main__':
    gui = Booth()
