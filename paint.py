import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np

class Paint(tk.Tk):
    def __init__(self, network):
        tk.Tk.__init__(self)
        self.network = network
        self.interval = 1/120
        self.pixel_size = 5
        self.quality_pixels = 27
        self.width_right = 50
        self.full_size = self.pixel_size * self.quality_pixels + self.width_right
        self.colors = ['#000000', '#520000', '#751700', '#953C00', '#A58400', '#A1A100', '#8BAE00', '#579100', '#46AE00',
                  '#00FF00']
        self.brush_colors = ['#000000', '#FFFFFF']
        self.brush_color = '#FFFFFF'
        self.chosen = 0
        self.lbls_console = []
        self.brush_size = 8

        self.title('Neural Network')
        self.resizable(width=False, height=False)
        self.columnconfigure(0, minsize=self.pixel_size * self.quality_pixels, weight=0)
        self.columnconfigure(1, minsize=self.full_size, weight=2)
        self.rowconfigure(0, minsize=self.full_size, weight=1)

        frm_left = tk.Frame(master=self)  # Left Frame
        frm_left.columnconfigure(0, minsize=self.pixel_size * self.quality_pixels, weight=0)
        frm_left.rowconfigure(0, minsize=self.pixel_size * self.quality_pixels, weight=0)
        frm_left.rowconfigure(1, minsize=self.width_right, weight=2)
        frm_left.grid(row=0, column=0, sticky='nsew', padx=100)

        frm_paint = tk.Frame(master=frm_left)  # Left Frame --> Paint Frame
        frm_paint.columnconfigure(0, minsize=self.full_size, weight=0)
        frm_paint.rowconfigure(0, minsize=self.full_size, weight=0)
        frm_paint.grid(row=0, column=0, padx=10, pady=10)

        self.cnv_paint = tk.Canvas(master=frm_paint, bg='#000000', width=27, height=27)
        self.cnv_paint.bind('<B1-Motion>', self.draw)
        self.cnv_paint.grid(row=0, column=0, sticky='nsew')

        frm_keys = tk.Frame(master=frm_left)  # Left Frame --> Keys Brush and Eraser
        btn_brush = tk.Button(master=frm_keys, text='Кисть', width=6, height=1, command=self.brash)
        btn_eraser = tk.Button(master=frm_keys, text='Ластик', width=6, height=1, command=self.eraser)
        btn_clear_all = tk.Button(master=frm_keys, text='Стереть', width=6, height=1, command=self.clear)
        btn_count = tk.Button(master=frm_keys, text='Распознать!', width=10, height=3, command=self.count)
        btn_brush.pack(side=tk.LEFT, padx=5, pady=5)
        btn_eraser.pack(side=tk.LEFT, padx=5, pady=5)
        btn_clear_all.pack(side=tk.LEFT, padx=5, pady=5)
        btn_count.pack(side=tk.RIGHT, padx=15, pady=5)
        frm_keys.grid(row=1, column=0, sticky='e')

        frm_right = tk.Frame(master=self)  # Right Frame
        frm_right.columnconfigure(0, minsize=0.2 * self.width_right, weight=0)
        frm_right.columnconfigure(1, minsize=0.8 * self.width_right, weight=2)
        frm_right.rowconfigure([i for i in range(10)], minsize=(self.pixel_size * self.quality_pixels + self.width_right) / 10, weight=1)
        frm_right.grid(row=0, column=1, sticky='nsew')

        for i in range(10):  # Right Frame --> Console labels
            lbl_num = tk.Label(master=frm_right, text=str(i))
            lbl_console = tk.Label(master=frm_right, width=20, height=2, bg=self.colors[0])
            lbl_num.grid(row=i, column=0, sticky='w', padx=5, pady=10)
            lbl_console.grid(row=i, column=1, sticky='w', pady=10)
            self.lbls_console.append(lbl_console)

    def brash(self):
        self.brush_color = self.brush_colors[1]

    def eraser(self):
        self.brush_color = self.brush_colors[0]

    def clear(self):
        self.cnv_paint.delete('all')

    def draw(self, event):
        self.cnv_paint.create_oval(event.x - self.brush_size,
                                   event.y - self.brush_size,
                                   event.x + self.brush_size,
                                   event.y + self.brush_size,
                                   fill=self.brush_colors[1], outline=self.brush_colors[1])

    def save_pic(self):
        x = self.winfo_rootx() + self.cnv_paint.winfo_x() + 157
        y = self.winfo_rooty() + self.cnv_paint.winfo_y() + 12
        x1 = x + self.cnv_paint.winfo_width() - 4
        y1 = y + self.cnv_paint.winfo_height() - 4
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L').resize((28,28))
        #img.save('num.png')
        img_array = np.array(img).ravel()#/255
        return np.matrix(img_array)

    def count(self):
        img = self.save_pic()
        prediction = self.network.Work(img)#[0]/100
        print(prediction)
        '''
        for i in range(10):
            if 0.0 <= prediction[i] < 0.1:
                self.lbls_console[i]['bg'] = self.colors[0]
            elif 0.1 <= prediction[i] < 0.2:
                self.lbls_console[i]['bg'] = self.colors[1]
            elif 0.2 <= prediction[i] < 0.3:
                self.lbls_console[i]['bg'] = self.colors[2]
            elif 0.3 <= prediction[i] < 0.4:
                self.lbls_console[i]['bg'] = self.colors[3]
            elif 0.4 <= prediction[i] < 0.5:
                self.lbls_console[i]['bg'] = self.colors[4]
            elif 0.5 <= prediction[i] < 0.6:
                self.lbls_console[i]['bg'] = self.colors[5]
            elif 0.6 <= prediction[i] < 0.7:
                self.lbls_console[i]['bg'] = self.colors[6]
            elif 0.7 <= prediction[i] < 0.8:
                self.lbls_console[i]['bg'] = self.colors[7]
            elif 0.8 <= prediction[i] < 0.9:
                self.lbls_console[i]['bg'] = self.colors[8]
            elif 0.9 <= prediction[i] <= 1.0:
                self.lbls_console[i]['bg'] = self.colors[9]
            else:
                self.lbls_console[i]['bg'] = self.colors[0]
                '''
        index = np.where(prediction == prediction.max())[1][0]
        print(index)
        for i in range(10):
            self.lbls_console[i]['bg'] = self.colors[0]
        self.lbls_console[index]['bg'] = self.colors[9]





    def updater(self):
        while True:
            self.update()

    def Run(self):
        self.mainloop()
'''
    def close(self):
        for task in self.tasks:
            task.cancel()
        self.loop.stop()
        self.destroy()'''


def main():
    app = Paint(2)
    app.Run()

if __name__ == '__main__':
    main()