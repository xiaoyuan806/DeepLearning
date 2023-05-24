
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
class Nerve(nn.Module):
    def __init__(self):
        super(Nerve, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

def open_file():
    result = messagebox.askyesno('提示', '只能识别飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车')
    if result == True:
        file_path = filedialog.askopenfilename(initialdir='C:/', filetypes=(('Image files', '*.jpg;*.png'),))
        if file_path:
            img = Image.open(file_path)
            image = img.convert('RGB')
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
            image = transform(image)
            model = torch.load("nerve_49.pth", map_location=torch.device('cuda'))
            image = torch.reshape(image, (1, 3, 32, 32))
            image = image.cuda()
            model.eval()
            with torch.no_grad():
                output = model(image)
            x = output.argmax(1)
            y = x.item()
            if y == 0:
                messagebox.showinfo("结果", "当前照片中的物体是飞机")
            elif y == 1:
                messagebox.showinfo("结果", "当前照片中的物体是汽车")
            elif y == 2:
                messagebox.showinfo("结果", "当前照片中的物体是鸟")
            elif y == 3:
                messagebox.showinfo("结果", "当前照片中的物体是猫")
            elif y == 4:
                messagebox.showinfo("结果", "当前照片中的物体是鹿")
            elif y == 5:
                messagebox.showinfo("结果", "当前照片中的物体是狗")
            elif y == 6:
                messagebox.showinfo("结果", "当前照片中的物体是青蛙")
            elif y == 7:
                messagebox.showinfo("结果", "当前照片中的物体是马")
            elif y == 8:
                messagebox.showinfo("结果", "当前照片中的物体是船")
            else:
                messagebox.showinfo("结果", "当前照片中的物体是卡车")

        else:
            print("No file selected.")

root = tk.Tk()
root.title('图片物体识别')

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()


x = int((screen_width - 400) / 2)
y = int((screen_height - 300) / 2)

root.geometry(f"400x300+{x}+{y}")
root.resizable(False, False)
root.iconbitmap('logo.ico')
label = tk.Label(root, text='选择一张图片进行识别', font=('Arial', 16))
label.pack(pady=20)

button = tk.Button(root, text='Open', font=('Arial', 14), bg='#409EFF', fg='white', command=open_file)
button.pack(pady=20)

root.mainloop()