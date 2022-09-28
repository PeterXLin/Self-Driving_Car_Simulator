# the GUI layout code was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer

from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import Car_and_Map
import Model
import numpy as np
from tkinter import filedialog as fd

# --------------global variable--------------
my_config = {
    'map_path': '',
    'log_path': ''
}

left_sensor = ''
right_sensor = ''
front_sensor = ''
# ----------------- backend ------------------------------


def load_map():
    """draw map and car in figure"""
    car_figure.clear()
    car_plot = car_figure.add_subplot(111)
    my_map = Car_and_Map.Map(my_config['map_path'])
    car_descriptor, head_descriptor = my_map.draw_map_and_car_start(car_plot)
    car_canvas.draw()
    return my_map, car_plot, car_descriptor, head_descriptor


def load_record():
    """load log, include 4d and 6d"""
    with open(my_config['log_path'], 'r') as fp:
        first_line = fp.readline()
        dim = len(first_line.split(' '))
    if dim == 4:
        load_4d_record()
    elif dim == 6:
        load_6d_record()


def load_6d_record():
    """load log(6d) and show in figure"""
    my_map, car_plot, car_descriptor, head_descriptor = load_map()
    my_car = Car_and_Map.Car(my_map.car_init_position, my_map.car_init_degree)

    with open(my_config['log_path'], 'r') as fp:
        for records in fp.readlines():
            tmp_list = records.split(' ')
            update_sensor_output([float(tmp_list[4]), float(tmp_list[2]), float(tmp_list[3])])
            my_car.move(float(tmp_list[5]))
            my_car.x = float(tmp_list[0])
            my_car.y = float(tmp_list[1])
            my_car.draw_car(car_descriptor, head_descriptor)
            car_canvas.draw()

            if my_car.detect_collision(my_map.border_linear_equations) or \
                    my_car.arrive(my_map.dest_up_left, my_map.dest_botton_right):
                break


def load_4d_record():
    """load log(4d) and plot in figure"""
    my_map, car_plot, car_descriptor, head_descriptor = load_map()
    my_car = Car_and_Map.Car(my_map.car_init_position, my_map.car_init_degree)

    with open(my_config['log_path'], 'r') as fp:
        for records in fp.readlines():
            tmp_list = records.split(' ')
            # sensor_value = my_car.sensor(my_map.border_linear_equations)
            # use  log to set sensor_value
            update_sensor_output([float(tmp_list[2]), float(tmp_list[0]), float(tmp_list[1])])
            # turn_degree = records.split(' ')[3]
            my_car.move(float(tmp_list[3]))
            my_car.draw_car(car_descriptor, head_descriptor)
            car_canvas.draw()

            if my_car.detect_collision(my_map.border_linear_equations) or \
                    my_car.arrive(my_map.dest_up_left, my_map.dest_botton_right):
                break


def self_drive():
    """self drive based on the pre train MLP model"""
    my_map, car_plot, car_descriptor, head_descriptor = load_map()
    my_car = Car_and_Map.Car(my_map.car_init_position, my_map.car_init_degree)

    sensor_value = my_car.sensor(my_map.border_linear_equations)
    update_sensor_output(sensor_value)
    # tmp_input is the input of model, used to predict the angle of steering wheels
    tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))
    # load model
    my_model = Model.load_model('sigmoid_model_3_data_from_me.txt')

    while True:
        turn_degree = Car_and_Map.valid_steering_wheel_angle(my_model.predict(tmp_input))
        set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
        save_log(front_sensor + ' ' + right_sensor + ' ' + left_sensor + ' ' + str(turn_degree), 'train4D.txt')
        save_log(str(my_car.x) + ' ' + str(my_car.y) + ' ' + front_sensor + ' ' + right_sensor + ' ' + left_sensor + ' '
                 + str(turn_degree), 'train6D.txt')

        my_car.move(turn_degree)
        my_car.draw_car(car_descriptor, head_descriptor)
        car_canvas.draw()

        if my_car.detect_collision(my_map.border_linear_equations):
            # if car accident happened, reset car position and restart
            my_car.reset()
            sensor_value = my_car.sensor(my_map.border_linear_equations)
            update_sensor_output(sensor_value)
            tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))
            continue
        if my_car.arrive(my_map.dest_up_left, my_map.dest_botton_right):
            # 加入到達終點的這一次的車子座標點
            set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
            save_log(front_sensor + ' ' + right_sensor + ' ' + left_sensor + ' ' + str(turn_degree), 'train4D.txt')
            save_log(
                str(my_car.x) + ' ' + str(my_car.y) + ' ' + front_sensor + ' ' + right_sensor + ' ' + left_sensor + ' '
                + str(turn_degree), 'train6D.txt')
            break

        # for next move
        sensor_value = my_car.sensor(my_map.border_linear_equations)
        tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))
        update_sensor_output(sensor_value)


def set_sensor_log(left_value, front_value, right_value):
    global left_sensor, right_sensor, front_sensor
    left_sensor = str(round(left_value, 4))
    front_sensor = str(round(front_value, 4))
    right_sensor = str(round(right_value, 4))


def save_log(log, file_name):
    with open('./data/' + file_name, 'a') as fp:
        fp.write(log + '\n')


def update_sensor_output(new_value: list):
    """update sensor label"""
    canvas.itemconfig(left_sensor_value_entry, text=round(new_value[0], 2))
    canvas.itemconfig(front_sensor_value_entry, text=round(new_value[1], 2))
    canvas.itemconfig(right_sensor_value_entry, text=round(new_value[2], 2))


def select_map():
    my_config['map_path'] = select_file()


def select_log():
    my_config['log_path'] = select_file()


def select_file():
    filetypes = (
        ('text files', '*.txt'),
    )
    file_path = fd.askopenfilename(
        title='Open a file',
        initialdir='./data',
        filetypes=filetypes)
    return file_path


# ---------------------------------------------------------
window = Tk()

window.geometry("1000x800")
window.configure(bg="#FFFFFF")
window.title("Self driving car")

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 800,
    width = 1000,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    250.0,
    0.0,
    1000.0,
    100.0,
    fill="#8D99AE",
    outline="")

canvas.create_rectangle(
    250.0,
    100.0,
    1000.0,
    800.0,
    fill="#EDF2F4",
    outline="")

canvas.create_rectangle(
    0.0,
    0.0,
    250.0,
    800.0,
    fill="#2B2D42",
    outline="")

button_image_1 = PhotoImage(
    file="assets/button_1.png")
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=load_record,
    relief="flat"
)
button_1.place(
    x=50.0,
    y=560.0,
    width=150.0,
    height=60.0
)

button_image_2 = PhotoImage(
    file="assets/button_2.png")
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=self_drive,
    relief="flat"
)
button_2.place(
    x=50.0,
    y=640.0,
    width=150.0,
    height=60.0
)

canvas.create_text(
    250.0,
    33.0,
    anchor="nw",
    text="Left",
    fill="#FFFFFF",
    font=("SuezOne Regular", 24 * -1)
)

canvas.create_text(
    500.0,
    33.0,
    anchor="nw",
    text="Front",
    fill="#FFFFFF",
    font=("SuezOne Regular", 24 * -1)
)

canvas.create_text(
    750.0,
    33.0,
    anchor="nw",
    text="Right",
    fill="#FFFFFF",
    font=("SuezOne Regular", 24 * -1)
)

right_sensor_value_entry = canvas.create_text(
    900.0,
    33.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("SuezOne Regular", 24 * -1)
)

front_sensor_value_entry = canvas.create_text(
    650.0,
    33.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("SuezOne Regular", 24 * -1)
)

left_sensor_value_entry = canvas.create_text(
    400.0,
    33.0,
    anchor="nw",
    text="0",
    fill="#FFFFFF",
    font=("SuezOne Regular", 24 * -1)
)

button_image_3 = PhotoImage(
    file="assets/button_3.png")
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=select_log,
    relief="flat"
)
button_3.place(
    x=50.0,
    y=250.0,
    width=150.0,
    height=40.0
)

button_image_4 = PhotoImage(
    file="assets/button_4.png")
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=select_map,
    relief="flat"
)
button_4.place(
    x=50.0,
    y=124.0,
    width=150.0,
    height=40.0
)
# ------------------------- try new things
car_figure = Figure(figsize=(7.5, 7), dpi=100)
car_canvas = FigureCanvasTkAgg(car_figure, master=window)  # A tk.DrawingArea.
car_canvas.draw()
car_canvas.get_tk_widget().place(x=250, y=100, width=750, height=700)
# ---------------------------------

window.resizable(False, False)
window.mainloop()
