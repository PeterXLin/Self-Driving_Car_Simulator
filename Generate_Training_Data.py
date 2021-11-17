import tkinter as tk
import Car_and_Map
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import Model
import random

sensor_log = list()
last_time_right = '0'
last_time_front = '0'
last_time_left = '0'
stop = False


def move():
    global last_time_front, last_time_right, last_time_left
    this_time_turn = float(turn_degree_entry.get())
    my_car.move(this_time_turn)
    my_car.draw_car(car_descriptor, head_descriptor)

    if my_car.detect_collision(my_map.border_linear_equations):
        my_car.reset()
        my_car.draw_car(car_descriptor, head_descriptor)
        sensor_value = my_car.sensor(my_map.border_linear_equations)
        set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
        update_sensor_output(sensor_value)
        sensor_log.clear()
    elif my_car.arrive(my_map.dest_up_left, my_map.dest_botton_right):
        save()
        my_car.reset()
        my_car.draw_car(car_descriptor, head_descriptor)
        sensor_value = my_car.sensor(my_map.border_linear_equations)
        set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
        update_sensor_output(sensor_value)
        sensor_log.clear()
    else:
        sensor_value = my_car.sensor(my_map.border_linear_equations)
        update_sensor_output(sensor_value)
        sensor_log.append(last_time_front+' '+last_time_right+' '+last_time_left+' '+str(this_time_turn))
        set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
    car_canvas.draw()


def self_drive():
    """self drive based on the pre train MLP model"""
    my_map, car_plot, car_descriptor, head_descriptor = load_map()
    my_car = Car_and_Map.Car(my_map.car_init_position, my_map.car_init_degree)

    sensor_value = my_car.sensor(my_map.border_linear_equations)
    update_sensor_output(sensor_value)
    # load model
    my_model = Model.load_model('sigmoid_model_2.txt')
    tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))

    while True:
        if stop:
            break

        # TODO here
        turn_degree = my_model.predict(tmp_input) * (random.random() / 2 + 0.75)
        if turn_degree < -40:
            turn_degree = -40
        elif turn_degree >= 40:
            turn_degree = 40

        this_time_turn = turn_degree
        my_car.move(turn_degree)
        my_car.draw_car(car_descriptor, head_descriptor)

        if my_car.detect_collision(my_map.border_linear_equations):
            # report collision and reset car position
            # tell user click self drive button again to restart
            sensor_log.clear()
            my_car.reset()
            my_car.draw_car(car_descriptor, head_descriptor)
            sensor_value = my_car.sensor(my_map.border_linear_equations)
            # for next turn input
            tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))
            update_sensor_output(sensor_value)

            set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
            continue
        if my_car.arrive(my_map.dest_up_left, my_map.dest_botton_right):
            # show congratulation message
            save()
            my_car.reset()
            my_car.draw_car(car_descriptor, head_descriptor)
            sensor_value = my_car.sensor(my_map.border_linear_equations)
            update_sensor_output(sensor_value)
            tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))
            set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
            sensor_log.clear()

            continue

        sensor_value = my_car.sensor(my_map.border_linear_equations)
        tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))
        update_sensor_output(sensor_value)
        sensor_log.append(last_time_front + ' ' + last_time_right + ' ' + last_time_left + ' ' + str(this_time_turn))
        set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
        car_canvas.draw()


def self_drive_by_rule():
    """self drive based on the pre train MLP model"""
    my_map, car_plot, car_descriptor, head_descriptor = load_map()
    my_car = Car_and_Map.Car(my_map.car_init_position, my_map.car_init_degree)

    sensor_value = my_car.sensor(my_map.border_linear_equations)
    update_sensor_output(sensor_value)
    # load model
    tmp_input = np.array([sensor_value[1], sensor_value[2], sensor_value[0]])

    while True:
        if stop:
            break

        # TODO here
        if tmp_input[1] < 6:
            turn_degree = random.random() * -20
        elif tmp_input[2] < 6:
            turn_degree = random.random() * 20
        elif tmp_input[0] >= 15:
            turn_degree = random.random()*10 - 5
        else:
            turn_degree = random.random()*(tmp_input[1] - tmp_input[2])*2.5
        if turn_degree < -40:
            turn_degree = -40
        elif turn_degree >= 10:
            turn_degree = 40

        this_time_turn = turn_degree
        my_car.move(turn_degree)
        my_car.draw_car(car_descriptor, head_descriptor)

        if my_car.detect_collision(my_map.border_linear_equations):
            # report collision and reset car position
            # tell user click self drive button again to restart
            sensor_log.clear()
            my_car.reset()
            my_car.draw_car(car_descriptor, head_descriptor)
            sensor_value = my_car.sensor(my_map.border_linear_equations)
            # for next turn input
            tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))
            update_sensor_output(sensor_value)

            set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
            continue
        if my_car.arrive(my_map.dest_up_left, my_map.dest_botton_right):
            # show congratulation message
            save()
            my_car.reset()
            my_car.draw_car(car_descriptor, head_descriptor)
            sensor_value = my_car.sensor(my_map.border_linear_equations)
            update_sensor_output(sensor_value)
            tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))
            set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
            sensor_log.clear()

            continue

        sensor_value = my_car.sensor(my_map.border_linear_equations)
        tmp_input = np.array(([sensor_value[1], sensor_value[2], sensor_value[0]]))
        update_sensor_output(sensor_value)
        sensor_log.append(last_time_front + ' ' + last_time_right + ' ' + last_time_left + ' ' + str(this_time_turn))
        set_sensor_log(sensor_value[0], sensor_value[1], sensor_value[2])
        car_canvas.draw()


def break_run():
    global stop
    stop = True


def set_sensor_log(left_value, front_value, right_value):
    global last_time_left, last_time_front, last_time_right
    last_time_left = str(round(left_value, 4))
    last_time_front = str(round(front_value, 4))
    last_time_right = str(round(right_value, 4))


def update_sensor_output(new_value: list):
    left_sensor_label['text'] = str(round(new_value[0], 3))
    front_sensor_label['text'] = str(round(new_value[1], 3))
    right_sensor_label['text'] = str(round(new_value[2], 3))


def load_map():
    car_figure.clear()
    car_plot = car_figure.add_subplot(111)
    my_map = Car_and_Map.Map('./data/軌道座標點.txt')
    car_ds, head_ds = my_map.draw_map_and_car_start(car_plot)
    car_canvas.draw()
    return my_map, car_plot, car_ds, head_ds


def save():
    with open('./data/drive_by_rule.txt', 'a') as fp:
        for line in sensor_log:
            fp.write(line + '\n')


window = tk.Tk()

window.geometry("1000x800")
window.configure(bg="#FFFFFF")
window.title("Self driving car")
window.resizable(False, False)

car_figure = Figure(figsize=(7.5, 7), dpi=100)
car_canvas = FigureCanvasTkAgg(car_figure, master=window)  # A tk.DrawingArea.
car_canvas.draw()
car_canvas.get_tk_widget().place(x=0, y=50, width=750, height=700)

right_part = tk.Frame(window, width=250, height=800)
right_part.place(x=750, y=0)

left_sensor_label = tk.Label(right_part, text='0')
left_sensor_label.pack()
front_sensor_label = tk.Label(right_part, text='0')
front_sensor_label.pack()
right_sensor_label = tk.Label(right_part, text='0')
right_sensor_label.pack()

turn_label = tk.Label(right_part, text='Turn Degree')
turn_label.pack()

turn_degree_entry = tk.Entry(right_part)
turn_degree_entry.pack()

run_btn = tk.Button(right_part, text='self drive', command=self_drive_by_rule)
run_btn.pack()

save_btn = tk.Button(right_part, text='break', command=break_run)
save_btn.pack()


# load map
my_map, car_plot, car_descriptor, head_descriptor = load_map()
my_car = Car_and_Map.Car(my_map.car_init_position, my_map.car_init_degree)
# my_car.reset()
# my_car.draw_car(car_descriptor, head_descriptor)

window.mainloop()

