import tkinter as tk
import Car_and_Map
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sensor_log = list()
last_time_right = '0'
last_time_front = '0'
last_time_left = '0'


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
    with open('./data/new_train_data.txt', 'a') as fp:
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
front_sensor_label = tk.Label(right_part, text = '0')
front_sensor_label.pack()
right_sensor_label = tk.Label(right_part, text = '0')
right_sensor_label.pack()

turn_label = tk.Label(right_part, text='Turn Degree')
turn_label.pack()

turn_degree_entry = tk.Entry(right_part)
turn_degree_entry.pack()

run_btn = tk.Button(right_part, text='move', command=move)
run_btn.pack()

save_btn = tk.Button(right_part, text='save', command=save)
save_btn.pack()


# load map
my_map, car_plot, car_descriptor, head_descriptor = load_map()
my_car = Car_and_Map.Car(my_map.car_init_position, my_map.car_init_degree)
# my_car.reset()
# my_car.draw_car(car_descriptor, head_descriptor)

window.mainloop()

