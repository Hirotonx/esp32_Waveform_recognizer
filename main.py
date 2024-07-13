from ADC import adc_run
from net import connect,udp_sent,udp_receive
from machine import Pin
import tm1637
from time import sleep

'''
连线:数码管16->CLK 17->DIO
    按键 k3 k4 26 27
    ADC -> DAC
    34->25

'''
#['sine','square','triangle','sawtooth']
#链接wifi
connect()
#启动数码管
smg=tm1637.TM1637(clk=Pin(16),dio=Pin(17))
smg.show("----")

key3 = Pin(26, Pin.IN, Pin.PULL_UP)
key4 = Pin(27, Pin.IN, Pin.PULL_UP)

show_list = [0,0,0] # label PVV T 
show_index = 0
# 计算峰峰值
def calculate_peak_to_peak(signal):
    max_value = max(signal)
    min_value = min(signal)
    peak_to_peak = max_value - min_value
    peak_to_peak = round(peak_to_peak, 2)
    # 分离整数部分
    integer_part = int(peak_to_peak)

    # 分离小数部分，最多两位小数
    decimal_part = round(peak_to_peak * 100) % 100

    # 将整数和小数部分转换为元组
    number_tuple = (integer_part, decimal_part)
    return number_tuple

# 计算周期
def calculate_period(signal, sampling_period):
    threshold = (max(signal) + min(signal)) / 2  # 阈值设为信号的平均值
    crossings = []

    # 找到信号穿过阈值的点
    for i in range(1, len(signal)):
        if (signal[i-1] < threshold and signal[i] >= threshold) or (signal[i-1] >= threshold and signal[i] < threshold):
            # 线性插值以找到更准确的交叉点
            t = (threshold - signal[i-1]) / (signal[i] - signal[i-1])
            crossing_point = i - 1 + t
            crossings.append(crossing_point)

    if len(crossings) < 2:
        return (0, 0)  # 如果没有足够的交叉点，则无法计算周期

    # 计算周期
    periods = []
    for i in range(1, len(crossings)):
        periods.append((crossings[i] - crossings[i-1]) * sampling_period)

    average_period = sum(periods) / len(periods)
    average_period = round(average_period, 2)

    # 分离整数部分
    integer_part = int(average_period)

    # 分离小数部分，最多两位小数
    decimal_part = round((average_period - integer_part) * 100)

    # 将整数和小数部分转换为元组
    number_tuple = (integer_part, decimal_part)
    return number_tuple



#回调函数
def key3_irq(key3):
    global show_index
    sleep(0.01)
    if key3.value() == 0:
        if show_index < 2:
            show_index +=1
        else:
            show_index = 0

def key4_irq(key4):
    global show_index
    sleep(0.01)
    if key4.value() == 0:
        if show_index > 0:
            show_index -=1
        else:
            show_index = 2

key3.irq(key3_irq, Pin.IRQ_FALLING)  # 配置key3外部中断，下降沿触发
key4.irq(key4_irq, Pin.IRQ_FALLING)  # 配置key4外部中断，下降沿触发

def show():
    global show_index,show_list
    if show_index == 0:
        smg.show("---" + str(show_list[0]))

    else:
        if show_list[show_index][0]>99 or show_list[show_index][1]>99:
            if show_list[show_index][0] <9999:
                smg.number(show_list[show_index][0])
            else:
                smg.show("FFFF")
        else:

            smg.numbers(show_list[show_index][0],show_list[show_index][1])

if __name__ == '__main__':
    show_index = 0
    # 采样周期
    sampling_period = 50  # 50微秒
    wave_list = ['sine','square','triangle','sawtooth']

    t = 50
    adc_pin = 34
    sum_point = 128
    debug = True
    debug_name = 'sawtooth'
    debug_freq = 400

    signal = adc_run(t ,adc_pin,sum_point ,debug,debug_name ,debug_freq)
    show_list[1] = calculate_peak_to_peak(signal)
    show_list[2]  = calculate_period(signal, sampling_period)

    udp_sent(str(signal))
    smg.show('AAAA')
    show_list[0] = udp_receive()
    #print(wave_list[show_list[0]])
    print(show_list)

    #
    while True:
        sleep(0.01)
        show()

