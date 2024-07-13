# 导入模块
from machine import Pin, ADC, Timer, DAC
import math
import array
import DAC_
from time import sleep
#'triangle'
def adc_run(t = 50,pin = 34,sum_point = 128 ,debug = True ,debug_name = 'square',debug_freq = 400):
    '''
    freq 采样频率
    pin adc针脚
    '''

    # 定义ADC控制对象
    adc = ADC(Pin(pin))
    adc.atten(ADC.ATTN_11DB)  # 开启衰减，量程增大到3.3V

    res = []

    loop_switch = True
    # 定时器0中断函数
    def time0_irq(timer):
        nonlocal res,loop_switch  # 使用 nonlocal 关键字，如果 res 需要在 adc_run 函数的其它地方修改
        adc_value = adc.read()
        adc_voltage = 3.3 * adc_value / 4095
        # 将ADC值映射到-3.3V到3.3V
        adc_voltage_mapped = (adc_value - 2047) * (3.3 / 2047)
        print("ADC检测电压（映射）：%.2fV" % adc_voltage_mapped)
        res.append(round(adc_voltage_mapped, 2))
        if len(res) >= sum_point + 10 :
            res = res[10:]
            print(res)
            loop_switch = False
            # 关闭定时器
            timer.deinit()

    time0 = Timer(0)  # 创建time0定时器对象
    if debug:
        out_wave = DAC_.Out_Wave(25,debug_freq)
        out_wave.out_wave(debug_name)
    time0.init(period=t, mode=Timer.PERIODIC, callback=time0_irq)

    while loop_switch:
        pass

    return res
    
# 程序入口
if __name__ == "__main__":
    wave_list = ['sine','square','triangle','sawtooth']
    freq_list = [200,250,300,350,400,450,500,550,600,650,700,750,800]
    for i in wave_list:
        sleep(5)
        for k in freq_list:
            adc_run(debug_name=i,debug_freq=k+25)
            sleep(5)
