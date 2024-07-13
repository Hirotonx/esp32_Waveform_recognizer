import math
from machine import DAC, Pin, Timer
import array

class Out_Wave():
    def __init__(self, dac_pin, frequency, amplitude=128, offset=128):
        '''
        self.dac: DAC对象
        frequency: 频率
        amplitude: 振幅
        offset: 偏移
        '''
        self.dac = DAC(Pin(dac_pin))
        self.freq = frequency
        self.amplitude = amplitude
        self.offset = offset
        self.samples = 199
        # 定时器周期应等于波形周期除以采样点数
        self.period_us = int((1 / self.freq) * 1e6 / self.samples)

    def generate_sine_wave(self):
        buffer = array.array('B', (int(self.offset + self.amplitude * math.sin(2 * math.pi * i / self.samples)) for i in range(self.samples)))
        #print(buffer)
        timer = Timer(1)
        index = 0

        # 定时器回调函数
        def output_wave(timer):
            nonlocal index
            self.dac.write(buffer[index])
            index = (index + 1) % self.samples

        # 启动定时器，定期调用output_wave函数
        timer.init(period=self.period_us, mode=Timer.PERIODIC, callback=output_wave)
        print("正在输出正弦波...")

    def generate_square_wave(self):
        buffer = array.array('B', (0 if i < self.samples // 2 else 255 for i in range(self.samples)))
        #print(buffer)
        timer = Timer(1)
        index = 0

        # 定时器回调函数
        def output_wave(timer):
            nonlocal index
            self.dac.write(buffer[index])
            index = (index + 1) % self.samples

        # 启动定时器，定期调用output_wave函数
        timer.init(period=self.period_us, mode=Timer.PERIODIC, callback=output_wave)
        print("正在输出方波...")

    def generate_triangle_wave(self):

        #buffer = array.array('B', [int(self.offset + self.amplitude * ((2 * i / self.samples) - 1)) if i < (self.samples // 2) + 1 else int(self.offset - self.amplitude * ((2 * (self.samples - i) / self.samples) - 1)) for i in range(self.samples)])
        buffer = array.array('B', (int(self.offset + self.amplitude * (4 * abs(i / self.samples - 0.5) - 1)) for i in
                                   range(self.samples)))
        #print(buffer)
        timer = Timer(1)
        index = 0

        # 定时器回调函数
        def output_wave(timer):
            nonlocal index
            self.dac.write(buffer[index])
            index = (index + 1) % self.samples

        # 启动定时器，定期调用output_wave函数
        timer.init(period=self.period_us, mode=Timer.PERIODIC, callback=output_wave)
        print("正在输出三角波...")

    def generate_sawtooth_wave(self):

        buffer = array.array('B', [int(self.offset + self.amplitude * ((2 * i / self.samples) - 1)) if i < (self.samples // 2) + 1 else int(self.offset - self.amplitude * ((2 * (self.samples - i) / self.samples) - 1)) for i in range(self.samples)])
        #print(buffer)
        timer = Timer(1)
        index = 0

        # 定时器回调函数
        def output_wave(timer):
            nonlocal index
            self.dac.write(buffer[index])
            index = (index + 1) % self.samples

        # 启动定时器，定期调用output_wave函数
        timer.init(period=self.period_us, mode=Timer.PERIODIC, callback=output_wave)
        print("正在输出锯齿波...")

    def out_wave(self,out_name:str):
        if out_name == 'sine':
            self.generate_sine_wave()

        elif out_name == 'square':
            self.generate_square_wave()

        elif out_name == 'triangle':
            self.generate_triangle_wave()

        elif out_name == 'sawtooth':
            self.generate_sawtooth_wave()

        else:
            print('输入的波形有误')

if __name__ == '__main__':
    # 示例使用
    wave_generator = Out_Wave(dac_pin=25, frequency=50)
    #wave_generator.generate_sine_wave()
    #wave_generator.generate_square_wave()
    #wave_generator.generate_triangle_wave()
    wave_generator.generate_sawtooth_wave()