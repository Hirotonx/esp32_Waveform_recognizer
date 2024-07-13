import network
from time import sleep
import socket 

def connect():
    
    wlan = network.WLAN(network.STA_IF) # create station interface
    wlan.active(True)       # activate the interface
    if not wlan.isconnected():
        print('connecting ...')
        #wlan.scan()             # scan for access points
        wlan.connect('hu', '12345678..')
        while not wlan.isconnected():
            pass 
    print('network config :', wlan.ifconfig())

def create_udp_socket():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    udp_socket.bind(("0.0.0.0",7890))
    return udp_socket
    
def udp_sent(send_data):
    connect()
    # # 1. 创建udp套接字
    udp_socket = create_udp_socket()

    # 2. 准备接收方的地址
    dest_addr = ('192.168.137.1', 7890)

    # 3. 从键盘获取数据
    #send_data = "hello world"

    # 4. 发送数据到指定的电脑上
    udp_socket.sendto(send_data.encode('utf-8'), dest_addr)

    # 5. 关闭套接字
    udp_socket.close()
    
def udp_receive():
    connect()
    # # 1. 创建udp套接字
    udp_socket = create_udp_socket()
    recevie_switch = True

    while recevie_switch:
        recv_data, sender_info =udp_socket.recvfrom(1024)
        if recv_data != None:
            recv_data = recv_data.decode('utf-8')
            print('发送者',sender_info,'信息',recv_data)
            recevie_switch = False
        sleep(0.01)
    return recv_data

if __name__ == '__main__':
    pass