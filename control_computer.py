import socket
from time import sleep
from predict import main as predict_
import ast
import matplotlib.pyplot as plt
def send_message(message, target_port = 7890):
    # 创建 UDP 套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    # 发送消息到广播地址和目标端口
    udp_socket.sendto(message.encode('utf-8'), ('192.168.137.242', target_port))

    # 关闭套接字
    udp_socket.close()


def receive_message(port=7890):
    # 创建 UDP 套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定到本地地址和端口
    dest_addr = ('192.168.137.1', port)
    udp_socket.bind(dest_addr)

    print(f"Listening on port {port}...")

    recv_data = ""
    close_switch = True
    while close_switch:
        sleep(0.001)
        # 接收数据
        data, sender_info = udp_socket.recvfrom(1024)

        # 打印发送者信息和接收到的数据
        print(f"Received message from {sender_info}: {data.decode('utf-8')}")

        if data:
            recv_data = data.decode('utf-8')
            close_switch = False

    # 关闭套接字
    udp_socket.close()
    return recv_data

def show_plt(signal):
    plt.plot(signal)
    plt.title("Time singal")
    plt.xlabel("point")
    plt.ylabel("V")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    signal = receive_message()
    signal = ast.literal_eval(signal)
    result = predict_(signal)
    sleep(2)
    send_message(str(result))
    show_plt(signal)
