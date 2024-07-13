import re


def parse_signals(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match the signal data and label
    pattern = re.compile(r'正在输出(\S+波)...\n\[(.*?)\]')

    data = []
    labels = {'正弦波': 0, '方波': 1, '三角波': 2, '锯齿波': 3}

    matches = pattern.findall(content)
    for match in matches:
        wave_type, values_str = match
        values = [float(val) for val in values_str.split(', ')]
        label = labels[wave_type]
        data.append((values, label))

    return data



if __name__ == '__main__':
    # Specify the path to your file
    file_path = 'datas.txt'
    signals_data = parse_signals(file_path)
    # Printing the parsed data to verify
    for signal, label in signals_data:
        print(f'Signal: {signal[:5]}... Length: {len(signal)}, Label: {label}')
