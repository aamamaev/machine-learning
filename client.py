import time
import socket


class ClientError(Exception):
    pass

class Client():
    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout

    def put(self, metric, value, timestamp=None):

        if timestamp == None:
            timestamp = str(int(time.time()))

        message = 'put ' + ' '.join([str(metric), str(value), str(timestamp)]) + '\n'

        sock = socket.socket()
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        sock.sendall(message.encode())
        data = sock.recv(1024).decode('utf8')
        if data == 'error\nwrong command\n\n':
            raise ClientError

        sock.close()

    def get(self, message):

        message = 'get ' + message + "\n"

        sock = socket.socket()
        sock.settimeout(self.timeout)
        sock.connect((str(self.host), self.port))
        sock.sendall(message.encode())
        data = sock.recv(1024).decode('utf8')

        if data == 'error\nwrong command\n\n':
            raise ClientError

        v_dict = {}

        def put(key, value):
            if key in v_dict:
                v_dict[key].append(value)
            else:
                v_dict[key] = [value]

        for string in data.split('\n')[1:-2]:
            key, value1, value2 = string.split(' ')
            put(key, (int(value2), float(value1)))

        sock.close()

        return v_dict



if __name__ == '__main__':
    c = Client('127.0.0.1', 9999)
    c.put("test", 0.5, 1)
    c.put("test", 0.4, 2)
    c.put("load", 301, 3)
    print('Запрос всех данных с сревера: \n', c.get('*'))
    print('Запрос метрики test: \n', c.get('test'))