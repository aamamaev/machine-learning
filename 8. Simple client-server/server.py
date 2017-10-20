import asyncio



class ClientError(Exception):
    pass


class ClientServerProtocol(asyncio.Protocol):

    store = {}

    @classmethod
    def put(cls, key, value):
        if key in cls.store:
            cls.store[key].append(value)
        else:
            cls.store[key] = [value]

    def connection_made(self, transport):
        self.transport = transport

    @classmethod
    def process_data(self, data):
        if data[:4] == 'get ':
            key = data[4:-1]
            if self.store.get(key) != None:
                message ='ok\n'
                for i,j in self.store.get(key):
                    message += f'{key} {i} {j}\n'
                message+='\n'
                return message.encode()

            elif '*' in key:
                message = 'ok\n'
                for key in self.store.keys():
                    for i, j in self.store.get(key):
                        message += f'{key} {i} {j}\n'
                message += '\n'
                return message.encode()
            else:
                return b'ok\n\n'

        elif data[:4] == 'put ':
            key, value, timestamp = data[:-1].split(' ')[1:]
            self.put(key, (value, timestamp))
            print(key, value, timestamp)
            return b'ok\n\n'
        else:
            return b'error\nwrongcommand\n\n'


    def data_received(self, data):
        resp = self.process_data(data.decode())
        print(resp)
        self.transport.write(resp)


def run_server(host, port):
    print('start')
    
    loop = asyncio.get_event_loop()
    coro = loop.create_server(
        ClientServerProtocol,
        host, port
    )

    server = loop.run_until_complete(coro)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()

if __name__ == '__main__':
    run_server('127.0.0.1', 9999)
