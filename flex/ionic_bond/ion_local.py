"""
Flex bond io
"""
#
#  Copyright 2020 The FLEX Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import platform
import socket
import io
import os
import logging
import pickle
from typing import List, Any, Union
import random

from flex.ionic_bond.configuration import Configuration


class Pool():
    """
    simple socket connection pool
    """
    def __init__(self, max_connections=300):
        self._pool = {}
        self.max_connections = max_connections

    @staticmethod
    def set_keepalive_linux(sock,
                            after_idle_sec=10,
                            interval_sec=10,
                            max_fails=5):
        """Set TCP keepalive on an open socket.
        It activates after 1 second (after_idle_sec) of idleness,
        then sends a keepalive ping once every 3 seconds (interval_sec),
        and closes the connection after
        5 failed ping (max_fails), or 15 seconds
        """
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, after_idle_sec)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_sec)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, max_fails)

    @staticmethod
    def set_keepalive_osx(sock,
                          after_idle_sec=10,
                          interval_sec=10,
                          max_fails=5):
        """Set TCP keepalive on an open socket.
           sends a keepalive ping once every 10 seconds (interval_sec)
        """
        # scraped from /usr/include, not exported by python's socket module
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, 0x10, interval_sec)

    @staticmethod
    def set_keepalive_win(sock):
        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 10000, 3000))

    @staticmethod
    def set_keepalive(sock):
        local_system = platform.system()
        if local_system == 'Linux':
            Pool.set_keepalive_linux(sock)
        elif local_system == 'Darwin':
            Pool.set_keepalive_osx(sock)
        elif local_system == 'Windows':
            Pool.set_keepalive_win(sock)

    def get_connection(self, server):
        """
        return socket connection to server, reuing existing ones
        """
        if server in self._pool:
            while len(self._pool[server]) >= self.max_connections:
                pos = random.randint(0, len(self._pool[server]) - 1)
                sock = self._pool[server][pos]
                if sock.fileno() != -1:
                    return sock
                del self._pool[pos]
        return self.get_new_connection(server)

    def get_new_connection(self, server):
        """
        return new socket connection to server
        """
        sock = socket.socket()
        sock.connect(server)
        Pool.set_keepalive_linux(sock)
        if server in self._pool:
            self._pool[server].append(sock)
        else:
            self._pool[server] = [sock]
        return sock

    def close(self):
        """
        close all connections
        """
        for _, sock in self._pool.items():
            buf = io.BytesIO()
            buf.write('Bye;CLOSE'.encode())
            sock.sendfile(buf, 0)
            sock.close()


class Ion(Configuration):
    """
    flex bond main class
    """
    pool = Pool()

    @staticmethod
    def send(key: str, value: Any, dst: Union[str, List[str]]) -> bool:
        """
        Send data to remote using key.

        Arg:
            key: str, a key contains all infomation about the data.
            value: Any, local variable.
            dst: str or a list of str, each is a id of fedration.

        Return:
            bool: True for success, False for failure
        """
        dst = ['localhost']
        rets = [Ion._send(key, value, _dst) for _dst in dst]
        return all(rets)

    @staticmethod
    def _send(key: str, value: Any, dst: str) -> bool:
        """
        Send data to one remote using key.
        """
        server = dst, Ion.socket_port
        fname = f'{key}.pt'
        buf = io.BytesIO()
        buf.write(fname.encode())
        loc = buf.tell()
        buf.seek(Ion.socket_head)
        pickle.dump(value, buf)
        buf_size = buf.tell()
        pickle_size = buf_size - Ion.socket_head
        buf.seek(loc)
        buf.write(f'#{pickle_size};'
                  'REQ_STORE_THEN_WAIT'.encode())
        logging.info('send msg %s to %s: size %i',
                     key,
                     dst,
                     pickle_size)
        buf.seek(0)

        sock = Ion.pool.get_connection(server)
        ret = sock.sendfile(buf, 0, count=buf_size)
        return ret == buf_size

    @staticmethod
    def recv(key: str, src: Union[str, List[str]] = None) -> Any:
        """
        Recv data from remote using key.

        Arg:
            key: str, a key contains all infomation about the data.
            src: Union[str, List[str]], Source is not used actually
                But it is kept to make a consistent API.
        Return:
            Any: pickle loaded object.
        """
        fname = f'{Configuration.socket_prefix}/{key}.pt'
        fname_done = f'{fname}.done'
        logging.info('receving %s from %s', key, src)
        while True:
            if os.path.exists(fname_done):
                break
        logging.info('ready for %s, trying to load', key)
        with open(fname, 'rb') as buf:
            ret = pickle.load(buf)
        os.system(f'rm "{fname}" "{fname_done}"')
        return ret
