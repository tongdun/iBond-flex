#!/bin/bash

#svn co -r81 https://github.com/floodyberry/ed25519-donna
gcc -O2 -std=c99 -fPIC -g -I /usr/include/python3.6m -I ed25519-donna/ -c ed25519_donna.c -o ed25519_donna.o
ld -shared -lpython3 ed25519_donna.o -o ed25519_donna.so
