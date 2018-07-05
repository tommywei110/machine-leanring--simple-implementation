#!/usr/bin/env python
import numpy as np
from server import Server

if __name__ == '__main__':
	s = Server(5, 0.9, 0.5)
	for _ in range(3):
		s.server_epoch()
	print(s.history_acc)