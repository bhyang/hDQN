import random

class game():
	def __init__(self):
		self.s = 2
		self.vis = False
		

	def reset(self):
		self.s = 2
		self.vis = False
		return [self.s, 0, False]

	def moveAndReturn(self, dir):
		if self.s == 1:
			return [1, 0, True]

		if dir == 0 or self.s == 6:
			self.s = self.s - 1
		elif dir == 1:
			self.s = self.s + random.choice([-1, 1])

		if self.s == 1:
			if self.vis:
				return [1, 1, True]
			else:
				return [1, 0.01, True]
		
		elif self.s == 6:
			self.vis = True

		return [self.s, 0, False]