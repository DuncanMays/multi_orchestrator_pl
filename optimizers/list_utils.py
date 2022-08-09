
def get_multilist(shape):
	if (len(shape) > 0):
		i = shape[0]
		l = [get_multilist(shape[1:]) for _ in range(i)]
		return l

	else:
		return None

# this function manually instantiates an empty 2D list, this can't be done with [None]*5 because of aliasing issues
def get_2D_list(W, H):
	l = []
	for w in range(W):
		r = []
		for h in range(H):
			r.append(None)
		l.append(r)
	return l