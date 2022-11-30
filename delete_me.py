l_1 = [22.2, 21.03, 20.6]
l_2 = [21.3, 21.3, 19.2, 19.7, 18.4, 16.3]
l_3 = [17.7, 22.3, 18.2, 22.7, 20.47, 21.97, 17.6, 20.8, 18.0]

def get_avg(l):
	return sum(l)/len(l)

print(get_avg(l_1))
print(get_avg(l_2))
print(get_avg(l_3))