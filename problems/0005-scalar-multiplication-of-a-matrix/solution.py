def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	res = []
	for row in matrix:
		r = []
		for i in range(len(row)):
			r.append(scalar*row[i])
		res.append(r)
	return res