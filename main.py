import numpy
from scipy.sparse import csr_matrix
import random
import pickle
import recommender
def writeRatingsToFile(fraction):
	users={}
	movies={}
	ratings=[]
	with open("ml-10M100K/ratings.dat","r") as f:
		for line in f:
			uid=int(line.split('::')[0])
			mid=int(line.split('::')[1])
			rating=float(line.split('::')[2])
			if uid not in users:
				users[uid]=len(users)
			if mid not in movies:
				movies[mid]=len(movies)
			ratings.append((uid,mid,rating))

	users1={}
	movies1={}
	r=[]
	for uid in users:
		if random.randrange(fraction)==0:
			users1[uid]=len(users1)
	for mid in movies:
		if random.randrange(fraction)==0:
			movies1[mid]=len(movies1)
	users=users1
	movies=movies1
	for rating in ratings:
		if rating[0] in users and rating[1] in movies:
			r.append((users[rating[0]],movies[rating[1]],rating[2]))

	ratings=r

	with open("users.pickle","wb") as f:
		pickle.dump(users,f)
	with open("movies.pickle","wb") as f:
		pickle.dump(movies,f)
	with open("ratings.pickle","wb") as f:
		pickle.dump(ratings,f)
	'''
	print(len(users))
	print(len(movies))
	print(len(ratings))
	matrix=numpy.ndarray((len(users),len(movies)))
	for elt in ratings:
		matrix[elt[0]][elt[1]]=elt[2]
	
	return matrix
	'''

def readRatingsFromFile():
	with open("users.pickle","rb") as f:
		users=pickle.load(f)
	with open("movies.pickle","rb") as f:
		movies=pickle.load(f)
	with open("ratings.pickle","rb") as f:
		ratings=pickle.load(f)
	
	matrix=numpy.ndarray((len(users),len(movies)))
	for elt in ratings:
		matrix[elt[0]][elt[1]]=elt[2]
	return matrix

def main():
	i=int(input())
	if i==0:
		n=int(input())
		writeRatingsToFile(n)
	
	elif i==1:
		#'''
		numpy.seterr(all='raise')
		matrix=readRatingsFromFile()
		#matrix=csr_matrix(matrix)
		
		#matrix=numpy.array([[1,2,3,4,0],[2,3,4,0,1],[3,4,0,1,2],[4,0,1,2,3]]).transpose()
		
		#matrix=numpy.array([[3,1,1],[-1,3,1]])
		#matrix=numpy.array([[3,2,2],[2,3,-2]])
		#matrix=numpy.array([[2,2,0],[-1,-1,0]])
		#matrix=numpy.array([[2,4],[1,3],[6,5],[7,9]])#.transpose()
		#print(matrix.shape)
		'''
		cf=recommender.CollaborativeFiltering(matrix)
		k=cf.predict()

		cf=recommender.CollaborativeFilteringBaseline(matrix)
		k=cf.predict()
		'''
		cf=recommender.CURRecommenderSystem(matrix,96)
		cf.predict()

if __name__ == '__main__':
	main()