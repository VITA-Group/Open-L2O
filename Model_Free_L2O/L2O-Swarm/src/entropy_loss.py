import tensorflow as tf
import numpy as np



def distance_matrix(x1, n1, x2, n2, dim): 
	# calculating the distance matrix of x1, x2; x1 shape:(n1, dim)  x2 shape(n2, dim), return shape (n1, n2)
	y1 = tf.tile(x1, [1, n2, 1])  # y1 shape: (n1*n2, dim)
	y2 = tf.tile(x2, [1, 1, n1]) # y2 shape: (n2, n1*dim)


	y2 = tf.reshape(y2, (-1, n1*n2, dim))

	#dis = tf.square(tf.norm(tf.math.subtract(y1,y2), axis=-1, ord='euclidean'))
	dis = tf.reduce_sum(tf.square(tf.math.subtract(y1,y2)), axis=-1)

	dis = tf.reshape(dis, (-1, n1,n2))

	
	return dis



def self_loss (x, fx_array, n):
	# lambda values for controling the  balance between exploitation and exploration.
	lam=0.001
	
	print (fx_array.shape)

	x, x1 = tf.split(x, [n, 0], 0)
	fx_array,f1 = tf.split(fx_array, [n,0], 0)
	problem_dim = x.shape.as_list()[-1]
	batch_size = x.shape.as_list()[1]
	x = tf.transpose(x, [1,0,2])
	fx_array = tf.transpose(fx_array, [1,0])

	print (fx_array.shape, x.shape)
	
	
	def entropy(x, fx_array,  preh):



		dis = (distance_matrix(x, n, x, n, problem_dim))

		#dis =  2* (tf.square(x)-tf.matmul(x, x, transpose_b=True))

		
		l=1
		dis1 =tf.math.divide(dis, 2*l)
		KK = tf.math.exp(tf.math.negative(dis1))


		#KK = tf.reshape(KK1, (n, n))
		epsilon =2.1


		KKK = tf.linalg.inv(tf.math.add(KK, tf.eye(n)*epsilon))  #(n,n)


		tt = []
		for i in range(1):
			numofsamples = 1000
			samples = tf.random.normal((batch_size, numofsamples, problem_dim), mean=0.0, stddev=0.01)
			#samples = tf.random.normal((batch_size, numofsamples, problem_dim), mean=0.0, stddev=1)
			dis2 = distance_matrix(samples, numofsamples, x, n, problem_dim)

		

			l=1
			dis3 =tf.math.divide(dis2, 2*l)
			Kx = (tf.math.exp(tf.math.negative(dis3)))  #  (n,100000)

		
			final1 = tf.matmul(tf.matmul(Kx, KKK, transpose_a=False), tf.expand_dims(fx_array,axis=-1))
			final1 = tf.reshape(final1, (batch_size, numofsamples, ))
			#tt.append(final1)

		#final1 = tf.reshape(tf.transpose(tf.convert_to_tensor(tt), [1, 0 ,2]), (batch_size, -1))
		print (final1.shape)
		
		

		h=preh[0]
		rho_0=1
		import numpy as np

		if(preh[1]==0):
			rho = rho_0*np.exp(1./h*(n**(1/2.)))
		else:
			rho = rho_0*tf.math.exp(1./h*(n**(1/2.)))
		
		
		# for numerical concern. minus the largest value to make all values non-positve before doing exponential
		rhofx = -rho*final1
		rhofx = rhofx - tf.expand_dims(tf.reduce_max(rhofx, -1), axis=-1)


		c = tf.math.exp(rhofx)

		# for numerical concern
		px = tf.divide(c+0.0001, tf.expand_dims(tf.reduce_sum(c+0.0001, axis=-1), axis=-1))
		


		ent = tf.reduce_mean(tf.reduce_sum(tf.math.negative(tf.multiply(px, tf.math.log(px))), axis=-1))

		print (c.shape, px.shape, final1.shape, ent.shape)
	

		return ent


	sumfx = tf.reduce_mean(tf.reduce_sum(fx_array, -1))
	preh = np.log(5.**problem_dim)
	h0 = entropy(x, fx_array,  [preh,0])
	h  = entropy(x, fx_array,  [h0,1])


	return sumfx+lam*h

if __name__ == "__main__":

	x = tf.get_variable("x", [300,128, 2], dtype=tf.float32, initializer=tf.random_normal_initializer)
	fx_array = tf.get_variable("y", [300, 128], dtype=tf.float32, initializer=tf.random_normal_initializer)

	loss, t1, t2 = self_loss(x, fx_array, 300)

	print (loss.shape)

	with tf.Session() as sess:

		for i in range(20):
			sess.run(tf.global_variables_initializer())
			#tt1 = sess.run(t1)
			#tt2 = sess.run(t2)
			#print (np.max(tt1), np.min(tt2))
			print (sess.run(t2))
