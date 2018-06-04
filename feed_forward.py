import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10 #{0,1,2,...,8,9}
batch_size = 100
#heightXwidth
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')
keep_prob = tf.placeholder(tf.float32)

def  neural_network_model(data):
	hidden_1_layer= {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),'baises':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer= {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'baises':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer= {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'baises':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer= {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'baises':tf.Variable(tf.random_normal([n_classes]))}

	#(input*weights) + baises
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['baises'])
	l1 = tf.nn.relu(l1)
	l1 = tf.nn.dropout(l1,keep_prob)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['baises'])
	l2 = tf.nn.relu(l2)
	l2 = tf.nn.dropout(l2,keep_prob)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['baises'])
	l3 = tf.nn.relu(l3)
	l3 = tf.nn.dropout(l3,keep_prob)

	output = (tf.matmul(l3,output_layer['weights'])+output_layer['baises'])
	return output

def train_neural_network(x):
	prediction =neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
	starter_learning_rate = 0.001
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.85, staircase=True)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	hm_epoch = 20

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epoch):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y,keep_prob:0.9})
				epoch_loss += c
			print('Epoch: ',epoch,'completed out of: ',hm_epoch,'loss: ',epoch_loss) 

		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Test Accuracy:',accuracy.eval({x:mnist.test.images ,y:mnist.test.labels,keep_prob:1}))
		print('Train Accuracy:',accuracy.eval({x:mnist.train.images,y:mnist.train.labels,keep_prob:1}))

train_neural_network(x)

	










