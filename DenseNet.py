import tensorflow as tf
import tflearn
from implementation import *
class_num=100
dropout_rate=0.2



def conv_layer(x,filter_num,kernel_size,stride=1,scope):
	with tf.name_scope(scope):
		x=tf.layers.conv2d(inputs=x,use_bias=False,filters=filter_num,kernel_size=kernel_size,strides=stride,padding='SAME')
	return x
def batch_normalization(x,training,scope):
	with arg_scope([tf.contrib.layers.batch_norm],scope=scope,updates_collection=None,decay=0.9,center=True,scale=True,zero_debias_moving_mean=True):
		return tf.contrib.layers.batch_norm(inputs=x,is_training=training,reuse=None) if training else tf.contrib.layers.batch_norm(input=x,is_training=training,reuse=True)
def relu(x):
	return tf.nn.relu(x)
def drop_out(x,rate,training):
	return tf.layers.drop_out(inputs=x,rate=rate,training=training)
def average_pooling(x,pool_size=[2,2],stride=2,padding='VALID'):
	return tf.layers.average_pooling2d(input=x,pool_size=pool_size,strides=stride,padding=padding)
def max_pooling(x,pool_size=[3,3],stride=2,padding='VALID'):
	return tf.layers.max_pooling2d(x,pool_size=pool_size,strides=stride,padding=padding)
def concation(x):
	return tf.concat(x,axis=3)
def linear(x):
	return tf.layers.dense(inputs=x,units=class_num,name='linear')
def global_average_pooling(x,stride=1):
	return tflearn.layers.conv.global_avg_pool(x,name='global_average_pooling')
class DenseNet:
	def __init__(self,data,k,training):
		self.data=data
		self.k=k
		self.training=training
		self.model=self.dense_net(data)
	def bottleneck(self,x,scope):
		with tf.name_scope(scope):
			x=batch_normalization(x,training=self.training,scope=scope+'_bn0')
			x=relu(x)
			x=conv_layer(x,filter_num=4*self.k,kernel_size=[1,1],stride=1,scope=scope+'_conv0')
			x=drop_out(x,rate=dropout_rate,training=self.training)
			x=batch_normalization(x,training=self.training,scope=scope+'_bn1')
			x=relu(x)
			x=conv_layer(x,filter_num=self.k,kernel_size=[3,3],stride=1,scope=scope+'_conv1')
			x=drop_out(x,rate=dropout_rate,training=self.training)
			return x
	def dense_block(self,x,num_in_block,scope):
		with tf.name_scope(scope):
			concat=list()
			concat.append(x) 
			for i in range(num_in_block):
				x=self.bottleneck(x,scope=scope+'_bottleneck'+str(i))
				concat.append(x)
				x=concation(concat) 
			return x
	def trainsition_block(self,x,scope):
		with tf.name_scope(scope):
			x=batch_norm(x,training=self.training,scope=scope+'_bn0')
			x=relu(x)
			cnl_num=x.shape[-1]
			x=conv_layer(x,filter_num=cnl_num*0.5,kernel=[1,1],scope=scope+'_conv0')
			x=drop_out(x,rate=dropout_rate,training=self.training)
			x=average_pooling(x,pool_size=[2,2],stride=2)
			return x

	def dense_net(self,x):
		x=conv_layer(x,filter_num=2*self.k,kernel_size=[7,7],stride=2)
		

		x=self.dense_block(x,num_in_block=16,scope='dense_0')
		x=self.trainsition_block(x,scope='trans_0')
		x=self.dense_block(x,num_in_block=16,scope='dense_1')
		x=self.trainsition_block(x,scope='trans_1')
		x=self.dense_block(x,num_in_block=16,scope='dense_2')
		x=batch_normalization(x,training=self.training,scope='bn')
		x=relu(x)
		x=global_average_pooling(x)
		x=tf.contrib.layers.flatten(x)
		x=linear(x)
		# x=self.
		# x=self.trainsition_block(x,scope='trains_2')
		# x=self.dense_block(x,num_in_block=32,scope='dense_3')
		return x























if __name__ == '__main__':
	train()
