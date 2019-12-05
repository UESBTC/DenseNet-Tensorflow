import tensorflow as tf
import tflearn
from implement import *
class_num=100
dropout_rate=0.2
image_size=32
channel_num=3
growth_rate=12
init_learning_rate=1e-4
epsilon=1e-4
total_epoch=300
iteration=782
test_iteration=157
batch_size=64

def conv_layer(x,filter_num,kernel_size,stride=1,scope=None):
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
	# return None
def evaluate(sess,test_data,test_labels):
	test_feed_dict={
		x:data,
		y_:test_labels,
		training_flag:False
	}
	_, test_loss,test_acc=sess.run(accuracy,feed_dict=test_feed_dict)
	return test_loss,test_acc


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
def train():
	train_data,train_labels,test_data,test_labels=prepare_data()
	train_data,test_data=color_preprocess(train_data,test_data)
	x=tf.placeholder(tf.float32,shape=[None,image_size,image_size,channel_num])
	y_=tf.placeholder(tf.float32,shape=[None,class_num])
	training_flag=tf.placeholder(tf.bool)
	learning_rate=tf.placeholder(tf.float32,name='learning_rate')

	logits=DenseNet(data=train_data,k=growth_rate,training=training_flag).model
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))

	train=tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=epsilon).minimize(cost)
	correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	saver=tf.train.Saver()

	with tf.Session() as sess:
		ckpt=tf.train.get_checkpoint_state('./model')
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			saver.restore(sess,ckpt.model_checkpoint_path)
		else:
			tf.global_veriables_initializer().run()
		cur_learning_rate=init_learning_rate
		for epoch in range(1,total_epoch+1):
			if epoch==(0.5*total_epoch) or epoch==(0.75*total_epoch):
				epoch/=10
			pre_ind=0
			train_loss=0.0
			train_acc=0.0
			for step in (1,iteration+1):
				if pre_ind+batch_size<50000:
					batch_x=train_data[pre_ind:pre_ind+batch_size]
					batch_y=train_labels[pre_ind:pre_ind+batch_size]
				else:
					batch_x=train_data[pre_ind:]
					batch_y=train_labels[pre_ind:]
				batch_x=data_augmentation(batch_x)

				train_feed_dict={
					x:batch_x,
					y_:batch_y,
					training_flag:True,
					learning_rate:cur_learning_rate
				}
				_, batch_loss=sess.run([train,cost],feed_dict=train_feed_dict)
				batch_acc=accuracy.eval(feed_dict=train_feed_dict)
				train_loss+=batch_loss
				train_acc+=batch_acc
				pre_ind+=batch_size

				if step==iteration:
					train_loss/=iteration
					train_acc/=iteration
					test_loss,test_acc=evaluate(sess,test_data,test_labels)
					print('epoch:%d/%d, train_loss=%.4f, train_acc=%.4f, test_loss:%.4f, test_acc:%.4f\n'%(epoch,train_loss,train_acc,test_loss,test_acc))
		saver.save(sess=sess,save_path='./model/dense.ckpt')




























if __name__ == '__main__':
	train()
