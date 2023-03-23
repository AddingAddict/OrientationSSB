import tensorflow as tf


nl_linear = lambda x: x
nl_rect = lambda x,thr=0: tf.where(tf.greater(x, thr), x, tf.zeros(tf.shape(x),dtype=tf.float32) )
exponent = 2.
nl_powerlaw = lambda x: tf.where(tf.greater(x, 0), x**exponent, tf.zeros(tf.shape(x),\
								 dtype=tf.float32) )