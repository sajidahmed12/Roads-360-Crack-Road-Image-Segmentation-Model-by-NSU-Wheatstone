import tensorflow as tf
import helper


num_classes = 2   # Road and Crack 2 Classes of Segmentation 
image_shape = (160, 576)

# Specify these directory paths

data_dir = './data'
imsave_dir = './output'
training_dir ='./data/data_road/training/'
model_path = './model/my_test_model'
vgg_path = './data/vgg/'


# PLACEHOLDER TENSORS


correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)


# Functions and Loading the VGG Pretrained Model
def load_vgg(sess, model_path):
  
  # load the model and weights
  model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

  # Get Tensors to be returned from graph of Pretrained Model
  graph = tf.get_default_graph()
  image_input = graph.get_tensor_by_name('image_input:0')
  keep_prob = graph.get_tensor_by_name('keep_prob:0')
  layer3 = graph.get_tensor_by_name('layer3_out:0')
  layer4 = graph.get_tensor_by_name('layer4_out:0')
  layer7 = graph.get_tensor_by_name('layer7_out:0')

  return image_input, keep_prob, layer3, layer4, layer7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
   
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer from the VGG model into our Model 
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) because to match size of layer 4 so that we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
    kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
  
  # Reshape 4D tensors to 2D, each row represents a pixel, each column a class[road,Crack]
  logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
  correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

  # Calculate distance from actual labels using cross entropy
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
  # Take mean for total loss
  loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

  # The model implements this operation to find the weights/parameters that would yield correct pixel labels
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

  return logits, train_op, loss_op


def runTest(speed_p=True):


  # A function to get batches
  get_batches_fn = helper.gen_batch_function(training_dir, image_shape)
  
  with tf.Session() as session:
        
    # Returns the three layers, keep probability and input layer from the vgg architecture
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, model_path)

    # The resulting network architecture from adding a decoder on top of the given vgg model
    model_output = layers(layer3, layer4, layer7, num_classes)

    # Returns the output logits, training operation and cost operation to be used
    # - logits: each row represents a pixel, each column a class[road,Crack]
    # - train_op: function used to get the right parameters to the model to correctly label the pixels
    # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy

    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)
    
    # Initialize all variables
    session.runTest(tf.global_variables_initializer())
    session.runTest(tf.local_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, model_path)

    print("Restored the saved Model in file: %s" % model_path)
    
    # Run the model with the test dataset  and save each labeled output images as (roads = Green,Cracks = Red) #Identified

    helper.save_inference_samples(imsave_dir, data_dir, session, image_shape, logits, keep_prob, image_input,speed_p)
    
    print("All done! yepe ")

# MAIN function call 
if __name__ == '__main__':
    runTest(speed_p=True)