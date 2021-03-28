import tensorflow as tf
from Model_mse_dctFullHpCov_noCali_noBlurTrain_Hinge import Model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_integer("image_size", 128, "the size of image input")
flags.DEFINE_integer("c_dim", 1, "the size of channel")
flags.DEFINE_integer("epoch", 1000, "number of epoch")
flags.DEFINE_integer("batch_size", 16, "the size of batch")
flags.DEFINE_float("g_lr", 1e-4, "g the learning rate")
flags.DEFINE_float("d_lr", 5e-4, "d the learning rate")
flags.DEFINE_string("checkpoint_dir", "data_NoNoised", "name of the checkpoint directory")
flags.DEFINE_string("checkpoint_mse_dctFullHpCov_noCali_noBlurTrain_Hinge", "checkpoint_mse_dctFullHpCov_noCali_noBlurTrain_Hinge", "save checkpoint data")

flags.DEFINE_string("result_dir", "result", "name of the result directory")
flags.DEFINE_string("train_input_set", "jpeg_pic", "name of the train set")
flags.DEFINE_string("train_label_set", "groundtruth_pic", "name of the train set")
flags.DEFINE_string("test_input_set", "jpeg_pic", "name of the train set")
flags.DEFINE_string("test_label_set", "groundtruth_pic", "name of the train set")
flags.DEFINE_integer("D", 8, "D")
flags.DEFINE_integer("C", 6, "C")
flags.DEFINE_integer("G", 32, "G")
flags.DEFINE_integer("G0", 64, "G0")
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")



def main(_):
    model = Model(tf.Session(),
                  is_train = FLAGS.is_train,
                  image_size = FLAGS.image_size,
                  c_dim = FLAGS.c_dim,
                  batch_size = FLAGS.batch_size,
                  D = FLAGS.D,
                  C = FLAGS.C,
                  G = FLAGS.G,
                  G0 = FLAGS.G0,
                  kernel_size = FLAGS.kernel_size
                 )

    if model.is_train:
        model.train(FLAGS)
    else:
        model.test(FLAGS)

if __name__=='__main__':
    tf.app.run()
