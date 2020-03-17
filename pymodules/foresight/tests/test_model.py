import unittest
import foresight.model
from tensorflow.keras import Sequential as tf_model
from numpy import ndarray as np_arr
from pandas import Timedelta as TD


class TestModel(unittest.TestCase):
    """tests for Model class"""
    """    def __init__(self, model, data, data_freq, seq_len, data_transform = None)"""
    def test_Model_Raise_if_Arg_Model_Is_None(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          model=None,
                          data=np_arr([0]),
                          data_freq=TD('1D'),
                          seq_len=30)

    def test_Model_Raise_if_Arg_Model_Is_Int(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          5,
                          data=np_arr([0]),
                          data_freq=TD('1D'),
                          seq_len=30)

    def test_Model_Raise_if_Arg_Model_Is_Str(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          'foresight.model.Model',
                          data=np_arr([0]),
                          data_freq=TD('1D'),
                          seq_len=30)

    def test_Model_Raise_if_Arg_Data_Is_None(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          tf_model(),
                          data=None,
                          data_freq=TD('1D'),
                          seq_len=30)

    def test_Model_Raise_if_Arg_Data_Is_Int(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          tf_model(),
                          data=1,
                          data_freq=TD('1D'),
                          seq_len=30)

    def test_Model_Raise_if_Arg_Data_Is_Str(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          tf_model(),
                          data='None',
                          data_freq=TD('1D'),
                          seq_len=30)

    def test_Model_Raise_if_Arg_DataFreq_Is_None(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          tf_model(),
                          data=np_arr([0]),
                          data_freq=None,
                          seq_len=30)

    def test_Model_Raise_if_Arg_DataFreq_Is_Int(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          tf_model(),
                          data=np_arr([0]),
                          data_freq=1,
                          seq_len=30)

    def test_Model_Raise_if_Arg_DataFreq_Is_Str(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          tf_model(),
                          data=np_arr([0]),
                          data_freq='1D',
                          seq_len=30)

    def test_Model_Raise_if_Arg_seq_len_Is_None(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          tf_model(),
                          data=np_arr([0]),
                          data_freq=TD('1D'),
                          seq_len=None)

    # def test_Model_Raise_if_Arg_seq_len_Is_Model(self):
    #     self.assertRaises(TypeError,
    #                       foresight.model.Model,
    #                       tf_model(),
    #                       data=np_arr([0]),
    #                       data_freq=TD('1D'),
    #                       seq_len=foresight.model.Model(tf_model(),
    #                                                     data=np_arr([0]),
    #                                                     data_freq=TD('1D'),
    #                                                     seq_len=30))

    def test_Model_Raise_if_Arg_seq_len_Is_Str(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          tf_model(),
                          data=np_arr([0]),
                          data_freq=TD('1D'),
                          seq_len='30')

    def test_Model_Raise_if_Arg_transform_not_function_object(self):
        self.assertRaises(TypeError,
                          foresight.model.Model,
                          tf_model(),
                          data=np_arr([0]),
                          data_freq=TD('1D'),
                          seq_len=30,
                          data_transform=30)


if __name__ == '__main__':
    unittest.main()
"""
    

try:
    print('transform not function object')
    b1 = fx.model.Model(Sequential(), np_arr([0]), TD('1D'), 25)
except TypeError as x:
    print(x)
try:
    print('transform is function object')
    f = lambda x: "moo-haha"
    b1 = fx.model.Model(Sequential(), np_arr([0]), TD('1D'), f)
except TypeError as x:
    print(x)
else:
    print('This also worked!')
"""
