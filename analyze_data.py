import numpy as np
import json
import matplotlib.pyplot as plt

def get_data(filename,attr):
    f = open(filename)
    data = json.load(f)

    for d in data: # iterate through each field, get the one we want
        x = d['x']
        y = d['y']
        if d['name'] == attr:
            return np.array(x),np.array(y)

dir = 'data/'
filenames = ['thirsty_sheep','joyous_convertible']

# plt.figure()
# for f in filenames:
#     x,y = get_data(dir+f,'test_loss')
#     plt.plot(x,y)

# plt.show()




plt.figure()
for f in filenames:
    x,a_loss = get_data(dir+f,'test_a_loss')
    _,sT_loss =  get_data(dir+f,'test_s_T_loss')
    _,kl_loss = get_data(dir+f,'test_kl_loss')
    _,loss = get_data(dir+f,'test_loss')
    print(np.argmin(loss))
    

    plt.plot(x,a_loss)
plt.show()

