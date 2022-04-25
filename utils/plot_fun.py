# -*- coding: utf-8 -*-
#Function of drawing error curve

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 


def plot_Loss(data,save_add,title="Loss"):

	plt.figure()
	x = np.arange(len(data))
	plt.plot(data)
	min_index = np.argmin(data)
	min_data = np.min(data)
	plt.plot(min_index,min_data,"ks")
	show_min = "[{0},{1}]".format(min_index,min_data)
	plt.title(title)
	plt.annotate(show_min,xytext=(min_index,min_data),xy=(min_index,min_data))
	plt.savefig(save_add,dpi=300)
	plt.close()
	
def plot_image(data,error,filename,data_location,cla="T2M"):

	plt.figure()
	data = np.squeeze(data)
	X  = np.arange(data.shape[0])
	Y  = np.arange(data.shape[1])
	x,y = np.meshgrid(Y,X)
	#print(x.shape)
	#print(y.shape)
	#print(data.shape)

	if cla == "T2M":
		clevs = np.arange(-80,60,2)
		plt.contourf(x,y,data,clevs,cmap="jet")
		plt.colorbar()
		plt.title("data_location:{0}".format(data_location))

	elif cla=="MSE":
		plt.contourf(x,y,data,cmap="jet")
		plt.colorbar()
		plt.title("RMSE: {0:.6f}  data_location:{1}".format(error,data_location))

	elif cla=="MAE":
		#clevs = np.array([-5,-4,-3.5,-3.2,-3.0,-2.8,-2.6,-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,5.0])
		clevs = np.arange(-12,12,1)
		plt.contourf(x,y,data,clevs,cmap="jet")
		plt.colorbar()
		plt.title("MAE:{0:.6f}    data_location:{1}".format(error,data_location))

	plt.savefig(filename,dpi=300)	
	plt.close()
    


