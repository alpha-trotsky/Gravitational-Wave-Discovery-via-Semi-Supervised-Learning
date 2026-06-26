#----------------------
# Import needed modules
#----------------------
import numpy as np
import matplotlib.pylab as plt
import h5py
#--------------
# Open the File
#--------------
filename = '/mnt/c/Users/Belia/OneDrive/Documents/GitHub/Gravitational-Wave-Discovery-via-Semi-Supervised-Learning/Model/train-1129037824-4096.hdf5'
dataFile = h5py.File(filename, 'r')
#-----------------
# Explore the file
#-----------------
for key in dataFile.keys():
    print(key)


#--------------------
# Read in strain data
#--------------------
strain = dataFile['strain']['Strain']
ts = dataFile['strain']['Strain'].attrs['Xspacing']
print(f"ts = {ts}s, sample rate = {1/ts}Hz")


#-------------------------
# Print out some meta data
#-------------------------
metaKeys = dataFile['meta'].keys()
meta = dataFile['meta']
for key in metaKeys:
    print(key, meta[key])




#---------------------
# Create a time vector
#---------------------
gpsStart = meta['GPSstart'][()]
duration = meta['Duration'][()]
gpsEnd   = gpsStart + duration

time = np.arange(gpsStart, gpsEnd, ts)

time_cut = time[:(11 * 4096)]
strain_cut = strain[:(11 * 4096)]

#---------------------
# Plot the time series
#---------------------
plt.plot(time, strain[()])
plt.xlabel('GPS Time (s)')
plt.ylabel('H1 Strain')
plt.show()

def find_ten_biggest_strain_values(strain, time):
    # Get the indices of the 10 largest absolute strain values
    largest_indices = np.argsort(np.abs(strain))[-10:]
    
    # Get the corresponding strain values and their times
    largest_strains = strain[largest_indices]
    corresponding_times = time[largest_indices]
    
    return corresponding_times, largest_strains 

time_val, strain_val = find_ten_biggest_strain_values(strain_cut, time_cut)
print(f"top 10 strain values: {strain_val}")

plt.plot(time_cut, strain[:(11 * 4096)])
plt.xlabel('GPS Time (s)')
plt.ylabel('H1 Strain')
plt.show()