#Author : Ravisutha Sakrepatna Srinivasamurthy
#Purpose: EDF to EEG database framework

import random as rand
import csv
import inspect
import matplotlib.pyplot as plt
import pyedflib
import sys
import scipy.signal
import numpy as np
from os import listdir
from os.path import isfile, join

class edf_convert:
    """ Framework for reading edf files and converting the sampled data into a matrix notation which then can be used for training the neural network."""

    def __init__ (self, samp_freq=256):
        """ Run all of the following functions to construct matrix which will be used in training the network."""

        self.samp_freq = 256

        #Get the path for edf files, montage.csv and fix-xx.csv
        self.get_path ()

        #Get names of each edf files
        self.get_files_name ()

        #Get montage list stored in montage.csv
        self.montage_list ()

        #Get signals
        self.get_signals ()

    def get_path (self):
        """ Get the path of folder containing EDF files."""

        if (len (sys.argv) < 4):
            if (len (sys.argv ) < 2):
                self.path_edf = input ('Please provide path for edf files:')
            if (len (sys.argv) < 3):
                self.path_montage = input ('Please provide path for montage.csv:')
            if (len (sys.argv) < 4):
                self.path_fix = input ('Please provide path for fix-xx.csv:')
        else:
            self.path_edf = sys.argv[1]
            self.path_montage = sys.argv[2]
            self.path_fix = sys.argv[3]

    def get_files_name (self):
        """ Get the list of edf files."""

        self.edf_files = []
        for f in listdir (self.path_edf):
            if (isfile(join (self.path_edf, f))):
                self.edf_files.append (join(self.path_edf, f))

    def montage_list (self):
        """ Get montage list and store it in a dictionary. """

        self.montage = {}
        self.mont_list = set ()
        with open(self.path_montage, newline='') as csvfile:
            mont = csv.reader(csvfile, delimiter=',')
            for row in mont:
                self.mont_list.add (row[2])
                self.mont_list.add (row[3])
                self.montage[row[0] + row[1]] = [row[2], row[3]]

        #Replace ECG by ECGL and ECGR
        self.mont_list.remove ('avg')
        self.mont_list.remove ('ECG')
        self.mont_list.add ('ECGL')
        self.mont_list.add ('ECGR')

    def get_signals (self):
        """ Get signals for the given event and convert it to montage signal """

        self.montage_info = {}
        self.files_mont = []
        self.time_info = []
        self.file = [] 
        file_no = 0
        j = 0

        with open(self.path_fix, newline='') as csvfile:
            fix = csv.reader(csvfile, delimiter=',')
            for row in fix:

                # Find the file number
                file_no = int (np.ceil (float (row[2]) / 30))

                # For some reason, I'm not considering avg montage
                if (self.montage[row[5]+row[4]][1] == 'avg'):
                    continue

                file_temp = []

                #Store file_no, start_time, end_time, montage number and montage signal
                file_temp.append (file_no)
                file_temp.append (str (np.fmod (float (row[2]), 30)))
                file_temp.append (str (np.fmod (float (row[3]), 30)))
                file_temp.append (self.montage[row[5] + row[4]])
                self.time_info.append ([file_no, file_temp[1], file_temp[2]])

                self.files_mont.append (self.montage[row[5]+row[4]])

                file_temp.append (self.read_edf_files (file_no, self.montage[row[5] + row[4]]))

                # self.file contains all montage information
                self.file.append (file_temp)
     
                # Put it in a dictionary
                if file_no not in self.montage_info:
                    self.montage_info[file_no] = []
                self.montage_info[file_no].append (file_temp[1:])

    def read_edf_files (self, file_no, mot_signals):
        """ Read given EDF file and return all signals. """

        # Convert file number to file name
        path = self.convert_to_file_name (file_no)

        # Read file
        f = pyedflib.EdfReader(path)

        # Get sampling frequencies
        samp_freq = f.getSampleFrequencies()[0]

        # Allocate two arrays for two electrode signal
        sigbufs = np.zeros((2, f.getNSamples()[0]))

        # Get all available electrode names
        self.label = f.getSignalLabels ()

        # Read first electrode signal
        sigbufs[0, :] = f.readSignal(self.label.index (mot_signals[0]))

        # Read second electrode signal
        if (mot_signals[1] != 'avg'):
            sigbufs[1, :] = f.readSignal(self.label.index (mot_signals[1]))
        else:
            sigbufs[1, :] = self.get_avg (f)

        f._close ()

        # Montage (difference in electrode signals)
        sig = sigbufs[0, :] - sigbufs[1, :]

        # Take only first 30seconds
        sig = sig[0: 30 * samp_freq]

        # Frequency normalize
        sig = self.frequency_normalize_256 (sig, samp_freq)

        return (sig)

    def convert_to_file_name (self, file_no):
        """ Convert the file no. to edf file name."""

        if (file_no / 100 < 1):
            if (file_no / 10 < 1):
                a = '00' + str(int (file_no))
            else:
                a = '0' + str(int (file_no))
        else:
            a = str (int (file_no))

        path = self.path_edf + '/s3_' + a + '.edf'

        return (path)

    def frequency_normalize_256 (self, sig, samp_freq):
        """ Upsample or downsample the signal to 256 Hz.""" 

        if (samp_freq > self.samp_freq):
            r = int (samp_freq / 256)
            sig = scipy.signal.decimate (sig, r, zero_phase=True)
            samp_freq = int (samp_freq / int (r))

        if (samp_freq < 256):
            sig = sig[0:samp_freq * 30]
            (r1, r2) = (samp_freq / 256).as_integer_ratio ()
            sig = scipy.signal.resample_poly (sig, r2, r1)

        return (sig)

    def get_avg (self, f):
        """ Get average of all the signals """

        n = f.signals_in_file
        temp = np.zeros ((len (self.mont_list), f.getNSamples ()[0]))
        j = 0

        for i in self.mont_list:
            temp[j, :] = f.readSignal (self.label.index (i))
            j = j + 1

        M = np.dot (np.ones ((1, len (temp))), temp)
        M = M / n
        return (M)

    def get_normal_signal (self, num_samples=1, sample_num=-1, random=False, truncate=False, montage=['Cz', 'C4']):
        """ Get signal which doesn't contain yellow-box. """

        if truncate:
            samples = np.zeros ((num_samples, 100))
        else:
            samples = np.zeros ((num_samples, 30 * 256))

        if random:
            rand.shuffle(self.file)

        row = 0;

        if sample_num != -1:
            for file_no in range (len(self.edf_files)):
                file_no += 1
                if file_no not in self.montage_info:
                    row += 1;
                    # Return "Row"th sample
                    if row == sample_num:
                        if truncate:
                            start = int (np.floor (np.random.rand()) * (250 * 30 - 100))
                            end = start + 100
                            samples[0] = self.read_edf_files (file_no, montage)[start : end]
                        else:
                            samples[0] = self.read_edf_files (file_no, montage)
                        break

        else:
            for file_no in range (len(self.edf_files)):
                file_no += 1
                if file_no not in self.montage_info:
                    if truncate:
                        start = int (np.floor (np.random.rand()) * (250 * 30 - 100))
                        end = start + 100
                        samples[row] = self.read_edf_files (file_no, montage)[start : end]
                    else:
                        samples[row] = self.read_edf_files (file_no, montage)
                    row += 1;

                if (row >= num_samples):
                    break

        return (samples)

    def get_yellow_signals (self, num_samples=1, sample_num=-1, random=False, truncate=False):
        """ Get signal which contains yellow-box. """

        if truncate:
            samples = np.zeros ((num_samples, 100))
        else:
            samples = np.zeros ((num_samples, 30 * 256))

        # Number of samples
        if num_samples == "ALL":
            num_samples = len (self.file)

        details = []
        row = 0;

        # Return specific sample
        if sample_num != -1:
            f = self.file[sample_num]
            if truncate:
                start = int (np.floor (float (f[1]) * 250))
                end = start + 100
                samples[0] = f[-1][start:end]
            else:
                samples[0] = f[-1]
            details.append (f[0:-1])

        # Return N-number of samples
        else:
            # Shuffle if random is true
            if random:
                rand.shuffle(self.file)

            for i, f in enumerate (self.file):
                if truncate:
                    start = int (np.floor (float (f[1]) * 250))
                    end = start + 100
                    samples[i] = f[-1][start:end]
                else:
                    samples[i] = f[-1]
                details.append (f[0:-1])
                row += 1;

                if (row >= num_samples):
                    break

        return ((samples, details))

def plot_eeg (num=3, only_eeg=False, show=False, save=False):
    """ Plot eeg signals containing yellow box.
    Input:
    num: Number of different signals needed.
    """
    (M, details) = edf_convert().get_yellow_signals (num, random=False)
    print (details)
    for i in range (len(M[:, 1])):
        start = int (np.floor (float (details[i][1]) * 250))
        end = int (np.ceil (float (details[i][2]) * 250))

        x = np.array ([start, end])
        m1 = max (M[i, int (np.ceil (start)) : int (np.ceil (end))])
        m2 = min (M[i, int (np.ceil (start)) : int (np.ceil (end))])
        y1 = np.array ([m1, m1])
        x2 = np.array ([start, start, end, end])
        y2 = np.array ([m1, m2, m2, m1])
        plt.figure (facecolor='b')

        if (only_eeg == True):
            plt.plot (M[i, start : end])
            plt.title ("Plot of yellow box. File_no: {0}, start_time: {1:0.2f}s, end_time: {2:0.2f}s".format (details[i][0], float (details[i][1]), float (details[i][2])))
        else:
            plt.plot (M[i], label="eeg signal", color="blue")
            plt.plot (x, y1, color="red")
            plt.plot (x2, y2, label="yellow box", color="red")
            plt.legend ()

        if (save):
            plt.savefig ("/home/ravi/Class/ANN/Takehome2/Output/Plots/eeg_" + str(i) + ".pdf", bbox_inches="tight")
        if show:
            plt.show()

def main ():
    plot_eeg ("ALL", only_eeg=True)

if (__name__=='__main__'):
    main()
