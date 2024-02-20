from artiq.experiment import *
from UM_localexp import LocalExperiment
from funcs_kernel import KernelSupportFunctions as funcs_kernel
import numpy as np
import time
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

'''
measure the rabi pi time of the zeeman qubit in different displacement
implemented using a localExp function
aim to provide tools for optical addressing
reference figure: arxiv1510.05618 Integrated optical addressing of an ion qubit Fig3.(d)
'''

class Ion_Experiment(LocalExperiment, EnvExperiment):
    """Pi time vs Dis"""

    def _build(self):
        # ion position
        self.setattr_argument("tickle_power_percentage", NumberValue(1, ndecimals=3, step=1e-3))
        self.setattr_argument("tickle_hardware_att", NumberValue(-6, ndecimals=3, step=1e-3, unit="dB"))
        self.setattr_argument("tickle_f_center", NumberValue(2500*kHz, ndecimals=3, step=1e-3, unit="kHz"))
        self.setattr_argument("tickle_f_full_span", NumberValue(800*kHz, ndecimals=3, step=1e-3, unit="kHz"))
        self.setattr_argument("tickle_f_step", NumberValue(20*kHz, ndecimals=3, step=1e-3, unit="kHz"))

        self.setattr_argument("shift_axis", EnumerationValue(["X", "Y", "Z"], default="X"))
        self.setattr_argument("shift_center", NumberValue(0.0, ndecimals=3, step=1e-3))
        self.setattr_argument("shift_span", NumberValue(2.0, ndecimals=3, step=1e-3))
        self.setattr_argument("shift_step", NumberValue(0.1, ndecimals=3, step=1e-3))
    
        # pi time
        self.setattr_argument("t_qubit_start", NumberValue(2*us, ndecimals=3, unit="us"))
        self.setattr_argument("t_qubit_stop", NumberValue(60*us, ndecimals=3, unit="us"))
        self.setattr_argument("t_step", NumberValue(5*us, ndecimals=3, unit="us"))
        self.setattr_argument("trials", NumberValue(50, ndecimals=0, step=1))

    def pause_scheduler(self):
        self.scheduler.pause()

    def _prepare(self):
        
        # Solve for pre-set hardware attenuation factor (-3dB -> 1/2, -10dB -> 1/10, -20dB -> 1/100, etc...)
        atten_factor = 1 / np.power(10, -self.tickle_hardware_att/10)              

        # Factoring in the HW attenuator, what software amplitude acheives the desired power percentage?   
        self.a_tickle = (self.const.a_trap_rf / atten_factor) * (self.tickle_power_percentage / 100)

        # Create and broadcast sweep array
        f_start = self.tickle_f_center - self.tickle_f_full_span/2
        f_stop = self.tickle_f_center + self.tickle_f_full_span/2
        self.f_array = np.arange(f_start, f_stop, self.tickle_f_step)
        self.set_dataset("f_array", self.f_array, broadcast=True)

        # Broadcast which axis is getting swept currently
        self.set_dataset("shift_axis", self.shift_axis, broadcast=True)

        # Array of shift factor, with broadcast
        self.shift_values = np.arange(self.shift_center - self.shift_span/2, self.shift_center + self.shift_span/2, self.shift_step)
        self.set_dataset("shift_values", self.shift_values, broadcast=True)

        # Probe frequencies, in mu, without offsets nor clock frequency - so same for every trial
        # f_array will be modified later and used for run(), so f_sweep is sent to dataset for applet plotting

        self.t_array = np.arange(self.t_qubit_start, self.t_qubit_stop + self.t_step, self.t_step)
        self.set_dataset("t_array", self.t_array, broadcast=True)

        # Creates array of offsets to add to probe frequencies, and prepares 2D data storage array for freqs, probabilities
        self.carrier_offsets = []
        self.p_data_out = np.full((5, len(self.t_array)), 1.01, dtype=float)
        self.t_data_out = np.full((5, len(self.t_array)), 0., dtype=float)
        
        for i in range(1, 6): 
            dataset_name = "detection.f" + str(i) + "_MHz"
            offset = self.get_dataset(dataset_name)
            self.carrier_offsets.append(offset)
        
        self.set_dataset("t_data_out", self.t_data_out, broadcast=True)
        self.set_dataset("p_data_out", self.p_data_out, broadcast=True)
        
        # Array for PMT readings
        self.sweep_points = len(self.t_array)
        self.total_counts = np.zeros(self.sweep_points,dtype=float) # Number of bright states for each frequency
        self.bright_prob = np.full(self.sweep_points,1.01,dtype=float)       
        self.set_dataset("bright_prob", self.bright_prob, broadcast=True)

        # Set up dataset for realtime histogram readouts
        self.pmt_hist = np.full(self.const.pmt_max_counts,0)
        self.set_dataset("pmt_hist", self.pmt_hist, broadcast=True)

        self.f_carrier = self.get_dataset("f.carrier")
        self.f_center = self.f_carrier

        # Arrays for Clocking data
        self.f_carrier_array = np.full(len(self.t_array), self.const.f_carrier, dtype=float)
        self.t_elapsed = np.zeros(len(self.t_array), dtype=float)
        self.set_dataset("f_carrier_array", self.f_carrier_array, broadcast=True)
        self.set_dataset("t_elapsed", self.t_elapsed, broadcast=True)

        self.f_accumulator = np.full((self.const.pmt_trials, len(self.t_array)), self.const.f_carrier, dtype=float)
        self.t_accumulator = np.zeros((self.const.pmt_trials, len(self.t_array)), dtype=float)
        self.set_dataset("f_accumulator", self.f_accumulator, broadcast=True)
        self.set_dataset("t_accumulator", self.t_accumulator, broadcast=True)

        self.set_dataset("qubit_spectroscopy_fit", self.bright_prob, broadcast=True)
    
    @funcs_kernel
    def run(self):
        self.initializeDevices()
        self.initRF()
        
        for s in range(len(self.shift_values)):
            
            if (self.shift_axis == "X"):
                self.writeElectrodes(self.shift_values[s], self.const.v_y_shim_factor, self.const.v_z_shim_factor)
            elif (self.shift_axis == "Y"):
                self.writeElectrodes(self.const.v_x_shim_factor, self.shift_values[s], self.const.v_z_shim_factor)
            elif (self.shift_axis == "Z"):
                self.writeElectrodes(self.const.v_x_shim_factor, self.const.v_y_shim_factor, self.shift_values[s])
            print(self.shift_values[s])
            self.core.break_realtime()

            t0 = now_mu()

            for offset_index in range(len(self.carrier_offsets)):

                offset = self.carrier_offsets[offset_index]
                
                self.clear_data()

                self.core.break_realtime()

                ##############################
                ######### PRE-CLOCK ##########
                ##############################

                for i in range(len(self.t_elapsed)):

                    new_freq_cl1 = self.f_center
                    new_freq_cl1 = self.clockPeak(new_freq_cl1, 0.01, 1)

                    clock1_time = now_mu()
                    t1s = self.core.mu_to_seconds(clock1_time - t0)

                    # Update guess for next loop
                    self.f_center = new_freq_cl1

                    # Broadcast results for realtime monitoring
                    self.mutate_dataset('f_carrier_array', i, new_freq_cl1)
                    self.mutate_dataset('t_elapsed', i, t1s)



                ###################################
                ######### Sweep Spectrum ##########
                ###################################

                for trial in range(self.trials):

                    self.pmt_hist = np.full(self.const.pmt_max_counts,0)
                    self.core.break_realtime()

                    for i in range(len(self.t_array)):

                        # Perform one clocking cycle
                        new_freq_cl1 = self.f_center
                        new_freq_cl1 = self.clockPeak(new_freq_cl1, 0.01, 1)
                        
                        clock1_time = now_mu()
                        t1s = self.core.mu_to_seconds(clock1_time - t0)
                        
                        self.f_center = new_freq_cl1
                        self.f_carrier_array[i] = new_freq_cl1
                        self.t_elapsed[i] = t1s

                        self.mutate_dataset('f_carrier_array', i, new_freq_cl1)
                        self.mutate_dataset('t_elapsed', i, t1s)

                        # Prepare next probe frequency
                        self.qubit.sw.off()
                        self.qubit.set(self.f_center*MHz + offset*MHz, 1.0, self.const.a_qubit)
                        
                        # Disable quench/doppler
                        self.quench.sw.off()
                        self.doppler.sw.off()

                        delay(10*us)
                        
                        # Pulse probe for interrogation time
                        self.qubit.sw.pulse(self.t_array[i])
                        
                        # Perform PMT Exposure
                        single_count = self.detectAndCount()

                        delay(100*us)
                        
                        # Quench and cool
                        self.dopplerRecool()
                        
                        # Update histogram
                        if single_count < self.const.pmt_max_counts:
                            self.pmt_hist[single_count] += 1
                            
                        # Calculate bright prob
                        if single_count > self.const.pmt_state_threshold:
                            self.total_counts[i] += 1

                    self.set_dataset("f.carrier", self.f_center, broadcast=True, persist=True)
                    self.set_dataset("pmt_hist", self.pmt_hist, broadcast=True)
                    self.mutate_dataset('f_accumulator', trial, self.f_carrier_array)
                    self.mutate_dataset('t_accumulator', trial, self.t_elapsed)

                    for i in range(len(self.bright_prob)):
                        self.bright_prob[i] = self.total_counts[i] / float(trial+1)
                        self.p_data_out[offset_index][i] = self.bright_prob[i]
                    
                    self.set_dataset("bright_prob", self.bright_prob, broadcast=True)
                    self.set_dataset("p_data_out", self.p_data_out, broadcast=True)

                    self.dopplerRecool()
                    print("COMPLETED: [ Autosearch Pi Times ]")

                    # VVVVV ~ ~ ~ ION LOSS AND RELOADING ~ ~ ~ VVVVV

                    if (not self.checkForIon()):
                        # Set to redo the trial
                        i-=1

                        # Reset DAC values to loading scheme
                        self.resetElectrodes()
                        
                        # Attempt to load ion
                        while(not self.checkForIon()):
                            self.attemptLoad()
                            delay(500*ms)
                            self.pause_scheduler()
                            self.core.break_realtime()

                        # Restore DAC values    
                        if (self.shift_axis == "X"):
                            self.writeElectrodes(self.shift_values[s], self.const.v_y_shim_factor, self.const.v_z_shim_factor)
                        elif (self.shift_axis == "Y"):
                            self.writeElectrodes(self.const.v_x_shim_factor, self.shift_values[s], self.const.v_z_shim_factor)
                        elif (self.shift_axis == "Z"):
                            self.writeElectrodes(self.const.v_x_shim_factor, self.const.v_y_shim_factor, self.shift_values[s])
                        self.core.break_realtime()

        self.resetElectrodes()
        print("COMPLETED: [ Rabi Freq Dis ]")