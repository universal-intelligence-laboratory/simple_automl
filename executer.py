# -*- coding: utf-8 -*-

#from policies import example_policy
import os
import signal
import subprocess
import sys
import time
from torchvision import datasets

class example_policy:
    def __init__(self, max_num_of_process):
        self.max_num_of_process = max_num_of_process
    
    def next_step(self,report,num_of_process,pid):
        #print("RRRRRRR",report)
        
        solution = make_empty_solution()
        
        # start many process even only one report fetch
        if num_of_process<self.max_num_of_process and not self.stop_policy():
            settings = make_empty_settings()
            solution['start_list'].append(settings)
            
        # if the report includes some value, kill it
        if ' 2' in report:
            solution['stop_list'].append(pid)
                    
        return solution

    def stop_policy(self):
        return False

def deepcopy(d):
    new_d = {}
    for key,value in d.items():
        new_d[key] = value
    return new_d
    
def make_empty_solution():
    solution = {}
    solution['start_list'] = []
    solution['stop_list'] = []
    return solution

def make_empty_settings():
    settings = ' '
    return settings

class executer:
    def __init__(self,max_exp,mode='initial'):
        if mode == 'initial':
            self.reports_history = {}
            self.settings = {}
            self.running_process = {}
            self.P = example_policy(max_exp)
        elif mode == 'restore':
            pass
            # TODO: restore training proceduce with pickles

    def phase_setting(self,experments_setting):
        return experments_setting
    
    def start_exp(self,experments_setting,traget_py_name = 'train.py'):
        # call train.py here
        string = self.phase_setting(experments_setting)
        #command = 'ping 192.168.1.1 ' # for debugging
        command = 'python ' + traget_py_name + string
        proc = subprocess.Popen(command, shell=True,stdout=subprocess.PIPE)
        self.running_process[proc.pid] = proc
        self.reports_history[proc.pid] = []
        print("Start: "+command + " with PID: " + str(proc.pid))
        return proc.pid
    
    def react_with_reports(self,reports, pid):
        self.reports_history[pid].append(reports)
        solution = self.P.next_step(reports,len(self.running_process),pid)
                
        if solution['stop_list'] != []:
            for pid_to_kill in solution['stop_list']:
                self.stop_exp(pid_to_kill)
                
        if solution['start_list'] != []:
            for experments_setting_to_start in solution['start_list']:
                pid = self.start_exp(experments_setting_to_start)
                self.settings[pid] = experments_setting_to_start
                
    def check_reports(self):
        if len(self.running_process) == 0:
            return True 
        
        print("======running processes: ",len(self.running_process))
        now = deepcopy(self.running_process)
        for pid,process in now.items():
            #get lastest line by pid
            output = process.stdout
            report  = output.readline().strip().decode("utf-8").split('\n')[-1]
            self.react_with_reports(report,pid)
        return False
            
        
    def stop_exp(self,pid):
        # shell kill pid
        process_to_be_kill = self.running_process.pop(pid)
        a = os.kill(pid, signal.SIGKILL)
        print('KILL: ',pid)
        
    def start(self):
        #
        init_run = make_empty_settings()
        self.start_exp(init_run)
        is_done = False
        while (True):
            time.sleep(5)
            is_done = self.check_reports()
            if is_done:
                break
                

if __name__ == '__main__':
    datasets.MNIST('./data', train=True, download=True)
    e = executer(max_exp=2)
    e.start()
