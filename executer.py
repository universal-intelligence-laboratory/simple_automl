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
    
    def next_step(self,reports,num_of_process):
        solution = make_empty_solution()
        if num_of_process<self.max_num_of_process:
            settings = make_empty_settings()
            solution['start_list'].append(settings)
        return solution


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
    def __init__(self,mode='initial',max_exp = 2):
        if mode == 'initial':
            self.reports_history = {}
            self.settings = {}
            self.running_process = {}
            self.P = example_policy(max_exp)

    def phase_setting(self,experments_setting):
        return experments_setting
    
    def start_exp(self,experments_setting,traget_py_name = 'train.py'):
        # call train.py here
        string = self.phase_setting(experments_setting)
        #command = 'ping 192.168.1.1 '
        command = 'python ' + traget_py_name + string
        proc = subprocess.Popen(command, shell=True,stdout=subprocess.PIPE)
        self.running_process[proc.pid] = proc
        self.reports_history[proc.pid] = []
        # return pid 
        #time.sleep(20) #wait until build
        print("Start: "+command + " with PID: " + str(proc.pid))
        return proc.pid
    
    def react_with_reports(self,reports, pid):
        print(reports)
        self.reports_history[pid].append(reports)
        solution = self.P.next_step(reports,len(self.running_process))
                
        if solution['stop_list'] != []:
            for pid_to_kill in solution['stop_list']:
                self.stop_exp(pid_to_kill)
                
        if solution['start_list'] != []:
            for experments_setting_to_start in solution['start_list']:
                #if len(self.running_process) < self.max_exps:
                pid = self.start_exp(experments_setting_to_start)
                self.settings[pid] = experments_setting_to_start
                #else:
                #    return False
        return True
    
    def check_reports(self):
        if len(self.running_process) == 0:
            return True #done
        
        print("======running processes: ",len(self.running_process))
        now = deepcopy(self.running_process)
        for pid,process in now.items():
            #get lastest line by pid
            output = process.stdout
            report  = output.readline().strip().decode("utf-8").split('\n')[-1]
            print("Checking "+ str(pid) + " Report: "+report)
            if not self.react_with_reports(report,pid):
                #print("reacting might fail.\n")
                pass
                
                
        return False
            
        
    def stop_exp(self,pid):
        # shell kill pid
        a = os.kill(pid, signal.SIGKILL)
        print(a)
        self.running_exps.remove(pid)
        
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
    e = executer()
    e.start()
