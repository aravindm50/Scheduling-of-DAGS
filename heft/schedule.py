
from collections import OrderedDict, namedtuple
from itertools import chain, permutations
from heft.util import reverse_dict

import numpy as np
import copy as cp

Event = namedtuple('Event', 'task start end')


LOGFILE = "heft_la.txt"
H_LOG = None

def initializeLog():
    global H_LOG
    global LOGFILE

    H_LOG = open(LOGFILE, 'w')

def logger(log_str):
    global H_LOG

def deinitialize():
    H_LOG.close()

class DAG(object):
    def __init__(self, dag):
        self.succ = dag['succ']
        self.cost = dag['cost']
        self.deadline = dag['deadline']
        self.comm_delay = dag['comm_delay']

        self.agents = dag['cost'].keys()
        self.tasks = dag['succ'].keys()
        self.sort = None

        self.wbar = np.mean(list(dag['cost'].values()), axis=0)
        self.rank = OrderedDict()
        self.prec = reverse_dict(self.succ)

        # Allocate constants
        self.schedule_order = {agent: [] for agent in self.agents}
        self.task_on = dict()
        
    def get_wbar(self, task):
        return self.wbar[task-1]

    def get_cbar(self, ni, nj):
        # Get number of resources
        n = len(self.agents)

        if n == 1:
            return 0

        cPairs = [self.comm_cost(ni, nj, a1, a2) for a1, a2 in permutations(self.agents, 2)]

        return np.mean(cPairs)

    def comm_cost(self, ni, nj, p1, p2):
        if p1 == p2:
            return 0
        elif (ni, nj) in self.comm_delay:
            return self.comm_delay[(ni, nj)]
        elif (ni, '*') in self.comm_delay:
            return self.comm_delay[(ni, '*')]

        return 0

    def comp_cost(self, task, agent):
        return self.cost[agent][task-1]

    def makespan(self):
        """ Finish time of last job """
        return max(ev_list[-1].end for ev_list in self.schedule_order.values() if ev_list)

    def get_rank(self, ni):
        raise NotImplementedError

    def calc_rank(self):
        # Get rank for each task
        for i in self.tasks:
            self.get_rank(i)

        # Round off rank values to int
        rTuple = [(j, round(r, 2)) for j, r in self.rank.items()]

        # Order tasks in ascending values of rank
        sortedTuple = sorted(rTuple, key=lambda x: (self.sort * x[1], x[0]))
        self.rank = OrderedDict([(k, v) for k, v in sortedTuple])

    # Allocate methods
    def addEvent(self, event, agent):
        # Add the task to the schedule order list
        self.schedule_order[agent].append(event)

        # Sort on start time
        self.schedule_order[agent] = sorted(self.schedule_order[agent], key=lambda e: e.start)

        # Update task_on list
        self.task_on[event.task] = agent

    def end_time(self, task):
        if task not in self.task_on:
            return 0

        agent = self.task_on[task]
        events = self.schedule_order[agent]

        # Endtime of job in list of events
        for e in events:
            if e.task == task:
                return e.end

    def start_time(self, task, agent):
        """ Earliest time that job can be executed on agent """
        duration = self.comp_cost(task, agent)

        # Schedule order in the given agent
        agent_orders = self.schedule_order[agent]
        if task in self.prec.keys() and (self.task_on.keys()):
        # Max time needed for all predecessor tasks to complete and move to the current agent
            prec_ready = max(self.end_time(p) + self.comm_cost(p, task, self.task_on[p], agent) \
                                                            for p in self.prec[task] if p in self.task_on.keys())# and not(None))
        else:
            prec_ready = 0

    # No tasks on current agent => can fit the task whenever it is ready to run
        if not self.schedule_order[agent]:
            return prec_ready

    # Try to fit task in between each pair of Events
        a = chain([Event(None, None, 0)], agent_orders[:-1])
        for e1, e2 in zip(a, agent_orders):
            earliest_start = max(prec_ready, e1.end)
            if e2.start - earliest_start > duration:
                return earliest_start

        # No gaps found => put it at the end, or whenever the task is ready
        return max(agent_orders[-1].end, prec_ready)

    def allocate(self, task):
        finish_time = lambda agent: self.start_time(task, agent) + self.comp_cost(task, agent)

        task_agent = min(self.agents, key=finish_time)
        task_start = self.start_time(task, task_agent)
        task_end = finish_time(task_agent)

        logger("Minimum EFT for {} is {} which is in resource {}\n".format(task, task_end, task_agent))

        ev = Event(task, task_start, task_end)

        # Ordered list of tasks
        self.addEvent(ev, task_agent)

    # Schedule : method to perform DAG scheuling
    def Schedule(self):
        self.calc_rank()

        for task in self.rank.keys():
            logger("Task allocation for {}\n".format(task))
            self.allocate(task)

        return self.schedule_order

class HEFT(DAG):
    def __init__(self, dag):
        DAG.__init__(self, dag)
        self.sort = -1

    def get_rank(self, ni):
        if ni in self.rank:
            return self.rank[ni]

        if ni in self.succ and self.succ[ni]:
            self.rank[ni] = self.get_wbar(ni) +\
                     max(self.get_cbar(ni, nj) + self.get_rank(nj) for nj in self.succ[ni])
            logger("rank of {} is {}\n".format(ni, self.rank[ni]))
        else:
            self.rank[ni] = self.get_wbar(ni)
            logger("rank of {} (end node) is {}\n".format(ni, self.rank[ni]))

        return self.rank[ni]

class HEFT_LA(DAG):
    def __init__(self, dag):
        DAG.__init__(self, dag)
        self.sort = -1

    def get_rank(self, ni):
        if ni in self.rank:
            return self.rank[ni]

        if ni in self.succ and self.succ[ni]:
            self.rank[ni] = self.get_wbar(ni) +\
                     max(self.get_cbar(ni, nj) + self.get_rank(nj) for nj in self.succ[ni])
            logger("rank of {} is {}\n".format(ni, self.rank[ni]))
        else:
            self.rank[ni] = self.get_wbar(ni)
            logger("rank of {} (end node) is {}\n".format(ni, self.rank[ni]))

        return self.rank[ni]

    def child_allocate(self, task): 
        finish_time = lambda agent: self.start_time(task, agent) + self.comp_cost(task, agent)
        task_agent = min(self.agents, key=finish_time)
        return finish_time(task_agent)

    def finish_time(self, task, agent):
        return self.start_time(task, agent) + self.comp_cost(task, agent)

    def allocate(self, task):
        finish_time_task = lambda agent: self.finish_time(task, agent)
        task_agent = None

        child_task = [t for t in self.rank.keys() if t in self.succ[task]]

        if child_task:
            # Save current state
            current_schedule_order = cp.deepcopy(self.schedule_order)
            current_task_on = cp.deepcopy(self.task_on.copy())

            child_end_time = {}

            for agent in self.agents:
                # Allocate the current task to current agent
                start = self.start_time(task, agent)
                end = self.finish_time(task, agent)
                self.addEvent(Event(task, start, end), agent)

                child_end_time[agent] = max([self.child_allocate(chld) for chld in child_task])

                # Restore previous state                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                self.schedule_order = cp.deepcopy(current_schedule_order)
                self.task_on = cp.deepcopy(current_task_on)

            minVal = min(child_end_time.values())
            task_agent = min(child_end_time, key=child_end_time.get)
            logger("Minimum EFT for {}'s children is {}".format(task, minVal))
            logger("{}'s children has minimum EFT when is sheduled to resource {}\n".format(task, task_agent))
        else:
            task_agent = min(self.agents, key=finish_time_task)

        task_start = self.start_time(task, task_agent)
        task_end = self.finish_time(task, task_agent)

        if not child_task:
            logger("No children for {}. Minimum EFT for {} is {} which is in resource {}\n".format(task, task, task_end, task_agent))

        ev = Event(task, task_start, task_end)

        # Add the task to the schedule order list
        self.addEvent(ev, task_agent)
    def get_next_independent_task(self, task):
        index = None
        for i,t in enumerate(self.rank.keys()):
            if task == t :
                index = i
                break  
            
        rank = list(self.rank.keys())
            
        for t in rank[index:]:
            index+=1
            if index < len(self.rank):              
                it = self.rank[index]
                if it in self.succ[task]:
                    continue
                break
            #index = None
            break
        task_1 = rank[index] if  index else None
       # self.ind_task = rank[index] if  index else None
        return task_1
        
   
    

class HEFT_LAP(DAG):
    def __init__(self, dag):
        DAG.__init__(self, dag)
        self.sort = -1

    def get_rank(self, ni):
        if ni in self.rank:
            return self.rank[ni]

        if ni in self.succ and self.succ[ni]:
            self.rank[ni] = self.get_wbar(ni) +\
                     max(self.get_cbar(ni, nj) + self.get_rank(nj) for nj in self.succ[ni])
        else:
            self.rank[ni] = self.get_wbar(ni)

        return self.rank[ni]

    def child_allocate(self, task):
        finish_time = lambda agent: self.start_time(task, agent) + self.comp_cost(task, agent)
        task_agent = min(self.agents, key=finish_time)
        return finish_time(task_agent)

    def finish_time(self, task, agent):
        if task == 1:
            return self.comp_cost(task,agent)
        else:
            return self.start_time(task, agent) + self.comp_cost(task, agent)

    def get_next_independent_task(self, task):
        index = None
        for i,t in enumerate(self.rank.keys()):
            if task == t :
                index = i
                break  
            
        rank = list(self.rank.keys())
            
        for t in rank[index:]:
            index+=1
            if index < len(self.rank):              
                it = self.rank[index]
                if it in self.succ[task]:
                    continue
                break
            index = None
            break
        task_1 = rank[index] if  index else None
       # self.ind_task = rank[index] if  index else None
        return task_1

    def allocate(self, task):
        finish_time_task = lambda agent: self.finish_time(task, agent)
        task_agent = None

        task_1 = self.get_next_independent_task(task)
        
        child_task = [t for t in self.rank.keys() if t in self.succ[task]]
        
        if child_task:
            # Save current state
            current_schedule_order = cp.deepcopy(self.schedule_order)
            current_task_on = cp.deepcopy(self.task_on.copy())

            child_end_time = {}
            for agent in self.agents:
                # Allocate the current task to current agent
                start = self.start_time(task, agent)
                end = self.finish_time(task, agent)
                self.addEvent(Event(task, start, end), agent)

                child_end_time[agent] = max([self.child_allocate(chld) for chld in child_task])

                # Restore previous state
                self.schedule_order = cp.deepcopy(current_schedule_order)
                self.task_on = cp.deepcopy(current_task_on)
                task_agent = min(child_end_time, key=child_end_time.get)
            if task_1 == None:
                child_task1 = None
            else:
                child_task1= [t1 for t1 in self.rank.keys() if t1 in self.succ[task_1]]
                if child_task1:
                    current_schedule_order = cp.deepcopy(self.schedule_order)
                    current_task_on = cp.deepcopy(self.task_on.copy())
        
                    child_end_time1 = {}
                    for agent in self.agents:
                        start1 = self.start_time(task_1, agent)
                        end1 = self.finish_time(task_1, agent)
                        self.addEvent(Event(task_1, start1, end1), agent)  
                        child_end_time1[agent] = max([self.child_allocate(chld1) for chld1 in child_task1])
        
                        # Restore previous state
                                       
                        task_agent1 = min(child_end_time1, key=child_end_time1.get)
                        if child_end_time[agent] > child_end_time1[agent]:
                            task_agent = min(child_end_time1, key=child_end_time1.get)
                        self.schedule_order = cp.deepcopy(current_schedule_order)
                        self.task_on = cp.deepcopy(current_task_on)    
                        
        else:            
            task_agent = min(self.agents, key=finish_time_task)
            
        
        task_start = self.start_time(task, task_agent)
        task_end = self.finish_time(task, task_agent)
        
        ev = Event(task, task_start, task_end)

        # Add the task to the schedule order list
        self.addEvent(ev, task_agent)
   