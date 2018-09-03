
import matplotlib.pyplot as plt
import numpy as np
from heft.schedule import *


# DAG Input 1
d1 = {
    'succ': {
        1: (2, 3, 4, 5, 6),
        2: (7, 8, 9, 10, 11),
        3: (7, 8, 9, 10, 11),
        4: (7, 8, 9, 10, 11),
        5: (),
        6: (),
        7: (),
        8: (),
        9: (),
        10:(),
        11:()
    },

    'cost': {
        'P1': [30, 14, 27, 36, 25, 29, 40, 37, 20, 34, 30],
        'P2': [44, 25, 31, 21, 32, 30, 42, 33, 33, 20, 35],
        'P3': [29, 23, 19, 40, 28, 26, 39, 43, 25, 28, 22]
    },

    'comm_delay': {
        (1, 2) : 19,
        (1, 3): 31,
        (1, 4): 45,
        (1, 5): 7,
        (1, 6): 22,
        (2, 7): 9,
        (2, 8): 9,
        (2, 9): 1,
        (2, 10):19,
        (2, 11):6,
        (3, 7): 17,
        (3, 8): 1,
        (3, 9): 10,
        (3, 10):14,
        (3, 11):2,
        (4, 7): 9,
        (4, 8): 19,
        (4, 9): 15,
        (4, 10):8,
        (4, 11):19,
    },

    'deadline': {
        5: 90,
        10: 130
    }
}

# DAG Input 2
d2 = {
    'succ': {
        1: (2, 3, 4, 5, 6),
        2: (7, 8),
        3: (7, 10),
        4: (7,),
        5: (),
        6: (8, 9, 10),
        7: (11,),
        8: (11,),
        9: (),
        10: (11,),
        11: ()
    },

    'cost': {
        'P1': [19, 28, 36, 15, 15, 30, 33, 12, 23, 13, 41],
        'P2': [41, 46, 34, 25, 21, 50, 35, 20, 34, 22, 68],
        'P3': [34, 20, 62, 37, 29, 54, 59, 21, 31, 24, 73]
    },

    'comm_delay': {
        (1, 2) : 31,
        (1, 3): 89,
        (1, 4): 80,
        (1, 6): 17,
        (1, 5): 31,
        (2, 7): 45,
        (2, 8): 59,
        (3, 7): 31,
        (3, 10): 14,
        (4, 7): 73,
        (6, 8): 41,
        (6, 9): 23,
        (6, 10): 33,
        (7, 11): 65,
        (8, 11): 11,
        (10, 11): 7,
    },

    'deadline': {
        5: 90,
        9: 100,
        11: 330
    }
}


initializeLog()


def printSchedule(d, sc,file):
    for ag, val in sorted(sc.items()):
        for ev in val:
            print (ag, '\t',ev.task, '\t', ev.start,'\t', ev.end,'\t',d.rank[ev.task])
            
            if ev.task in list(d.deadline.keys()):
                deadline = d.deadline[ev.task]
                print("Deadline of ",ev.task," = ",deadline)
                if deadline - ev.end < 0:
                    print("Deadline Missed for task",ev.task)
            file.write('%s %s %s %s %s \n'%(ag,ev.task,ev.start,ev.end,d.rank[ev.task]))           
    print('\nMakespan = ', d.makespan(), '\n')
    
# Scheduling Both DAG's with all the algorithms    
a = [d1,d2]  
for i  in [1]:
        heft_dag = HEFT(a[i-1])
        la_dag = HEFT_LA(a[i-1])
        lap_dag = HEFT_LAP(a[i-1])
        heft_sched = heft_dag.Schedule()
        la_sched = la_dag.Schedule()
        lap_sched = lap_dag.Schedule()
        file = open('Makespan_HEFT.txt','w')
        file1 = open('Makespan_HEFTLA.txt','w')
        file2 = open('Makespan_HEFTLAP.txt','w')
        printSchedule(heft_dag, heft_sched,file)
        printSchedule(la_dag,la_sched,file1)
        printSchedule(lap_dag,lap_sched,file2)
#       
deinitialize()
