import random as rd

times = 100  
simulation = []

for i in range(times):
    U = rd.uniform(1, 100)  
    if 1 <= U <= 15: 
        simulation.append("0")
    elif 16 <= U <= 35:  
        simulation.append("1")
    elif 36 <= U <= 65:
        simulation.append("2")
    elif 66 <= U <= 85:
        simulation.append("3")
    elif 86 <= U <= 95:
        simulation.append("4")
    elif 96 <= U <= 100:
        simulation.append("5")

for times, sales in enumerate(simulation, start=1): 
    print(f"第 {times} 次 : {sales}")

