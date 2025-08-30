import simpy

def global_pause_controller(env, pause_flag):
    """每5s触发一次暂停，持续0.5s"""
    while True:
        yield env.timeout(5)
        print(f"[{env.now}] Global pause starts")
        pause_flag['pause'] = True
        yield env.timeout(0.5)
        pause_flag['pause'] = False
        print(f"[{env.now}] Global pause ends")

def my_process(env, name, pause_flag):
    """模拟运行，每步检查是否需要暂停"""
    while True:
        if pause_flag['pause']:
            print(f"[{env.now}] {name} pauses")
            yield env.timeout(0.1)  # 持续等待直到全局暂停结束
        else:
            print(f"[{env.now}] {name} running")
            yield env.timeout(1)

env = simpy.Environment()
pause_flag = {'pause': False}

env.process(global_pause_controller(env, pause_flag))
env.process(my_process(env, "P1", pause_flag))
env.process(my_process(env, "P2", pause_flag))
env.process(my_process(env, "P3", pause_flag))

env.run(until=20)
