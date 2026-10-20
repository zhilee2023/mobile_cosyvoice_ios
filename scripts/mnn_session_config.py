"""MNN 3.6.1 Interpreter binding requires integer memory/power enums.

Unlike precision's legacy string compatibility, memory='low' is silently
ignored by getScheduleConfig in pymnn/src/util.h. Never pass that string.
"""


def cpu_session_config(precision="high",memory="normal",threads=4):
    precision_modes={"normal":0,"high":1,"low":2,"Low_BF16":3}
    memory_modes={"normal":0,"high":1,"low":2}
    if precision not in precision_modes or memory not in memory_modes or not 1<=threads<=8:
        raise ValueError("Unknown precision/memory mode or invalid thread count")
    return {"backend":"CPU","precision":precision_modes[precision],
            "memory":memory_modes[memory],"numThread":threads}
