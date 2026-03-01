import math

def get_temperature(step, total_steps, T_start, T_end, schedule):
    t = step / total_steps

    if schedule == "linear":
        return T_start + (T_end - T_start) * t

    if schedule == "cos":
        return T_end + (T_start - T_end) * 0.5 * (1 + math.cos(math.pi * t))

    if schedule == "exp":
        return T_start * ((T_end / T_start) ** t)

    if schedule == "power":
        p = 4.0
        return T_start + (T_end - T_start) * (t ** p)
    
    if schedule == "linear_min1.0":
        return max(1.0, T_start + (T_end - T_start) * t)
    
    if schedule == "linear_MinEpoch3":
        step = 70312 * 3
        min = get_temperature(step, total_steps, T_start, T_end, "linear")
        return max(min, T_start + (T_end - T_start) * t)