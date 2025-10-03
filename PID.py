import matplotlib.pyplot as plt

# https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller

Kp = 0.5
Ki = 5.0
Kd = 0.5

previousError = 0
integral = 0
dt = 0.001

setpoint = 0
processVariable = -45

values = []
elapsedTime = 0

while 1:
    error = setpoint - processVariable
    proportional = error
    integral += error * dt
    derivative = (error - previousError) / dt
    previousError = error
    output = Kp * proportional + Ki * integral + Kd * derivative
    
    # print(f"processVariable: {processVariable:.6f} | output: {output:.6f}")
    processVariable += output * dt

    values.append(processVariable)
    elapsedTime += dt

    if abs(processVariable - setpoint) < 0.01 and sum([abs(value) for value in values[-1000:]]) / 1000 - setpoint < 0.01:
        break

print(f"elapsedTime: {elapsedTime:.3f}")
plt.plot(values)
plt.show()