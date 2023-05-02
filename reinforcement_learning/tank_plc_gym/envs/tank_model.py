import numpy as np
import matplotlib.pyplot as plt

# make class tank


class WaterTank:
    chamber_area = 10.5  # in m^2
    max_height = 6  # in meters
    qin = 100  # liters per second
    # tank_level = 50  # level of the tank in percentage (0..100)
    input_flow_rate = 0  # input flow rate after the valve in liters per second
    # add initialize method

    def __init__(self, dt, valve_cmd, qout, qout_pattern, tank_level):
        self.dt = dt  # in seconds
        self.valve_cmd = valve_cmd  # in percentage (0...1)
        self.qout = qout  # in liters per second (0 ...100)
        self.qout_next = qout  # experimental to model the future demand
        self.qout_avg = qout  # experimental to model the future avg demand
        # time series patter for qout
        self.qout_pattern = np.array(qout_pattern)
        self.step = 0
        # level of the tank in percentage (0..100)
        self.tank_level = tank_level
        self.max_tank_volume = (
            self.calculate_max_tank_volume()  # calculates max tank volume in liters
        )

        self.current_tank_volume = (
            self.calc_current_volume()  # calculate current tank volume in liters
        )

        self.low_sensor_1 = self.calculate_low_sensor()
        self.low_sensor_2 = self.calculate_low_sensor()
        self.height_sensor_1 = self.calculate_height_sensor()
        self.height_sensor_2 = self.calculate_height_sensor()
        self.volume_sensor = self.calculate_volume_sensor()

    def change_command(self, cmd):
        self.valve_cmd = cmd

    def change_demand(self, dem):
        self.qout = dem

    def calculate_max_tank_volume(self):
        chamber_volume = self.max_height * self.chamber_area  # Volume in m^3
        chamber_volume = chamber_volume * 10 ** 6  # convert m^3 to cm^3
        chamber_volume = chamber_volume / 1000  # convert cm^3 to liters
        return chamber_volume * 2

    def calc_in_fwr(self):
        FlowRate = self.qin * self.valve_cmd
        self.input_flow_rate = FlowRate

    def calc_current_volume(self):
        volume = (self.tank_level / 100) * self.max_tank_volume
        return volume

    def calculate_tank_level(self, Volume):
        new_volume = (
            Volume + (self.input_flow_rate - self.qout) * self.dt
        )  # calculate the volume in liters
        if new_volume <= 0:
            new_volume = 0
        if new_volume >= self.max_tank_volume:
            new_volume = self.max_tank_volume
        self.current_tank_volume = new_volume  # update the current Volume value
        new_lvl = (new_volume / self.max_tank_volume) * \
            100  # transform in percentage
        self.tank_level = new_lvl  # update the tank_level value
        return new_lvl  # return new LVL in percentage

    def calculate_low_sensor(self):

        res = self.current_tank_volume < 0.05 * self.max_tank_volume
        return res
        # implement here
        mu, sigma = 5, 5
        faults = np.random.normal(mu, sigma, 10)
        randN = np.random.randint(10)
        # calculate fault probability
        if self.current_tank_volume < 0.05 * self.max_tank_volume:

            if faults[randN] >= mu + 3 * sigma:
                # if there is an error reverse the sensor reading
                res = 0
            else:
                res = 1

        else:
            if faults[randN] >= 4 * sigma:
                res = 1
            else:
                res = 0

    def calculate_height_sensor(self):
        res = self.current_tank_volume > 0.95 * self.max_tank_volume
        return res
        mu, sigma = 5, 5
        faults = np.random.normal(mu, sigma, 10)
        randN = np.random.randint(10)
        # calculate fault probability
        if self.current_tank_volume > 0.95 * self.max_tank_volume:

            if faults[randN] >= mu + 3 * sigma:
                # if there is an error reverse the sensor reading
                res = 0
            else:
                res = 1

        else:
            if faults[randN] >= 4 * sigma:
                res = 1
            else:
                res = 0

    def calculate_volume_sensor(self):
        sensor = (
            self.current_tank_volume / 2
        ) / self.chamber_area  # calculate current level in meters
        return sensor * 10 ** 2  # return level in centimeters

    def calculateAll(self):
        index = self.step % len(self.qout_pattern)
        self.qout = self.qout_pattern[index]

        self.qout_next = self.qout_pattern[(index+1) % len(self.qout_pattern)]

        # TODO add paramter for count of future demands to average
        if index + 10 > len(self.qout_pattern):
            avg = (np.concatenate(
                [self.qout_pattern[index:index+10], self.qout_pattern[:(index+10) % len(self.qout_pattern)]])).mean()
        else:
            avg = self.qout_pattern[index:index+10].mean()

        self.qout_avg = avg
        self.step += 1

        self.calc_in_fwr()
        self.calculate_tank_level(self.current_tank_volume)
        self.low_sensor_1 = self.calculate_low_sensor()
        self.low_sensor_2 = self.calculate_low_sensor()
        self.height_sensor_1 = self.calculate_height_sensor()
        self.height_sensor_2 = self.calculate_height_sensor()
        self.volume_sensor = self.calculate_volume_sensor()


def run():
    dt = 1

    bull = [i for i in range(20, 100)]
    bear = [i for i in range(100, 20, -1)]
    pattern = (bull + bear)
    tank = WaterTank(dt, 1, 0, pattern)

    Samples = 3600
    time = np.arange(Samples)
    time *= dt
    level = np.zeros(Samples).astype(float)
    level_centimeters = np.zeros(Samples).astype(float)
    volume = np.zeros(Samples).astype(float)

    for i in range(Samples):
        if i == 1500:  # change dynamically demand and command
            # tank.change_demand(100)
            tank.change_command(0)
        if i == 2500:  # change dynamically demand and command
            # tank.change_demand(100)
            tank.change_command(1)
        tank.calculateAll()
        level[i] = tank.tank_level
        level_centimeters[i] = tank.volume_sensor
        volume[i] = tank.current_tank_volume / 1000
        # print("Max tankVolume", tank.max_tank_volume)
        # print("current tankVolume", tank.current_tank_volume)
        # print("current tank level", tank.tank_level)
        # print("LowSensor1", tank.low_sensor_1)
        # print("LowSensor2", tank.low_sensor_2)
        # print("HighSensor1", tank.height_sensor_1)
        # print("HighSensor2", tank.height_sensor_2)
        # print("Volume Sensor", tank.volume_sensor)
    plt.figure(1)
    plt.plot(time, level)
    plt.xlabel("Seconds")
    plt.ylabel("Level in percents")
    plt.figure(2)
    plt.plot(time, volume)
    plt.xlabel("Seconds")
    plt.ylabel("Volume in m^3")
    plt.figure(3)
    plt.plot(time, level_centimeters)
    plt.xlabel("Seconds")
    plt.ylabel("Level in centimeters")
    plt.show()


if __name__ == "__main__":
    run()
