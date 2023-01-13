# Scenario and MLC Output Encodings for Pylot

The details on the scenario and MLC output encodings for Pylot is presented below.

(NOTE: if you do not have a proper markdown viewer, you can use [this online viewer](https://dillinger.io))

## Scenario

Scenario is a vector of 7 parameters which are detailed in the following table:

|  Parameter   | Type  |                                                    Value Range                                                    |                     Description                      |
| :----------: | :---: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------: |
| time_of_day  | `int` |                                     `0 -> noon`, `1 -> sunset`, `2 -> night`                                      |           the position of the sun in Carla           |
|   weather    | `int` | `0-> clear`, `1 -> cloudy`, `2 -> wet`, `3 -> wet cloudy`, `4 -> medium rain`, `5 -> hard rain`, `6 -> soft rain` |                 the weather in Carla                 |
|  pedestrian  | `int` |                                                `0 -> 0`, `1 -> 18`                                                |    presence of pedestrians during the simulation.    |
|  road curve  | `int` |                           `0 -> straight`, `1 -> right`, `2 -> left`, `3 -> cross road`                           |                     type of road                     |
|   road ID    | `int` |                                                  {`0`, `1`, `2`}                                                  |      different parts of map (start/end points)       |
| road length  | `int` |                                    {`0`, `1`, `2`} (from shortest to longest)                                     |                  length of the road                  |
| driving task | `int` |                               `0 -> follow road`, `1 -> 1st exit`, `2 -> 2nd exit`                                | the driving task that the ego vehicle should perform |


## MLC Output (MLCO)

An MLCO is a vector comprised of an arbitrary number of trajectories (in the pylot case study, an MLCO is comprised of 2 trajectories).

$\mathit{mlco} = \langle trj_1, \dots, trj_n \rangle$

Whereas, each trajectory is a vector of 11 parameters as described in the below table:

|    Parameter    |  Type   |          Value Range          |                           Description                            |
| :-------------: | :-----: | :---------------------------: | :--------------------------------------------------------------: |
|     `label`     |  `int`  | `0 -> vehicle`, `1 -> person` |            the label assigned to a detected obstacle             |
|      `t0`       | `float` |     `[0 - sim_duration]`      |            the start time of an obstacle trajectory.             |
|      `t1`       | `float` |     `[0 - sim_duration]`      |             the end time of an obstacle trajectory.              |
| `bbox_t0_x_min` | `float` |          `[0, 750]`           | the x_min coordinate of the obstacle bounding box (bbox) at `t0` |
| `bbox_t0_y_min` | `float` |          `[0, 550]`           |               the y_min coordinate of bbox at `t0`               |
| `bbox_t0_x_max` | `float` |          `[50, 800]`          |               the x_max coordinate of bbox at `t0`               |
| `bbox_t0_y_max` | `float` |          `[50, 600]`          |               the y_max coordinate of bbox at `t0`               |
| `bbox_t1_x_min` | `float` |          `[0, 750]`           |               the x_min coordinate of bbox at `t1`               |
| `bbox_t1_y_min` | `float` |          `[0, 550]`           |               the y_min coordinate of bbox at `t1`               |
| `bbox_t1_x_max` | `float` |          `[50, 800]`          |               the x_max coordinate of bbox at `t1`               |
| `bbox_t1_y_max` | `float` |          `[50, 600]`          |               the y_max coordinate of bbox at `t1`               |


**Assumptions**:
- The coordinates of the bounding box is bound by the size of the camera frame which is `800*600`.
- The minimum size of the bounding box is 50 pixels.
- `sim_duration` is assumed to be `350` seconds for the experiments run on Pylot.