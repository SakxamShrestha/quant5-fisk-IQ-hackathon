# Heavy-duty delivery truck schedules (NLR, 2021)

Real-world operating schedules of three delivery fleets, copied unchanged from
`data/fleet-schedules/` of the National Laboratory of the Rockies (formerly NREL)
repository <https://github.com/NatLabRockies/hdev-depot-charging-2021>
(commit `61530df51f6cd7d64f014ebbf8714cef24c8ca70`). The schedules come from NREL's
Fleet DNA real-world fleet data. BSD-3-Clause license, see [`LICENSE.md`](LICENSE.md).

Cite: B. Borlaug, M. Muratori, M. Gilleran et al. (2021). Heavy-duty truck
electrification and the impacts of depot charging on electricity distribution systems.
*Nature Energy* 6, 673–682. doi:10.1038/s41560-021-00855-0.
Dataset: doi:10.7799/1787031 (<https://data.nlr.gov/submissions/162>).

## Files

Each fleet folder has two files. One row of `veh_op_days.csv` is one truck on one day
(a *truck-day*); there is no truck ID or date, so truck-days cannot be linked across days.

| file | columns |
|---|---|
| `veh_op_days.csv` | `veh_op_day_id`, `vmt` (miles driven), `time_on_shift_s`, `time_off_shift_s` |
| `veh_schedules.csv` | `veh_op_day_id`, `start_time`, `end_time` (HH:MM:SS, midnight to midnight), `total_time_s`, `on_shift` (1 = driving, 0 = parked at the depot), `vmt` |

## How this project uses them (`evqaoa/real_data.py`)

- Charging needed: miles × 1.8 kWh/mile, delivered at 100 kW, rounded up to whole hours.
  Both figures are the NLR study's own assumptions (`generate_load_profiles` in its code).
- A truck-day's evening is followed by a morning with the same schedule, as the NLR code assumes.

## SHA-256

```
5cd171575baabb61a5d1faf9fad5b58463b5b4fde9d55250c3f5eb3d79605ca1  fleet1-beverage-delivery/veh_op_days.csv
66ebc46729a40c4b71365c6df78a203c68098059b7e00eeec7c7f23fde2acb94  fleet1-beverage-delivery/veh_schedules.csv
f097dd653ce8b894c4d155c05e51571091571c225f1a872e9e2cf040d4807627  fleet2-warehouse-delivery/veh_op_days.csv
5fdecb232de748e6eca2b00a897f55cb55912fb14136e99e616fc4d935421842  fleet2-warehouse-delivery/veh_schedules.csv
6207fb821e865475280ad881ccfbd18bc15f6cfc187695a732f66c7444a952e8  fleet3-food-delivery/veh_op_days.csv
473e64af4ef981a8e3d33a298ef5db0f3cb3741ff46139a59f2c1c6dd9c8c18a  fleet3-food-delivery/veh_schedules.csv
```
