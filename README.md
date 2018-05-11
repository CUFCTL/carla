DataSet:

| Map Name | Weather |  Time  | Weather ID | Archive Name |
|:--------:|:-------:|:------:|:----------:|:------------:|
|  Town01  |  Rainny |  Noon  |     56     |   Town01_56  |
|  Town02  |  Sunny  | Sunset |     89     |   Town02_89  |
|  Town01  |  Sunny  | Sunset |     89     |   Town01_89  |
|  Town01  |  Sunny  |  Noon  |     12     |   Town01_12  |
|  Town02  |  Sunny  |  Noon  |     12     |   Town02_12  |

Models:

| Map Name | Weather | Time   | Weather ID | Trails           | Noise      | Modle Version | Archive Name      | Available |
|----------|---------|--------|------------|------------------|------------|---------------|-------------------|:---------:|
| Town01   | Rainny  | Noon   | 56         | Curve            | High       | 1             | Town01_56_CH      |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Curve            | Low        | 1             | Town01_56_CL      |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Straight         | High       | 1             | Town01_56_SH      |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Straight         | Low        | 1             | Town01_56_SL      |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Curve            | High & Low | 1             | Town01_56_CHL     |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Straight         | High & Low | 1             | Town01_56_SHL     |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Curve & Straight | High       | 1             | Town01_56_CSH     |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Curve & Straight | Low        | 1             | Town01_56_CSL     |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Curve & Straight | High & Low | 1             | Town01_56_CSHL    |     ✔     |
| ---      | ---     | ---    | ---        | ---              | ---        | ---           | ---               |    ---    |
| Town02   | Sunny   | Sunset | 89         | Curve            | High       | 1             | Town02_89_CH      |     ✔     |
| Town02   | Sunny   | Sunset | 89         | Curve            | Low        | 1             | Town02_89_CL      |     ✔     |
| Town02   | Sunny   | Sunset | 89         | Straight         | High       | 1             | Town02_89_SH      |     ✔     |
| Town02   | Sunny   | Sunset | 89         | Straight         | Low        | 1             | Town02_89_SL      |     ✔     |
| Town02   | Sunny   | Sunset | 89         | Curve            | High & Low | 1             | Town02_89_CHL     |     ✔     |
| Town02   | Sunny   | Sunset | 89         | Straight         | High & Low | 1             | Town02_89_SHL     |     ✔     |
| Town02   | Sunny   | Sunset | 89         | Curve & Straight | High       | 1             | Town02_89_CSH     |     ✔     |
| Town02   | Sunny   | Sunset | 89         | Curve & Straight | Low        | 1             | Town02_89_CSL     |     ✔     |
| Town02   | Sunny   | Sunset | 89         | Curve & Straight | High & Low | 1             | Town02_89_CSHL    |     ✔     |
| ---      | ---     | ---    | ---        | ---              | ---        | ---           | ---               |    ---    |
| Town01   | Rainny  | Noon   | 56         | Curve            | High       | 2             | Town01_56_CH_V2   |     ✘     |
| Town01   | Rainny  | Noon   | 56         | Curve            | Low        | 2             | Town01_56_CL_V2   |     ✘     |
| Town01   | Rainny  | Noon   | 56         | Straight         | High       | 2             | Town01_56_SH_V2   |     ✘     |
| Town01   | Rainny  | Noon   | 56         | Straight         | Low        | 2             | Town01_56_SL_V2   |     ✘     |
| Town01   | Rainny  | Noon   | 56         | Curve            | High & Low | 2             | Town01_56_CHL_V2  |     ✘     |
| Town01   | Rainny  | Noon   | 56         | Straight         | High & Low | 2             | Town01_56_SHL_V2  |     ✘     |
| Town01   | Rainny  | Noon   | 56         | Curve & Straight | High       | 2             | Town01_56_CSH_V2  |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Curve & Straight | Low        | 2             | Town01_56_CSL_V2  |     ✔     |
| Town01   | Rainny  | Noon   | 56         | Curve & Straight | High & Low | 2             | Town01_56_CSHL_V2 |     ✔     |
| ---      | ---     | ---    | ---        | ---              | ---        | ---           | ---               |    ---    |
| Town01   | Sunny   | Sunset | 89         | Curve            | High       | 1             | Town01 _89_CH     |     …     |
| Town01   | Sunny   | Sunset | 89         | Curve            | Low        | 1             | Town01 _89_CL     |     …     |
| Town01   | Sunny   | Sunset | 89         | Straight         | High       | 1             | Town01_89_SH      |     ✘     |
| Town01   | Sunny   | Sunset | 89         | Straight         | Low        | 1             | Town01_89_SL      |     ✘     |
| Town01   | Sunny   | Sunset | 89         | Curve            | High & Low | 1             | Town01_89_CHL     |     …     |
| Town01   | Sunny   | Sunset | 89         | Straight         | High & Low | 1             | Town01_89_SHL     |     ✘     |
| Town01   | Sunny   | Sunset | 89         | Curve & Straight | High       | 1             | Town01_89_CSH     |     …     |
| Town01   | Sunny   | Sunset | 89         | Curve & Straight | Low        | 1             | Town01_89_CSL     |     …     |
| Town01   | Sunny   | Sunset | 89         | Curve & Straight | High & Low | 1             | Town01_89_CSHL    |     …     |
| ---      | ---     | ---    | ---        | ---              | ---        | ---           | ---               |    ---    |
| Town01   | Sunny   | Noon   | 12         | Curve            | High       | 1             | Town01_12_CH      |     ✘     |
| Town01   | Sunny   | Noon   | 12         | Curve            | Low        | 1             | Town01_12_CL      |     ✘     |
| Town01   | Sunny   | Noon   | 12         | Straight         | High       | 1             | Town01_12_SH      |     ✘     |
| Town01   | Sunny   | Noon   | 12         | Straight         | Low        | 1             | Town01_12_SL      |     ✘     |
| Town01   | Sunny   | Noon   | 12         | Curve            | High & Low | 1             | Town01_12_CHL     |     ✘     |
| Town01   | Sunny   | Noon   | 12         | Straight         | High & Low | 1             | Town01_12_SHL     |     ✘     |
| Town01   | Sunny   | Noon   | 12         | Curve & Straight | High       | 1             | Town01_12_CSH     |     …     |
| Town01   | Sunny   | Noon   | 12         | Curve & Straight | Low        | 1             | Town01_12_CSL     |     …     |
| Town01   | Sunny   | Noon   | 12         | Curve & Straight | High & Low | 1             | Town01_12_CSHL    |     ✘     |


